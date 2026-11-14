// src/models/deepseek41/model.cpp - loads a DeepSeek-V4.1 pack onto the GPU.
#include "basalt/core/memory_audit.hpp"
#include "basalt/core/device.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/core/vmm.hpp"
#include "models/deepseek41/prompt_host.hpp"
#include "basalt/core/expert_source.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/log.hpp"
#include "models/deepseek41/model.hpp"

#include "basalt/artifact/dequant.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "models/deepseek41/attn_batch.hpp"
#include "models/deepseek41/context_policy.hpp"
#include "models/deepseek41/session.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <sstream>

namespace basalt::models::deepseek41 {

DsModel::~DsModel() {
    {
        std::lock_guard<std::mutex> lock(pool_mu_);
        pool_quit_ = true;
    }
    pool_cv_.notify_all();
    if (pool_refill_.joinable()) pool_refill_.join();
    shared_tier_.tier.reset();
    tier_profile_.report();
    overlay_.clear();
    dense_.reset();
    if (embd_fd_ >= 0) platform::close_fd(embd_fd_);
    if (experts_fd_ >= 0) platform::close_fd(experts_fd_);
}

std::string DsModel::expert_hotspots(const std::string& action) {
    std::string result = hotspots_.command(action, geo().n_layers, geo().n_expert);
    if (result.find("\"error\"") != std::string::npos) return result;
    result.pop_back();
    std::ostringstream out;
    out << result << ",\"top_k\":" << geo().n_used << ",\"formats\":[";
    for (size_t l = 0; l < pack_.experts.size(); ++l) {
        const auto& e = pack_.experts[l];
        if (l) out << ',';
        out << "[\"" << ggml_type_name((ggml_type) e.gu_type) << "\",\""
            << ggml_type_name((ggml_type) e.d_type) << "\"," << e.blob_bytes << ']';
    }
    out << "],\"overrides\":[";
    bool comma = false;
    if (has_overlay()) for (int l = 0; l < geo().n_layers; ++l) for (int e = 0; e < geo().n_expert; ++e) {
        const auto* r = replacement(l, e);
        if (!r) continue;
        if (comma) out << ',';
        comma = true;
        out << '[' << l << ',' << e << ",\"" << ggml_type_name((ggml_type) r->layout.gu_type)
            << "\",\"" << ggml_type_name((ggml_type) r->layout.d_type) << "\"," << r->layout.bytes << ']';
    }
    out << "]}";
    return out.str();
}

bool DsModel::load_overlay(const std::string& path, std::string& err) {
    core::ModelFile file;
    if (!file.open(path, err)) return false;
    auto layout = file.range("overlay.layout"), header = file.range("overlay.base-header"), data = file.range("experts");
    if (!layout.bytes || layout.bytes > 4*1024*1024 || header.bytes != core::kHeaderBytes || !data.bytes) {
        err = "missing or invalid overlay section"; return false;
    }
    std::vector<uint8_t> ref, actual(core::kHeaderBytes), text;
    if (!file.read("overlay.base-header", ref, err) ||
        !core::read_range_at(core::FileRange(path_, 0, core::kHeaderBytes), 0, actual.data(), actual.size(), err) ||
        !file.read("overlay.layout", text, err)) return false;
    if (ref != actual) { err = "overlay belongs to a different base pack"; return false; }
    std::istringstream in(std::string(text.begin(), text.end()));
    std::string magic; in >> magic;
    if (magic != "BASALT_DS_EXPERTS_V1") {err = "unknown overlay format"; return false;}
    const auto& g = geo();
    overlay_.resize((size_t) g.n_layers*g.n_expert);
    std::vector<std::pair<size_t,uint64_t>> entries;
    uint64_t cursor = 0;
    while (in >> std::ws && !in.eof()) {
        int l,e,gu,d; uint64_t off,bytes;
        if (!(in >> l >> e >> gu >> d >> off >> bytes) || l<0 || l>=g.n_layers || e<0 || e>=g.n_expert ||
            off!=cursor || !bytes || bytes>data.bytes-cursor ||
            !((gu==19 || gu==16 || gu==17 || gu==18 || gu==21 || gu==22) &&
              (d==10 || d==16 || d==17 || d==18 || d==19 || d==21 || d==22 || d==39)) ||
            !kernels::native_expert_supported(gu,d,g.n_embd,g.n_ff_exp)) {
            err="invalid overlay entry or unsupported format"; return false;
        }
        const size_t key=(size_t)l*g.n_expert+e;
        auto& ent=overlay_[key];
        if (ent.layout.bytes) {err="duplicate overlay expert"; return false;}
        ent.layout=kernels::native_expert_layout(gu,d,g.n_embd,g.n_ff_exp);
        ent.layout.swiglu_limit = g.clamp_exp[l];
        if (!kernels::cpu::native_fmt(gu,d,g.n_embd,g.n_ff_exp,ent.cpu,err,g.clamp_exp[l])) return false;
        if (ent.layout.bytes!=bytes) {err="overlay expert size does not match quantization"; return false;}
        ent.offset = off;
        entries.emplace_back(key,off); cursor+=bytes;
    }
    if (cursor!=data.bytes || entries.empty()) {err="incomplete overlay layout"; return false;}
    overlay_bytes_=cursor;
    overlay_source_ = data;
    const uint64_t grain = expert_host_vmm_granularity();
    overlay_ram_budget_ = 0;
    for (const auto& entry : entries) {
        const auto bytes = overlay_[entry.first].layout.bytes;
        overlay_ram_budget_ += grain ? (bytes+grain-1)/grain*grain : bytes;
    }
    basalt::log::info("basalt deepseek41","expert overlay: %zu replacements, %.2f GiB weights; loading after tier placement",
                    entries.size(),cursor/1073741824.);
    return true;
}

bool DsModel::read_initial_replacement(int layer, int expert, uint8_t* to, std::string& err) const {
    const auto* v = replacement(layer,expert);
    if (!v || !to) { err = "invalid initial replacement destination"; return false; }
    return core::read_range_at(overlay_source_,v->offset,to,v->layout.bytes,err);
}

uint8_t* DsModel::prepare_replacement_ram(int layer, int expert) const {
    const auto* v = replacement(layer,expert);
    if (!v) return nullptr;
    if (v->ram) return v->ram->data;
    auto ram = acquire_expert_ram(v->layout.bytes);
    if (!ram) return nullptr;
    auto* data = ram->data;
    v->ram = std::move(ram);
    return data;
}

void DsModel::publish_replacement_ram(int layer, int expert) const {
    const auto* v = replacement(layer,expert);
    if (v && v->ram) v->ram->ready = true;
}

void DsModel::release_replacement_ram(int layer, int expert) const {
    if (const auto* v = replacement(layer,expert)) recycle_expert_ram(std::move(v->ram));
}

std::unique_ptr<ExpertRam> DsModel::acquire_expert_ram(uint64_t bytes) const {
    TierScope scope(tier_profile_,TierProfile::Acquire);
    std::unique_lock<std::mutex> lock(pool_mu_);
    if (auto ram = swap_pool_.take(bytes)) {
        if (tier_profile_.enabled) ++tier_profile_.calls[TierProfile::RamPoolHit];
        if (swap_pool_.usable(pool_block_) < pool_target_) pool_cv_.notify_one();
        return ram;
    }
    core::detail::HostMemory hm;
    if (!core::detail::host_available_memory(hm)) return nullptr;
    auto room = core::ram_tier_room(hm.available,platform::ram_reserve(),true);
    if (bytes > room && !swap_pool_.empty()) {
        swap_pool_.clear();
        if (!core::detail::host_available_memory(hm)) return nullptr;
        room = core::ram_tier_room(hm.available,platform::ram_reserve(),true);
    }
    if (bytes > room) return nullptr;
    lock.unlock();
    auto ram = std::make_unique<ExpertRam>();
    if (tier_profile_.enabled) ++tier_profile_.calls[TierProfile::RamPoolMiss];
    if (!ram->allocate(bytes) || ram->allocated() > room) return nullptr;
    return ram;
}

void DsModel::trim_expert_ram_pool(uint64_t limit) const {
    std::lock_guard<std::mutex> lock(pool_mu_);
    swap_pool_.trim(limit);
}

void DsModel::recycle_expert_ram(std::unique_ptr<ExpertRam> ram) const {
    TierScope scope(tier_profile_,TierProfile::Recycle);
    std::unique_lock<std::mutex> lock(pool_mu_);
    if (!swap_pool_.put(ram)) {
        if (ram && tier_profile_.enabled) ++tier_profile_.calls[TierProfile::RamPoolFree];
        lock.unlock();
        ram.reset(); return;
    }
}

void DsModel::keep_expert_ram(size_t blocks, uint64_t bytes, int device) const {
    if (tiers_.diag_tier_timing)
        basalt::log::info("basalt deepseek41", "expert RAM refill: %zu blocks of at least %llu bytes", blocks,
                          (unsigned long long) bytes);
    {
        std::lock_guard<std::mutex> lock(pool_mu_);
        pool_target_ = blocks;
        pool_block_ = bytes;
    }
    if (!pool_refill_.joinable()) pool_refill_ = std::thread([this, device] { refill_expert_ram(device); });
    pool_cv_.notify_one();
}

void DsModel::refill_expert_ram(int device) const {
    cudaSetDevice(device);
    std::unique_lock<std::mutex> lock(pool_mu_);
    for (;;) {
        pool_cv_.wait(lock, [&] { return pool_quit_ || swap_pool_.usable(pool_block_) < pool_target_; });
        if (pool_quit_) return;
        const uint64_t bytes = pool_block_;
        lock.unlock();
        // the floor a decode-time allocation keeps
        std::unique_ptr<ExpertRam> ram;
        core::detail::HostMemory hm;
        if (core::detail::host_available_memory(hm)) {
            const uint64_t room = core::ram_tier_room(hm.available,platform::ram_reserve(),true);
            ram = std::make_unique<ExpertRam>();
            if (bytes > room || !ram->allocate(bytes) || ram->allocated() > room || ram->registered() < bytes) ram.reset();
        }
        lock.lock();
        if (!ram) {
            pool_target_ = 0;   // RAM is short: a promotion pins its own place, or finds none, as without the refill
            continue;
        }
        if (pool_quit_) { lock.unlock(); return; }
        // Foreground recycling may have filled the pool while this block was being pinned.
        if (swap_pool_.usable(pool_block_) >= pool_target_ || !swap_pool_.put(ram)) {
            if (swap_pool_.usable(pool_block_) < pool_target_) pool_target_ = 0;
            lock.unlock();
            ram.reset();
            lock.lock();
        }
    }
}

bool DsModel::load(const std::string& path, std::string& err) {
    if (!open_pack(path, pack_, err)) return false;
    path_ = path;
    cudaGetDevice(&device_);
    const Geometry& g = pack_.geo;
    desc_.arch = "deepseek41";
    desc_.n_layers = g.n_layers;
    desc_.n_embd = g.n_embd;
    desc_.n_vocab = g.n_vocab;
    desc_.n_expert = g.n_expert;
    desc_.n_expert_used = g.n_used;
    desc_.n_ff_exp = g.n_ff_exp;
    desc_.n_expert_shared = g.n_shared;
    desc_.context_length = g.context_length;
    for (const ExpertLayer& e : pack_.experts) {
        desc_.expert_gate_up_type.push_back(e.gu_type);
        desc_.expert_down_type.push_back(e.d_type);
    }

    embd_fd_ = platform::open_fd_utf8(path);
    experts_fd_ = platform::open_fd_utf8(path);
    if (embd_fd_ < 0 || experts_fd_ < 0) {
        err = "cannot open " + path;
        return false;
    }

    core::MemoryAuditScope weight_audit("model-weights");
    // dense tensors to the device; the token embedding and the Engram table stay in the file
    std::vector<uint8_t> staging;
    std::vector<float> wide;
    uint64_t total = 0;
    uint64_t dense_bytes = 0, arena_bytes = 0;
    for (const auto& [name, t] : pack_.tensors) {
        if (name == "token_embd.weight" || name == "per_layer_token_embd.weight") continue;
        const bool wide_gate = t.type == 30 && name.find("ffn_gate_inp.weight") != std::string::npos;
        const uint64_t limit = std::numeric_limits<uint64_t>::max();
        if (wide_gate && t.bytes > limit / 2) { err = "dense tensor size overflow: " + name; return false; }
        const uint64_t bytes = wide_gate ? t.bytes * 2 : t.bytes;
        const uint64_t padding = (256 - arena_bytes % 256) % 256;
        if (padding > limit - arena_bytes || bytes > limit - arena_bytes - padding) {
            err = "dense arena size overflow: " + name;
            return false;
        }
        arena_bytes += padding + bytes;
        dense_bytes += bytes;
    }
    try {
        // One owner, with aligned views for every tensor; no per-tensor allocation slack.
        dense_ = std::make_unique<core::DeviceArena>(arena_bytes, device_);
    } catch (const core::CudaError& e) {
        err = std::string("cannot allocate dense weights: ") + e.what();
        return false;
    }
    weight_audit.requested(arena_bytes);
    core::LoadProgress::announce_total(dense_bytes + pack_.experts_range.size());
    core::LoadProgress::plan(core::LoadProgress::cuda(0), dense_bytes);
    core::LoadProgress dense_load(core::LoadProgress::cuda(0), dense_bytes);
    for (const auto& [name, t] : pack_.tensors) {
        if (name == "token_embd.weight" || name == "per_layer_token_embd.weight") continue;
        DevTensor d;
        d.type = t.type;
        d.bytes = t.bytes;
        d.shape = t.shape;
        // the router reads its gate as F32; BF16 widens exactly (the compressor and indexer kernels read BF16)
        const bool widen = t.type == 30 && name.find("ffn_gate_inp.weight") != std::string::npos;
        if (widen) {
            d.type = 0;
            d.bytes = t.bytes * 2;
        }
        try {
            d.ptr = dense_->alloc(d.bytes);
        } catch (const core::CudaError& e) {
            err = "cannot place dense tensor " + name + ": " + e.what();
            return false;
        }
        constexpr uint64_t kChunk = 64ull << 20;
        staging.resize((size_t) std::min<uint64_t>(t.bytes, kChunk));
        for (uint64_t off = 0; off < t.bytes; off += kChunk) {
            const uint64_t n = std::min<uint64_t>(kChunk, t.bytes - off);
            if (!platform::pread_all(embd_fd_, staging.data(), n, t.offset + off)) {
                err = "short read of " + name;
                return false;
            }
            if (widen) {
                wide.resize((size_t) n / 2);
                dequantize_bf16(staging.data(), wide.data(), (int) (n / 2));
            }
            if (cudaMemcpy((char*) d.ptr + off * (widen ? 2 : 1), widen ? (const void*) wide.data() : staging.data(),
                           widen ? n * 2 : n, cudaMemcpyHostToDevice) != cudaSuccess) {
                err = "upload of " + name + " failed";
                return false;
            }
            dense_load.add(widen ? n * 2 : n);
        }
        total += d.bytes;
        dev_.emplace(name, std::move(d));
    }
    dense_load.finish();
    weight_audit.finish();
    basalt::log::info("basalt deepseek41", "%zu dense tensors, %.2f GiB on the GPU", dev_.size(),
                 (double) total / (1024.0 * 1024.0 * 1024.0));

    if (pack_.engram.present) {
        engram_ = std::make_unique<ngram::PleReader>();
        const EngramInfo& e = pack_.engram;
        // Keep recently fetched rows across requests instead of relying on the filesystem to retain their pages.
        constexpr uint64_t kEngramCacheRows = 1ull << 20;
        if (!engram_->open(path, e.file_offset, e.rows, 64, kEngramCacheRows, err, e.row_bytes)) return false;
    }
    return true;
}

int64_t DsModel::ctxp_default() { return ctxp::kLegacyContext; }

static std::vector<int> cache_ratios(const Geometry& g) {
    std::vector<int> ratios;
    for (int il = 0; il < g.n_layers; ++il)
        if (ctxp::owns_cache(il, g.compress_ratio[(size_t) il])) ratios.push_back(g.compress_ratio[(size_t) il]);
    return ratios;
}

uint64_t DsModel::context_bytes() const { return ctxp::context_bytes(context(), cache_ratios(pack_.geo), cache_format_); }

int64_t DsModel::keep_positions(int64_t backed) const {
    return prefillp::keep_positions(backed, cache_ratios(pack_.geo), cache_format_);
}

prefillp::Dims DsModel::prompt_dims() const {
    const Geometry& g = pack_.geo;
    prefillp::Dims d;
    d.n_embd = g.n_embd;
    d.n_used = g.n_used;
    d.n_expert = g.n_expert;
    d.n_ff_exp = g.n_ff_exp;
    d.engram_cols = (int64_t) pack_.engram.hash.n_cols() * 256;
    d.replacement_blob_bytes = overlay_max_bytes();
    d.reuse_stage = core::vmm::tcc_mode();
    for (const ExpertLayer& e : pack_.experts) {
        d.expert_blob_bytes = std::max<uint64_t>(d.expert_blob_bytes, d.reuse_stage ? staged_stride(e) : e.blob_bytes);
    }
    d.attn_fixed_bytes = attn_batch_fixed_bytes(cache_format_);
    d.attn_token_bytes = attn_batch_token_bytes();
    return d;
}

bool DsModel::resolve_context(int64_t requested, std::string& err) {
    const Geometry& g = pack_.geo;
    const std::vector<int> ratios = cache_ratios(g);
    uint64_t free_b = 0, total_b = 0;
    std::string why;
    if (!core::device_memory(-1, free_b, total_b, why)) {
        err = "deepseek41: cannot query the free device memory: " + why;
        return false;
    }
    const ctxp::Plan p = ctxp::plan_context(requested, g.context_length, free_b, ratios, cache_format_);
    if (!p.error.empty()) {
        err = "deepseek41: " + p.error;
        return false;
    }
    ctx_ = p.context;
    const uint64_t full = ctxp::context_bytes(ctx_, ratios, cache_format_);
    core::LoadProgress::kv(core::LoadProgress::cuda(0), ctxp::context_bytes(0, ratios, cache_format_), ctx_, "", full);
    basalt::log::info("basalt deepseek41", "context %lld tokens (caches of a full context: %.2f GiB, --kv %s)%s%s", (long long) ctx_,
                 (double) full / (1024.0 * 1024.0 * 1024.0), cache_format_ == CacheFormat::Fp8 ? "fp8" : "fp16",
                 p.note.empty() ? "" : "; ", p.note.c_str());
    return true;
}

const DevTensor* DsModel::tensor(const std::string& name) const {
    auto it = dev_.find(name);
    return it == dev_.end() ? nullptr : &it->second;
}

bool DsModel::embedding_row(int32_t token, float* out, std::string& err) const {
    const TensorRef* t = pack_.find("token_embd.weight");
    if (!t || (t->type != 14 && t->type != 13 && t->type != 8) || t->shape.size() != 2) {
        err = "token_embd.weight is missing or not Q5_K, Q6_K or Q8_0";
        return false;
    }
    const uint64_t n_embd = t->shape[0];
    if (token < 0 || (uint64_t) token >= t->shape[1]) {
        err = "token id out of range";
        return false;
    }
    const bool q8 = t->type == 8;
    const uint64_t q5_block = 176;
    const uint64_t row_bytes = q8 ? n_embd / 32 * 34 : n_embd / 256 * (t->type == 13 ? q5_block : 210);
    std::vector<uint8_t> row(row_bytes);
    if (!platform::pread_all(embd_fd_, row.data(), row_bytes, t->offset + (uint64_t) token * row_bytes)) {
        err = "cannot read the embedding row";
        return false;
    }
    if (q8) {
        for (uint64_t b = 0; b < n_embd / 32; ++b) dequantize_q8_0(row.data() + b * 34, out + b * 32);
    } else if (t->type == 13) {
        for (uint64_t b = 0; b < n_embd / 256; ++b) dequantize_q5_K(row.data() + b * q5_block, out + b * 256);
    } else {
        for (uint64_t b = 0; b < n_embd / 256; ++b) dequantize_q6_K(row.data() + b * 210, out + b * 256);
    }
    return true;
}

void DsModel::warm_token_table() const {
    if (const TensorRef* t = pack_.find("token_embd.weight")) platform::advise_willneed(embd_fd_, t->offset, t->bytes);
}

bool DsModel::open_mmproj(vision::Mmproj& mp, std::string& err) const {
    if (pack_.vision.empty() || pack_.vision.bytes == 0) {
        err = "this pack carries no vision projector (convert with --mmproj)";
        return false;
    }
    return mp.open(pack_.vision, err, "deepseek41v");
}

model::Result DsModel::make_session(std::unique_ptr<model::ModelSession>& out) {
    std::string err;
    out = make_session_impl(*this, err);
    if (!out) return {model::Status::Error, err};
    return {};
}

}  // namespace basalt::models::deepseek41
