// src/models/deepseek41/model.hpp - a loaded DeepSeek-V4.1 pack: dense tensors on the GPU, the experts section and the
// Engram table as file ranges.
#pragma once

#include "models/deepseek41/tier_profile.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/core/ram_load.hpp"
#include "basalt/core/device.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/ngram/ple_reader.hpp"
#include "basalt/models/deepseek41/vision.hpp"
#include "basalt/vision/mmproj.hpp"
#include "models/deepseek41/pack.hpp"
#include "models/deepseek41/prefill_plan.hpp"
#include "models/deepseek41/expert_ram.hpp"

#include <map>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::deepseek41 {

struct DevTensor {
    void* ptr = nullptr;
    int type = 0;
    uint64_t bytes = 0;
    std::vector<uint64_t> shape;
};

/// y[c][r] = W[r] . x[c] for `ncols` float columns, quantized through `scratch_q8_1`, whatever W's type.
inline void dense_f32(const DevTensor& w, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols,
                      void* stream) {
    kernels::native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    kernels::native_mmvq(w.type, w.ptr, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

class ExpertTier;
class DsparkDrafter;

class DsModel final : public model::LoadedModel {
public:
    ~DsModel() override;
    /// Reads the pack and uploads every dense tensor except the token embedding (read per token from the file).
    bool load(const std::string& path, std::string& err);
    bool load_overlay(const std::string& path, std::string& err);
    struct Replacement {
        kernels::NativeExpertLayout layout;
        kernels::cpu::NativeFmt cpu;
        uint64_t offset = 0;
        mutable std::unique_ptr<ExpertRam> ram;
        const uint8_t* host() const { return ram && ram->ready ? ram->data : nullptr; }
    };
    TierProfile& tier_profile() const { return tier_profile_; }
    bool has_overlay() const { return overlay_bytes_ != 0; }
    uint64_t initial_overlay_ram_budget() const { return overlay_ram_budget_; }
    bool read_initial_replacement(int layer, int expert, uint8_t* to, std::string& err) const;
    uint8_t* prepare_replacement_ram(int layer, int expert) const;
    void publish_replacement_ram(int layer, int expert) const;
    void release_replacement_ram(int layer, int expert) const;
    std::unique_ptr<ExpertRam> acquire_expert_ram(uint64_t bytes) const;
    void recycle_expert_ram(std::unique_ptr<ExpertRam> ram) const;
    uint64_t overlay_max_bytes(int layer = -1) const {
        uint64_t bytes = 0;
        if (has_overlay())
            for (int l = 0; l < geo().n_layers; ++l) if (layer < 0 || l == layer)
                for (int e = 0; e < geo().n_expert; ++e)
                    bytes = std::max<uint64_t>(bytes, overlay_[(size_t) l * geo().n_expert + e].layout.bytes);
        return bytes;
    }
    const Replacement* replacement(int l, int e) const {
        if (!has_overlay()) return nullptr;
        const auto& v = overlay_[(size_t) l * geo().n_expert + e];
        return v.layout.bytes ? &v : nullptr;
    }

    const model::ModelDesc& desc() const override { return desc_; }
    int64_t context() const override { return ctx_ ? ctx_ : ctxp_default(); }
    /// Fixes the context from --max-context (`set`) or the model's maximum, bounded by the free device memory. Call
    /// once after load(), before a session is made. False with `err` when the request cannot be honoured.
    bool resolve_context(int64_t requested, std::string& err);
    /// The caches' format (--kv); set before resolve_context.
    CacheFormat cache_format() const { return cache_format_; }
    void set_cache_format(CacheFormat f) { cache_format_ = f; }
    model::Result make_session(std::unique_ptr<model::ModelSession>& out) override;

    void memory_usage(core::MemoryUsage& usage) const override {
        if (!usage.claim(this)) return;
        for (const auto& v : overlay_) if (v.ram) {
            usage.host["resident_experts"] += v.ram->allocated();
            usage.host["pinned_experts"] += v.ram->registered();
            usage.host["locked_experts"] += v.ram->pin.locked_bytes;
        }
        usage.host["expert_swap_workspace"] += swap_pool_bytes_;
        if (engram_) usage.host["engram_reader"] += engram_->allocated_bytes();
        if (dense_) usage.device(device_, "weights", dense_->capacity());
    }
    const Pack& pack() const { return pack_; }
    const Geometry& geo() const { return pack_.geo; }
    /// Device tensor `blk.<layer>.<suffix>` or a global name; null when absent.
    const DevTensor* tensor(const std::string& name) const;
    const DevTensor* tensor(int layer, const char* suffix) const { return tensor("blk." + std::to_string(layer) + "." + suffix); }
    /// One token-embedding row, dequantised into `out` (n_embd floats).
    bool embedding_row(int32_t token, float* out, std::string& err) const;
    /// Asks the OS file cache for the whole token embedding: a window reads a row per token, the drafter's anchor
    /// first, and a row the cache lacks waits for the disk.
    void warm_token_table() const;
    ngram::PleReader& engram_reader() const { return *engram_; }
    int experts_fd() const { return experts_fd_; }
    const std::string& path() const { return path_; }
    /// Diagnostics (off by default): replay recorded expert outputs; dump per-layer tensors.
    const std::string& diag_dump() const { return diag_dump_; }
    void set_diag_dump(std::string dump) { diag_dump_ = std::move(dump); }
    /// Fixed expert-tier sizes per layer for repeatable runs (-1 = sized from free memory).
    int diag_tier_ram() const { return diag_tier_ram_; }
    int diag_tier_vram() const { return diag_tier_vram_; }
    void set_diag_tier(int ram, int vram) { diag_tier_ram_ = ram; diag_tier_vram_ = vram; }
    /// The expert tier and the pinned rows it shares with a second card: one per model, whatever the sessions.
    struct SharedTier {
        std::shared_ptr<ExpertTier> tier;
        float* h_x = nullptr;
        float* h_rows = nullptr;
    };
    SharedTier& shared_tier() const { return shared_tier_; }
    /// The drafter the sessions share, while one holds it.
    std::weak_ptr<DsparkDrafter>& shared_drafter() const { return shared_drafter_; }
    /// The sessions that will hold a context each.
    int sessions() const { return tiers_.sessions > 0 ? tiers_.sessions : 1; }
    /// What the flags ask of the expert tiers.
    const model::LoadRequest& tiers() const { return tiers_; }
    void set_tiers(const model::LoadRequest& req) { tiers_ = req; }
    /// The CUDA device of the second card's expert tier, or -1.
    int peer_device() const { return peer_device_; }
    void set_peer_device(int d) { peer_device_ = d; }
    /// The CUDA device the dense tensors live on.
    int device() const { return device_; }
    /// The sizes the prompt path's workspaces depend on.
    prefillp::Dims prompt_dims() const;
    /// Device bytes of the compressed caches of a full context.
    uint64_t context_bytes() const;
    /// Positions a new sequence keeps backed of `backed` (prefill_plan.hpp's keep_positions with this model's caches).
    int64_t keep_positions(int64_t backed) const;
    /// Opens the image projector: the diagnostic standalone file when set, else the pack's vision section.
    /// False with `err` set when neither exists.
    bool open_mmproj(vision::Mmproj& mp, std::string& err) const;
    /// The image encoders' cuBLAS handle, opened by the expert tier at load when images are on.
    VisionBlas& vision_blas() const { return vision_blas_; }

private:
    CacheFormat cache_format_ = CacheFormat::Fp8;
    mutable TierProfile tier_profile_;
    std::vector<Replacement> overlay_;
    mutable std::vector<std::unique_ptr<ExpertRam>> swap_pool_;
    mutable uint64_t swap_pool_bytes_ = 0;
    size_t overlay_bytes_ = 0;
    uint64_t overlay_ram_budget_ = 0;
    core::FileRange overlay_source_;
    Pack pack_;
    model::ModelDesc desc_;
    std::map<std::string, DevTensor> dev_;
    std::unique_ptr<core::DeviceArena> dense_;
    std::string diag_dump_, path_;
    int diag_tier_ram_ = -1, diag_tier_vram_ = -1;
    int peer_device_ = -1;
    int device_ = 0;
    model::LoadRequest tiers_;
    mutable SharedTier shared_tier_;
    mutable std::weak_ptr<DsparkDrafter> shared_drafter_;
    mutable VisionBlas vision_blas_;
    int64_t ctx_ = 0;
    static int64_t ctxp_default();
    int embd_fd_ = -1;
    int experts_fd_ = -1;
    std::unique_ptr<ngram::PleReader> engram_;
};

}  // namespace basalt::models::deepseek41
