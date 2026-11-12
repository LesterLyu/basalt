// src/models/deepseek41/tier.cpp - the DeepSeek-V4.1 routed-expert tiers.
#include "basalt/log.hpp"
#include "models/deepseek41/tier.hpp"
#include "basalt/platform/compiler.hpp"

#include "basalt/core/device.hpp"
#include "basalt/core/memory_audit.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/core/spin_wait.hpp"

#include "basalt/core/load_progress.hpp"
#include "basalt/core/vmm.hpp"
#include <optional>
#include <stdexcept>

#include "basalt/core/expert_source.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "models/deepseek41/model.hpp"
#include "models/deepseek41/prefill_plan.hpp"
#include "models/deepseek41/tier_slots.hpp"
#include "models/deepseek41/ram_plan.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <deque>
#include <exception>
#include <thread>
#include <vector>

namespace basalt::models::deepseek41 {

void ExpertTier::memory_usage(core::MemoryUsage& usage) const {
    if (!usage.claim(this)) return;
    usage.device(m_.device(), "experts", cache_.mapped_bytes() + ext_.mapped_bytes());
    usage.device(m_.device(), "scratch", fetch_stride_*(uint64_t)fetch_slots_*kFetchRings);
    if (peer_) {
        usage.device(peer_->device(), "expert_helper", peer_->cache().mapped_bytes() - peer_workspace_bytes_);
        usage.device(peer_->device(), "scratch", peer_->device_scratch_bytes());
    }
    if (share_) share_->memory_usage(usage);
    for (size_t l = 0; l < ram_.size(); ++l)
        if (ram_[l]) usage.host["resident_experts"] += host_vmm_bytes_.empty()
            ? core::ram_padded(ram_bytes_[l]) : host_vmm_bytes_[l];
    for (uint64_t n : host_vmm_bytes_) usage.host["pinned_experts"] += n;
    for (const auto& pin : ram_pins_) {
        usage.host["pinned_experts"] += pin.registered_bytes;
        usage.host["locked_experts"] += pin.locked_bytes;
    }
    for (size_t l = 0; l < extra_ram_.size(); ++l) for (const auto& r : extra_ram_[l]) if (r) {
        usage.host["resident_experts"] += r->allocated();
        usage.host["pinned_experts"] += r->registered();
        usage.host["locked_experts"] += r->pin.locked_bytes;
    }
    host_.memory_usage(usage);
    usage.host["decode_workspace"] += kernels::cpu::kNativeActBytes * kMaxWindow * kMaxK;
}

namespace {

constexpr size_t kMaxPending = 128;              // copies claimed a window
constexpr size_t kEvents = 2 * kMaxPending;      // a window's and the last one's copies
constexpr uint64_t kPrereadBytes = 12ull << 30;  // file-backed experts read ahead into the page cache

// The usual routing order of this model's experts: the RAM and VRAM tiers start from the head of it.
constexpr int kRankLayers = 40, kRankExperts = 384;
}  // namespace

// The helper's async copies can block the submitting Windows thread. Queue the submissions,
// then fence the refill stream before publishing the new expert homes.
struct ExpertTier::PeerSubmitter {
    struct Move { int slot; uint8_t* out; const uint8_t* in; uint64_t off, bytes; };
    core::PeerExperts& peer;
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<Move> queue;
    bool stop = false, busy = false;
    std::exception_ptr error;
    std::thread thread;

    explicit PeerSubmitter(core::PeerExperts& p) : peer(p), thread([this] { run(); }) {}
    ~PeerSubmitter() {
        { std::lock_guard lock(mutex); stop = true; }
        cv.notify_all();
        thread.join();
    }
    void run() {
        std::unique_lock lock(mutex);
        for (;;) {
            cv.wait(lock, [&] { return stop || !queue.empty(); });
            if (queue.empty()) return;
            Move job = queue.front();
            queue.pop_front();
            busy = true;
            lock.unlock();
            std::exception_ptr failure;
            try {
                if (job.out) peer.slot_out(job.slot,job.out,job.off,job.bytes);
                else peer.slot_in(job.slot,job.in,job.off,job.bytes);
            }
            catch (...) { failure = std::current_exception(); }
            lock.lock();
            busy = false;
            if (failure) { error = failure; queue.clear(); cv.notify_all(); return; }
            if (queue.empty()) cv.notify_all();
        }
    }
    void submit(Move m) {
        { std::lock_guard lock(mutex); if (error) std::rethrow_exception(error); queue.push_back(m); }
        cv.notify_all();
    }
    void wait() {
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] { return queue.empty() && !busy; });
        if (error) std::rethrow_exception(error);
    }
};

// The inputs remain valid through finish_peer(); all runtime API calls for this flight stay on the helper thread.
struct ExpertTier::PeerCompute {
    core::PeerExperts& peer;
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<bool> done{true};
    bool quit=false, pending=false, ok=true, mixed=false;
    int layer=0, tokens=0, k=0;
    const float* x=nullptr;
    kernels::NativeExpertLayout layout{}, formats[kMaxK]{};
    const kernels::NativeExpertLayout* layouts[kMaxK]{};
    int32_t ids[kMaxK]{},kind[kMaxK]{};
    std::string error;
    std::thread thread;

    explicit PeerCompute(core::PeerExperts& p):peer(p),thread([this]{run();}) {}
    ~PeerCompute() {
        { std::lock_guard lock(mutex); quit=true; }
        cv.notify_one(); thread.join();
    }
    void run() {
        cudaSetDevice(peer.device());
        for (;;) {
            { std::unique_lock lock(mutex); cv.wait(lock,[&]{return quit||pending;});
              if (!pending && quit) return; pending=false; }
            try {
                error.clear();
                ok=peer.launch_window(layer,layout,x,ids,tokens,k,kind,error,mixed?layouts:nullptr);
                if (ok) ok=peer.finish(nullptr,error);
            } catch (const std::exception& e) { ok=false; error=e.what(); }
            done.store(true,std::memory_order_release);
        }
    }
    bool finish(std::string& err) {
        core::SpinWait spin;
        while (!done.load(std::memory_order_acquire)) spin();
        if (!ok) err=error;
        return ok;
    }
    bool launch(int il, const kernels::NativeExpertLayout& f, const float* input, const int32_t* experts,
                int nt, int topk, const int32_t* mask, std::string& err,
                const kernels::NativeExpertLayout* const* replacements) {
        if (!finish(err)) return false;
        std::lock_guard lock(mutex);
        layer=il; layout=f; x=input; tokens=nt; k=topk; mixed=replacements!=nullptr;
        for (int i=0;i<nt*topk;++i) {
            ids[i]=experts[i]; kind[i]=mask[i]; layouts[i]=nullptr;
            if (replacements && replacements[i]) { formats[i]=*replacements[i]; layouts[i]=&formats[i]; }
        }
        done.store(false,std::memory_order_release); pending=true;
        cv.notify_one();
        return true;
    }
};

ExpertTier::ExpertTier(const DsModel& model) : m_(model) {}

bool ExpertTier::launch_peer(int layer, const kernels::NativeExpertLayout& layout, const float* x, const int32_t* ids,
                             int tokens, int k, const int32_t* kind, std::string& err,
                             const kernels::NativeExpertLayout* const* layouts) {
    return peer_compute_ ? peer_compute_->launch(layer,layout,x,ids,tokens,k,kind,err,layouts)
                         : peer_->launch_window(layer,layout,x,ids,tokens,k,kind,err,layouts);
}

bool ExpertTier::finish_peer(std::string& err) {
    return peer_compute_ ? peer_compute_->finish(err) : peer_->finish(nullptr,err);
}

ExpertTier::~ExpertTier() {
    peer_compute_.reset();
    try { if (copy_stream_) settle(); }
    catch (const std::exception& e) { basalt::log::warn("basalt deepseek41","cache cleanup: %s",e.what()); }
    preread_quit_ = true;
    if (preread_.joinable()) preread_.join();
    if (worker_.joinable()) {
        {
            std::lock_guard<std::mutex> g(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        worker_.join();
    }
    peer_submitter_.reset();   // drain outstanding submissions before freeing their RAM destinations
    report("final");
    if (copy_stream_) issue(-1, kAllPieces);
    if (peer_ && !peer_moves_.empty()) {
        std::string e;
        peer_->read_wait(e);
    }
    if (copy_stream_) cudaStreamSynchronize(copy_stream_);
    if (return_stream_) cudaStreamSynchronize(return_stream_);
    if (fetch_stream_) cudaStreamSynchronize(fetch_stream_);
    for (cudaEvent_t e : events_) cudaEventDestroy(e);
    for (cudaEvent_t e : return_events_) cudaEventDestroy(e);
    if (fetch_ev_) cudaEventDestroy(fetch_ev_);
    for (auto e:fetch_scratch_done_) if(e) cudaEventDestroy(e);
    if (fetch_scratch_) cudaFree(fetch_scratch_);
    if (gap_ev_) cudaEventDestroy(gap_ev_);
    if (copy_stream_) cudaStreamDestroy(copy_stream_);
    if (fetch_stream_) cudaStreamDestroy(fetch_stream_);
    if (return_stream_) cudaStreamDestroy(return_stream_);
    for (size_t l = 0; l < ram_.size(); ++l) {
        if (ram_[l] == nullptr) continue;
        if (l < ram_pins_.size()) core::unpin_ram(ram_[l], ram_pins_[l]);
        if (!host_vmm_bytes_.empty()) core::vmm::free_host(ram_[l], host_vmm_bytes_[l]);
        else platform::free_aligned(ram_[l]);
    }
    platform::free_aligned(act_);

}

bool ExpertTier::open(const Reserve& reserve, float* h_rows, std::string& err) {
    core::MemoryAuditScope memory_audit("expert-tiers");
    const Geometry& g = m_.geo();
    if (m_.tiers().vision && m_.vision_device() != m_.device()) {
        const core::OnDevice on(m_.vision_device());
        if (!m_.vision_blas().open(err) || !m_.vision_blas().close(err)) return false;
    }
    n_layers_ = g.n_layers;
    n_expert_ = g.n_expert;
    uint64_t one_each_ram = 0;
    for (int l = 0; l < n_layers_; ++l) {
        const auto& e = m_.pack().experts[l];
        one_each_ram += e.blob_bytes;
        if (!m_.has_overlay()) {
            fetch_stride_ = std::max<uint64_t>(fetch_stride_, e.blob_bytes);
            continue;
        }
        std::vector<uint64_t> sizes;
        for (int expert = 0; expert < n_expert_; ++expert) sizes.push_back(std::max<uint64_t>(e.blob_bytes, expert_bytes(l,expert)));
        std::sort(sizes.begin(), sizes.end());
        sorted_blob_bytes_.push_back(std::move(sizes));
    }
    if (!m_.tiers().diag_tier_frozen) {
        for (const auto& sizes : sorted_blob_bytes_) if (!sizes.empty()) fetch_stride_=std::max(fetch_stride_,sizes.back());
        fetch_stride_=(fetch_stride_+255)/256*256;
        fetch_slots_=std::max(1,std::min(kMaxK,(int)g.n_used*std::clamp(m_.tiers().spec,1,8))/2);
        if (cudaMalloc((void**)&fetch_scratch_,fetch_stride_*(uint64_t)fetch_slots_*kFetchRings) != cudaSuccess) {
            err="deepseek41: cannot allocate transient expert fetch slots"; return false;
        }
    }
    uint64_t free_b = 0, total_b = 0;
    std::string why;
    if (!core::device_memory(-1, free_b, total_b, why)) {
        err = "deepseek41: cannot query the free device memory: " + why;
        return false;
    }
    auto slots_in = [&](uint64_t room, uint64_t left_free) {
        return slots_fit(room, left_free, n_expert_);
    };
    if (!load_rank(err)) return false;
    base_count_.assign(n_layers_, n_expert_);
    if (m_.has_overlay()) for (int l = 0; l < n_layers_; ++l) {
        for (int e = 0; e < n_expert_; ++e) if (m_.replacement(l, e)) --base_count_[l];
    }
    const int max_base = *std::max_element(base_count_.begin(), base_count_.end());
    const uint64_t asked_free = (uint64_t) m_.tiers().vram_reserve_mib << 20;
    int copies = slots_in(free_b, reserve.copies + asked_free);
    int main_homes = slots_in(free_b, reserve.homes + asked_free);
    start_floor_ = free_floor_ = asked_free;
    if (m_.tiers().expert_cache > 0) cap_per_layer_ = m_.tiers().expert_cache / n_layers_;
    if (m_.diag_tier_vram() >= 0) cap_per_layer_ = cap_per_layer_ >= 0 ? std::min<int64_t>(cap_per_layer_, m_.diag_tier_vram()) : m_.diag_tier_vram();
    if (cap_per_layer_ >= 0) {
        copies = (int) std::min<int64_t>(copies, cap_per_layer_);
        main_homes = (int) std::min<int64_t>(main_homes, cap_per_layer_);
    }

    int peer_slots = 0;
    uint64_t peer_room = 0;
    if (m_.peer_device() >= 0) {
        peer_ = std::make_unique<core::PeerExperts>();
        std::string pe;
        if (peer_->open_plain(m_.peer_device(), n_layers_, n_expert_, g.n_embd, g.n_ff_exp, h_rows, pe) && open_share(pe)) {
            const uint64_t reserve = (uint64_t) m_.tiers().peer_reserve_mib << 20;
            peer_room = peer_->room();
            peer_slots = slots_fit(peer_room, reserve, n_expert_, 0, false, true);
        } else {
            share_.reset();
            basalt::log::warn("basalt deepseek41", "second card not used: %s", pe.c_str());
            peer_.reset();
        }
    }

    if (m_.tiers().diag_tier_peer >= 0) peer_slots = std::min(peer_slots, m_.tiers().diag_tier_peer);
    main_homes = std::min(main_homes, n_expert_ - peer_slots);
    copies = std::min(copies, n_expert_ - peer_slots);
    const int peer_homes = std::min(peer_slots, n_expert_);

    // The prompt and image buffers first, so the free memory the tier reads already lacks them. The ring holds five layers
    // of the second card's homes, or its cap when experts may stay in the file.
    core::detail::HostMemory hm;
    auto available = [&] {
        if (!core::detail::host_available_memory(hm)) return uint64_t{0};
        // Leave room for replacement weights and on-demand image buffers before placing experts.
        const uint64_t reserved = m_.initial_overlay_ram_budget() + host_.image_budget_bytes();
        return hm.available > reserved ? hm.available-reserved : uint64_t{0};
    };
    const uint64_t ram_reserve = platform::ram_reserve();
    // the image encoder's cuBLAS handle takes host memory when created, so it is taken before the reading
    if (m_.tiers().vision && m_.vision_device() == m_.device() && !m_.vision_blas().open(err))
        return false;
    if (!host_.open(m_, 0, m_.sessions(), m_.tiers().vision, err))
        return false;
    uint64_t avail = available();
    // places per layer above the reserve, less the spare each layer keeps, in `bytes` of RAM
    auto places_in = [&](uint64_t bytes) {
        const uint64_t places = core::ram_tier_room(bytes, ram_reserve, true) / one_each_ram;
        return places > 0 ? places - 1 : 0;
    };
    const bool ram_given = m_.tiers().resident_budget > 0 || m_.diag_tier_ram() >= 0;
    auto ram_in = [&] {
        int slots = (int) std::min<uint64_t>(places_in(avail), (uint64_t) n_expert_);
        if (m_.tiers().resident_budget > 0)
            slots = std::min(slots, (int) std::min<uint64_t>(m_.tiers().resident_budget / one_each_ram, (uint64_t) n_expert_));
        if (m_.diag_tier_ram() >= 0) slots = std::min(m_.diag_tier_ram(), n_expert_);
        return slots;
    };
    int ram_slots = ram_in();
    // what the machine can give the tier whatever else runs: its RAM (or the container's) less the engine's own buffers.
    // The home-versus-copies choice reads this alone, so the cache state at start never moves it.
    const uint64_t total = std::min(platform::total_physical_memory(), hm.cgroup_limit);
    const uint64_t host_budget = host_.bytes() + host_.image_budget_bytes();
    const int ram_cap = ram_given ? -1 : (int) std::min<uint64_t>(places_in(total > host_budget ? total - host_budget : 0), 1u << 20);
    if (ram_slots < (m_.has_overlay() ? max_base : n_expert_ - peer_homes)) {
        if (!host_.set_ring(prefillp::kRingPieces, err)) return false;
        avail = available();
        ram_slots = ram_in();
    }
    // Rank replacements with ordinary experts; their conservative host budget is already reserved above.
    bool keep_main_homes = false;
    auto plan_homes = [&] {
        auto layout = [&](int main) { return policy::Homes{main, peer_slots, n_expert_ - main - peer_slots, 0}; };
        int required = 0;
        for (int l = 0; l < n_layers_; ++l) {
            int count = 0;
            for (int r = 0; r < n_expert_; ++r)
                if (!m_.replacement(l, ranked(l,r)) && policy::place_of_rank(layout(0), copies, r) == policy::Place::Ram) ++count;
            required = std::max(required, count);
        }
        homes_ = layout(required > ram_slots || keep_main_homes ? main_homes : 0);
        layer_homes_.resize(n_layers_);
        int max_ram = 0, max_file = 0;
        for (int l = 0; l < n_layers_; ++l) {
            int left = 0;
            for (int r = 0; r < n_expert_; ++r)
                if (!m_.replacement(l, ranked(l,r)) && policy::place_of_rank(homes_, copies, r) == policy::Place::Ram) ++left;
            auto& h = layer_homes_[l];
            h = homes_;
            h.ram = std::min(left, ram_slots);
            h.file = left - h.ram;
            max_ram = std::max(max_ram, h.ram);
            max_file = std::max(max_file, h.file);
        }
        homes_.ram = max_ram;
        homes_.file = max_file;
        if (!m_.has_overlay()) {
            homes_ = policy::plan_homes(n_expert_, main_homes, peer_slots, ram_slots, ram_cap);
            std::fill(layer_homes_.begin(), layer_homes_.end(), homes_);
        }
    };
    plan_homes();
    keep_main_homes = homes_.main > 0;
    if (peer_ && m_.has_overlay()) {
        const uint64_t reserve = (uint64_t)m_.tiers().peer_reserve_mib << 20;
        // The conservative estimate opens the host buffers safely. Once the home/copy
        // arrangement is known, use its actual ranked formats to fill the helper's budget.
        for (int pass = 0; pass < 4; ++pass) {
            const int first = homes_.main > 0 ? main_homes : copies;
            int fit = slots_fit(peer_room,reserve,n_expert_-first,first);
            if (m_.tiers().diag_tier_peer >= 0) fit = std::min(fit,m_.tiers().diag_tier_peer);
            if (fit == peer_slots) break;
            peer_slots = fit;
            if (!host_.set_ring(prefillp::kRingPieces,err)) return false;
            avail = available(); ram_slots = ram_in();
            plan_homes();
        }
        // A changed RAM arrangement can change the helper's start rank. Refuse any
        // overshoot after iteration; reducing its slots always leaves the RAM plan valid.
        for (;;) {
            uint64_t bytes = 0;
            for (auto b : slot_sizes(peer_slots,homes_.main > 0 ? homes_.main : copies)) bytes += ((uint64_t)b+255)/256*256;
            if (bytes + reserve <= peer_room || peer_slots == 0) break;
            --peer_slots; plan_homes();
        }
    }
    if (homes_.peer == 0) { share_.reset(); peer_.reset(); }

    // the base arena (homes, or copies a long prompt borrows) leaves its reserve free; grow() fills it beside a short prompt
    reserve_ = reserve.beside_homes;
    modules_ = reserve.modules;
    per_layer_ = homes_.main > 0 ? homes_.main : copies;
    if (homes_.main == 0 && per_layer_ < g.n_used && !m_.has_overlay()) {
        err = "deepseek41: not enough device memory for the expert tier";
        return false;
    }
    const auto slot_bytes = cache_sizes((int)per_layer_,false);
    cache_.set_segment_bytes(policy::kCacheSegment);
    cache_.set_malloc_slabs(core::vmm::tcc_mode());
    if (!slot_bytes.empty() && !cache_.open_sized(slot_bytes, n_layers_, n_expert_, err)) return false;
    if (malloc_slab_backed())
        basalt::log::info("basalt deepseek41", "TCC: using reclaimable cudaMalloc slabs");
    res_.assign((size_t) n_layers_ * n_expert_, -1);
    stamp_.assign((size_t) n_layers_ * n_expert_, 0);
    owner_.assign(slot_bytes.size(), -1);
    busy_.assign(slot_bytes.size(), 0);
    int lo_prio = 0, hi_prio = 0;
    cudaDeviceGetStreamPriorityRange(&lo_prio, &hi_prio);
    if (cudaEventCreateWithFlags(&fetch_ev_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&gap_ev_, cudaEventDisableTiming) != cudaSuccess ||
        cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&return_stream_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithPriority(&fetch_stream_, cudaStreamNonBlocking, hi_prio) != cudaSuccess) {
        err = "deepseek41: cannot create the promotion stream";
        return false;
    }
    for (auto& e:fetch_scratch_done_) {
        if (cudaEventCreateWithFlags(&e,cudaEventDisableTiming) != cudaSuccess) { err="cannot create a fetch reuse event"; return false; }
    }
    for (size_t i = 0; i < kEvents; ++i) {
        cudaEvent_t e;
        if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
            err = "deepseek41: cannot create a promotion event";
            return false;
        }
        events_.push_back(e);
        cudaEvent_t back;
        if (cudaEventCreateWithFlags(&back,cudaEventDisableTiming) != cudaSuccess) { err="cannot create an expert return event"; return false; }
        return_events_.push_back(back);
        free_events_.push_back((int) i);
    }

    // The CPU-side formats and worker pool.
    for (int l = 0; l < n_layers_; ++l) {
        const ExpertLayer& el = m_.pack().experts[(size_t) l];
        kernels::cpu::NativeFmt f;
        if (!kernels::cpu::native_fmt(el.gu_type, el.d_type, g.n_embd, g.n_ff_exp, f, err, g.clamp_exp[(size_t) l]))
            return false;
        fmt_.push_back(f);
    }
    act_ = (uint8_t*) platform::alloc_aligned(kernels::cpu::kNativeActBytes * kMaxWindow * kMaxK, 64);
    const int workers = m_.tiers().pool_workers > 0 ? m_.tiers().pool_workers : (int) kernels::cpu::physical_cores(true).size();
    pool_ = std::make_unique<kernels::cpu::ExpertPool>(workers, m_.tiers().pool_affinity_given ? m_.tiers().pool_affinity : kernels::cpu::PoolAffinity::All);

    if (!load_ram(err)) return false;
    if constexpr (platform::kDedicatedPeerCompute) {
        if (peer_) peer_compute_=std::make_unique<PeerCompute>(*peer_);
    }
    if constexpr (platform::kDirectHostVmmExchange) {
        if (peer_ && !host_vmm_bytes_.empty() && !profile_peer_exchange_) {
            peer_submitter_ = std::make_unique<PeerSubmitter>(*peer_);
            basalt::log::info("basalt deepseek41", "helper expert exchange: dedicated submission thread");
        }
    }
    promoted_ = 0;
    exclusive_cache_ = true;
    {
        // a promotion's victim needs a RAM place at once; a burst of them takes these instead of pinning on the token's path
        uint64_t blob = m_.overlay_max_bytes();
        for (int l = 0; l < n_layers_; ++l) blob = std::max<uint64_t>(blob, m_.pack().experts[(size_t) l].blob_bytes);
        m_.keep_expert_ram(kSpareBlocks, blob, m_.device());
    }
    m_.tier_profile().enabled=m_.tiers().diag_tier_timing;
    basalt::log::info("basalt deepseek41","resident expert RAM %.2f GiB; loaded directly into final tier placement",
                      resident_ram_bytes()/1073741824.);
    // what the start picked by itself, on the server's auto line; the RAM floor the tier left free
    std::printf("auto flags: resident %.1f GB, pool-workers %d\n",
                (double) resident_ram_bytes() / 1e9, pool_->workers());   // the server keeps the words after the name
    std::fflush(stdout);
    const uint64_t ram_floor = ram_reserve + ram_file_keep_ + core::pin_tracking_bytes(resident_ram_bytes());
    basalt::log::info("basalt generate", "RAM floor %.1f GiB of %.1f GiB free", (double) ram_floor / 1073741824.0,
                      (double) platform::available_memory() / 1073741824.0);
    if (peer_) basalt::log::info("basalt deepseek41", "peer GPU %d: %lld experts", peer_->device(), (long long) n_layers_ * homes_.peer);
    basalt::log::info("basalt deepseek41", "expert homes of %d per layer: main card %d, second card %d, RAM %d, file %d; main card copies %lld",
                 n_expert_, homes_.main, homes_.peer, homes_.ram, homes_.file, (long long)((cache_.slots()+ext_.slots())/n_layers_-homes_.main));
    program::warn_spill(spill(), peer_ ? "" : ", or --gpu 0,1");
    basalt::log::info("basalt deepseek41", "VRAM expert tier %.2f GiB, %lld of %d experts per layer (%.0f%%); CPU pool %d workers",
                 cache_.gib(), (long long) per_layer_, n_expert_, 100.0 * (double) per_layer_ / n_expert_, pool_->workers());
    // the RAM the tier took emptied the file cache; with no expert read from the file, it takes the token embedding back
    if (homes_.file == 0) m_.warm_token_table();
    return true;
}

// The second card's prompt share takes its buffers before the card's slots are sized, and each layer's prices.
bool ExpertTier::open_share(std::string& err) {
    const Geometry& g = m_.geo();
    prefill::PeerShare::Dims d;
    d.n_embd = g.n_embd;
    d.n_ff = g.n_ff_exp;
    size_t stride = 0;
    for (const ExpertLayer& el : m_.pack().experts) stride = std::max<size_t>(stride, staged_stride(el));
    d.group_bytes = stride * (size_t) d.launch_experts;
    d.act_bytes = prefill::mmq::q8_bytes(prefillp::kMaxChunk, g.n_embd);
    d.max_tokens = prefillp::kMaxChunk;
    d.max_rows = (int64_t) prefillp::kMaxChunk * g.n_used;
    d.n_expert = n_expert_;
    int main_dev = 0;
    cudaGetDevice(&main_dev);
    share_ = std::make_unique<prefill::PeerShare>();
    if (!share_->open(main_dev, peer_->device(), d, err)) return false;
    // a row of a layer's products is three matrices of n_embd x n_ff multiply-adds
    const core::DeviceRates r0 = core::device_rates(main_dev), r1 = core::device_rates(peer_->device());
    const double macs = 3.0 * (double) g.n_embd * (double) g.n_ff_exp;
    rates_.assign((size_t) n_layers_, {});
    for (prefill::ShareRates& r : rates_) {
        r.link0 = r0.link_bytes;
        r.link1 = r1.link_bytes;
        r.rows0 = r0.sm_hz * prefill::kMmqMacsPerSmCycle / macs;
        r.rows1 = r1.sm_hz * prefill::kMmqMacsPerSmCycle / macs;
        r.sm_hz0 = r0.sm_hz;
    }
    return true;
}

program::ExpertSpill ExpertTier::spill() const {
    int64_t count = 0;
    uint64_t bytes = 0;
    for (int l = 0; l < (int) layer_homes_.size(); ++l) {
        count += layer_homes_[l].file;
        bytes += (uint64_t) layer_homes_[l].file * m_.pack().experts[l].blob_bytes;
    }
    return {count, (int64_t) n_layers_ * n_expert_, bytes};
}

namespace {

// The second card's real path over the homes of the layer it holds most of, one token at a time: each share is armed
// as a layer arms it with its head, the card waits at its word, and the round trip runs from the plan to the rows.
class PeerPath final : public program::ProbePeerPath {
public:
    PeerPath(core::PeerExperts& peer, int top_k, const DsModel& model) : peer_(peer), k_(top_k), model_(model) {}
    bool setup(const program::ProbeEnv& env, const kernels::cpu::NativeFmt& f, std::string& why) override {
        L_ = kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
        for (int64_t e = 0; e < env.n_expert && (int) ids_.size() < 8; ++e)
            if (peer_.slot_ptr(0, e) != nullptr && !model_.replacement(0,(int)e)) ids_.push_back((int32_t) e);
        if (ids_.empty()) { why = "no_experts_on_peer"; return false; }
        if (!x_.alloc((size_t) f.n_embd * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped)) { why = "alloc_failed"; return false; }
        std::mt19937 rng(5);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (int64_t i = 0; i < f.n_embd; ++i) ((float*) x_.p)[i] = nd(rng);
        return true;
    }
    int max_entries() const override { return (int) ids_.size(); }
    int rings() const override { return 24; }
    int blocks() const override { return 0; }
    void fill(int n) override { n_ = n; }
    bool window(const kernels::cpu::NativeFmt&, const std::function<void(int)>& before, std::vector<double>& rt,
                std::string& err) override {
        int32_t ids[ExpertTier::kMaxK] = {}, kind[ExpertTier::kMaxK] = {};
        const int k = std::max(k_, n_);
        for (int i = 0; i < n_; ++i) { ids[i] = ids_[(size_t) i]; kind[i] = 2; }
        rt.clear();
        bool ok = true;
        for (int r = 0; r < rings() && ok; ++r) {
            before(r);
            ok = peer_.arm(L_, (const float*) x_.p, 1, k, policy::peer_groups(1), err);
            const auto ta = program::ProbeClock::now();
            while (ok && program::us_since(ta) < kHeadUs) {
            }
            const auto t0 = program::ProbeClock::now();
            ok = ok && peer_.raise(0, ids, kind, err) && peer_.finish(nullptr, err);
            rt.push_back(program::us_since(t0));
        }
        return ok;
    }

private:
    static constexpr double kHeadUs = 100;   // about a head's time: the arm's launches are on the card by then
    core::PeerExperts& peer_;
    program::HostBuf x_;
    int k_, n_ = 1;
    const DsModel& model_;
    kernels::NativeExpertLayout L_;
    std::vector<int32_t> ids_;
};

}  // namespace

void ExpertTier::probe_env(program::ProbeEnv& env) {
    env.pool = pool_.get();
    env.ram_blob = [this](int64_t l, int64_t e) -> const uint8_t* {
        return m_.replacement((int) l, (int) e) ? nullptr : locate((int) l, (int) e).pinned;
    };
    env.fmt = fmt_;
    env.n_layers = n_layers_;
    env.n_expert = n_expert_;
    env.top_k = m_.geo().n_used;
    if (peer_) {
        env.peer_device = peer_->device();
        env.peer_path = std::make_shared<PeerPath>(*peer_, m_.geo().n_used, m_);
    }
}

bool ExpertTier::load_ram(std::string& err) {
    core::MemoryAuditScope memory_audit("expert-ram-map");
    uint64_t max_blob = 0;
    for (const ExpertLayer& e : m_.pack().experts) {   // one expert in every layer
        max_blob = std::max<uint64_t>(max_blob, e.blob_bytes);
    }
    cover_ = homes_.ram;
    ram_.assign((size_t) n_layers_, nullptr);
    ram_idx_.assign((size_t) n_layers_ * n_expert_, -1);
    spare_.assign((size_t) n_layers_, -1);
    extra_ram_.resize(n_layers_);
    free_places_.resize(n_layers_);
    swapping_.assign((size_t) n_layers_, 0);
    spare_since_.assign((size_t) n_layers_, 0);
    misses_.assign((size_t) n_layers_ * n_expert_, {});
    preread_experts_.resize(n_layers_);
    const int64_t copies = homes_.main > 0 ? 0 : per_layer_;
    std::vector<int32_t> device_home;   // [layer * n_expert + e] the slot of a home on either card, or -1
    device_home.assign((size_t) n_layers_ * n_expert_, -1);
    if (peer_) {
        peer_owner_.assign((size_t) n_layers_ * homes_.peer, -1);
        const auto sizes = slot_sizes(homes_.peer, homes_.main > 0 ? homes_.main : (int)copies);
        if (!peer_->open_slots(sizes, err)) return false;
        // scratch slots for the misses the card fetches over its own link, in what its homes leave above its reserve
        const uint64_t keep = (uint64_t) m_.tiers().peer_reserve_mib << 20, room = peer_->room();
        const uint64_t slot = (max_blob + 255) / 256 * 256;
        const int n = room > keep ? (int) std::min<uint64_t>(policy::kPeerFetchSlots, (room - keep) / slot) : 0;
        std::string se;
        if (!m_.has_overlay() && n > 0 && !peer_->open_scratch(n, max_blob, se))
            basalt::log::warn("basalt deepseek41", "the second card fetches no misses over its link: %s", se.c_str());
    }
    for (int l = 0; l < n_layers_; ++l) {
        std::vector<int> held;
        int n_main = 0, n_peer = 0;
        auto empty_slot = [&](bool helper, uint64_t bytes) {
            const int count = helper ? homes_.peer : (int)per_layer_;
            const auto& owners = helper ? peer_owner_ : owner_;
            const auto& cache = helper ? peer_->cache() : cache_;
            int found = -1; uint64_t capacity = ~0ull;
            for (int s = l * count; s < (l+1) * count; ++s) {
                const uint64_t cap = cache.bytes_of(s+1)-cache.bytes_of(s);
                if (owners[s] < 0 && cap >= bytes && cap < capacity) { found=s; capacity=cap; }
            }
            return found;
        };
        const policy::Homes placement = m_.has_overlay() ?
            policy::Homes{homes_.main, homes_.peer, n_expert_ - homes_.main - homes_.peer, 0} : homes_;
        for (int r = 0; r < n_expert_; ++r) {
            const int e = ranked(l, r);
            const bool replacement = m_.replacement(l,e) != nullptr;
            const size_t idx = (size_t) l * n_expert_ + e;
            switch (policy::place_of_rank(placement, (int) copies, r)) {
                case policy::Place::Main: {
                    const int32_t slot = m_.has_overlay() ? empty_slot(false, expert_bytes(l,e)) : l*(int)per_layer_+n_main++;
                    if (slot < 0) { err="deepseek41: primary slot layout cannot hold its ranked experts"; return false; }
                    res_[idx] = slot;
                    owner_[(size_t) slot] = e;
                    if (!replacement) device_home[idx] = slot;
                    break;
                }
                case policy::Place::Peer: {
                    const int32_t slot = m_.has_overlay() ? empty_slot(true, expert_bytes(l,e)) : l*homes_.peer+n_peer++;
                    if (slot < 0) { err="deepseek41: helper slot layout cannot hold its ranked experts"; return false; }
                    peer_->admit(l, e, slot);
                    peer_owner_[(size_t) slot] = e;
                    if (!replacement) device_home[idx] = slot;
                    break;
                }
                case policy::Place::Ram:
                    if (!replacement && (int)held.size() < layer_homes_[l].ram) held.push_back(e);
                    else if (!replacement) preread_experts_[l].push_back(e);
                    break;
                case policy::Place::File:
                    if (!replacement) preread_experts_[l].push_back(e);
                    break;
            }
        }
        std::sort(held.begin(), held.end());
        for (size_t i = 0; i < held.size(); ++i) ram_idx_[(size_t) l * n_expert_ + held[i]] = (int32_t) i;
    }
    initial_seeds_.resize(n_layers_);
    for (int l = 0; l < n_layers_; ++l) {
        auto& seed = initial_seeds_[l];
        for (int r = 0; r < n_expert_; ++r) {
            const int e = ranked(l,r);
            const size_t idx = (size_t)l*n_expert_+e;
            if (res_[idx] < 0 && !(peer_ && peer_->slot_ptr(l,e)) &&
                (m_.replacement(l,e) || ram_idx_[idx] >= 0)) seed.push_back(e);
        }
        // A reclaimable base arena owns its initial experts just like permanent GPU homes.
        if (homes_.main == 0) {
            const int take = std::min<int>((int)per_layer_,(int)seed.size());
            for (int r = 0; r < take; ++r) {
                const int e = seed[r];
                int found = -1; uint64_t best = ~0ull;
                for (int r = 0; r < per_layer_; ++r) {
                    const int s = main_slot(l,r);
                    const auto capacity = slot_capacity(s);
                    if (owner_[s] < 0 && capacity >= expert_bytes(l,e) && capacity < best) { found=s; best=capacity; }
                }
                if (found < 0) { err="deepseek41: initial base cache cannot hold its ranked expert"; return false; }
                owner_[found] = e;
                res_[(size_t)l*n_expert_+e] = found;
                stamp_[(size_t)l*n_expert_+e] = ++tick_;
                if (!m_.replacement(l,e)) device_home[(size_t)l*n_expert_+e] = found;
            }
            seed.erase(seed.begin(),seed.begin()+take);
        }
    }
    uint64_t free_b = 0, total_b = 0;
    std::string memory_error;
    if (!core::device_memory(-1,free_b,total_b,memory_error)) {
        err="deepseek41: querying initial cache room failed: " + memory_error;
        return false;
    }
    const int limit = (int)(cap_per_layer_ >= 0 ? std::min<int64_t>(n_expert_,cap_per_layer_) : n_expert_);
    const int n = slots_fit(free_b,reserve_+modules_+free_floor_,std::max(0,limit-(int)per_layer_),0,true);
    // Mixed-size permanent homes can have a loanable cache beside them.
    const bool slab_extra = malloc_slab_backed() && m_.has_overlay();
    const bool elastic_extra = !malloc_slab_backed() || slab_extra;
    const int capacity = elastic_extra ? std::max(0,limit-(int)per_layer_) : n;
    if (capacity > 0) {
        auto sizes = cache_sizes(capacity,true);
        const auto first = slot_sizes(n,0,true);
        uint64_t initial_bytes = 0;
        for (int r = 0; r < n; ++r) for (int l = 0; l < n_layers_; ++l) {
            sizes[(size_t)r*n_layers_+l] = first[(size_t)l*n+r];
            initial_bytes += ((uint64_t)first[(size_t)l*n+r]+255)/256*256;
        }
        if (elastic_extra) ext_.set_segment_bytes(policy::kCacheSegment);
        if (slab_extra) ext_.set_malloc_slabs(true);
        ext_.set_initial_bytes((int64_t)initial_bytes);
        if (!ext_.open_sized(sizes,n_layers_,n_expert_,err)) return false;
        ext_per_layer_ = capacity;
        const int64_t nb = (int64_t)n_layers_*per_layer_;
        owner_.resize((size_t)slot_total(),-1);
        busy_.resize((size_t)slot_total(),0);
        for (int l = 0; l < n_layers_; ++l) {
            const auto& seed = initial_seeds_[l];
            for (int r = 0; r < std::min<int>(n,(int)seed.size()); ++r) {
                const int e = seed[r];
                int found = -1; uint64_t best = ~0ull;
                for (int rank = 0; rank < capacity; ++rank) {
                    const int at = rank*n_layers_+l;
                    if (at >= ext_.slots()) break;
                    const auto bytes = ext_.bytes_of(at+1)-ext_.bytes_of(at);
                    if (owner_[(size_t)nb+at] < 0 && bytes >= expert_bytes(l,e) && bytes < best) { found=at; best=bytes; }
                }
                if (found < 0) { err="deepseek41: initial extra cache cannot hold its ranked expert"; return false; }
                const int slot = (int)nb+found;
                owner_[slot] = e;
                res_[(size_t)l*n_expert_+e] = slot;
                stamp_[(size_t)l*n_expert_+e] = ++tick_;
                if (!m_.replacement(l,e)) device_home[(size_t)l*n_expert_+e] = slot;
            }
        }
    }
    initial_seeds_.clear();
    if (cudaDeviceSynchronize() != cudaSuccess) { err="deepseek41: initializing expert arenas failed"; return false; }
    // Rebudget after final GPU placement, before reading or allocating any expert RAM.
    // Only CPU-owned replacements are mandatory; GPU homes never receive host copies.
    const uint64_t grain = platform::kDirectHostVmmExchange ? core::vmm::host_granularity() : 0;
    uint64_t replacements = 0;
    std::vector<std::vector<int>> remaining(n_layers_);
    std::vector<policy::RamLayer> needs;
    for (int l = 0; l < n_layers_; ++l) {
        for (int r = 0; r < n_expert_; ++r) {
            const int e = ranked(l,r);
            if (res_[(size_t)l*n_expert_+e] >= 0 || (peer_ && peer_->slot_ptr(l,e))) continue;
            if (const auto* v = m_.replacement(l,e))
                replacements += policy::ram_aligned(v->layout.bytes,grain);
            else remaining[l].push_back(e);
        }
        int count = (int)remaining[l].size();
        if (m_.diag_tier_ram() >= 0) count = std::min(count,m_.diag_tier_ram());
        needs.push_back({count,m_.pack().experts[l].blob_bytes});
    }
    core::detail::HostMemory hm;
    if (!core::detail::host_available_memory(hm)) { err="deepseek41: cannot query available RAM for final placement"; return false; }
    ram_file_keep_ = 0;
    if (m_.tiers().resident_budget == 0 && m_.diag_tier_ram() < 0) {
        core::FileCache cache;
        for (int l=0;l<n_layers_;++l) {
            uint64_t layer_bytes=0;
            for (int e=0;e<n_expert_;++e) layer_bytes+=expert_bytes(l,e);
            cache.layer_bytes=std::max(cache.layer_bytes,layer_bytes);
        }
        const auto full=policy::plan_ram(needs,~uint64_t{0},replacements,grain);
        ram_file_keep_=core::file_cache_keep(hm.available,platform::ram_reserve(),true,full.bytes,cache);
    }
    uint64_t budget = core::ram_tier_room(hm.available,platform::ram_reserve()+ram_file_keep_,true);
    if (m_.tiers().resident_budget > 0) budget = std::min(budget,m_.tiers().resident_budget);
    const auto plan = policy::plan_ram(needs,budget,replacements,grain);
    if (!plan.fits) {
        err="deepseek41: CPU replacement experts exceed the final RAM budget above the safety floor";
        return false;
    }
    std::fill(ram_idx_.begin(),ram_idx_.end(),-1);
    homes_.ram = cover_ = homes_.file = 0;
    for (int l = 0; l < n_layers_; ++l) {
        auto held = remaining[l];
        held.resize(plan.counts[l]);
        std::sort(held.begin(),held.end());
        for (size_t i = 0; i < held.size(); ++i) ram_idx_[(size_t)l*n_expert_+held[i]]=(int32_t)i;
        preread_experts_[l].assign(remaining[l].begin()+plan.counts[l],remaining[l].end());
        layer_homes_[l].ram = plan.counts[l];
        layer_homes_[l].file = (int)preread_experts_[l].size();
        homes_.ram = cover_ = std::max(cover_,plan.counts[l]);
        homes_.file = std::max(homes_.file,layer_homes_[l].file);
    }
    if (homes_.file > 0) {
        auto mapped = platform::MappedFile::open_ro(m_.path(),&err);
        if (!mapped) return false;
        file_ = std::move(*mapped);
    }
    basalt::log::info("basalt deepseek41","final RAM plan: %.2f GiB experts including spares, %.2f GiB CPU replacements; %lld file-backed experts",
                      plan.bytes/1073741824.,replacements/1073741824.,(long long)spill().experts);
    const bool swaps = homes_.file > 0 && cover_ > 0;   // the file's experts swap into RAM from a thread
    for (int l = 0; l < n_layers_; ++l)
        if (layer_homes_[l].ram > 0) spare_[(size_t) l] = layer_homes_[l].ram;
    const auto t0 = std::chrono::steady_clock::now();
    // what the start places from here on: the file's share never loads, a second card and the RAM tier have their bars
    uint64_t ram_bytes = 0, replaced_bytes = 0;
    for (int l = 0; l < n_layers_; ++l) {
        ram_bytes += (uint64_t) layer_homes_[l].ram * m_.pack().experts[l].blob_bytes;
        replaced_bytes += (uint64_t) (n_expert_ - base_count_[l]) * m_.pack().experts[l].blob_bytes;
        for (int e = 0; e < n_expert_; ++e)
            if (const auto* v = m_.replacement(l,e); v && res_[(size_t)l*n_expert_+e] < 0 && !(peer_ && peer_->slot_ptr(l,e)))
                ram_bytes += v->layout.bytes;
    }
    uint64_t main_bytes = 0, peer_bytes = 0;
    for (int l = 0; l < n_layers_; ++l) for (int e = 0; e < n_expert_; ++e) {
        if (res_[(size_t)l*n_expert_+e] >= 0) main_bytes += expert_bytes(l,e);
        else if (peer_ && peer_->slot_ptr(l,e)) peer_bytes += expert_bytes(l,e);
    }
    core::LoadProgress::skipped(spill().bytes + replaced_bytes);
    if (ram_bytes > 0) core::LoadProgress::plan("ram", ram_bytes);
    std::optional<core::LoadProgress> ram_load, main_load, peer_load;
    if (ram_bytes > 0) ram_load.emplace("ram", ram_bytes);
    if (main_bytes > 0) main_load.emplace(core::LoadProgress::cuda(0), main_bytes);
    if (peer_bytes > 0) {
        core::LoadProgress::plan(core::LoadProgress::cuda(peer_->device()), peer_bytes);
        peer_load.emplace(core::LoadProgress::cuda(peer_->device()), peer_bytes);
    }
    // the layout: one span per layer, each with its swap spare
    ram_bytes_.assign((size_t) n_layers_, 0);
    auto& span_bytes = ram_bytes_;
    core::RamGrid grid;
    grid.n_layers = n_layers_;
    grid.n_expert = n_expert_;
    grid.file_base = m_.pack().experts_range.offset;
    for (int l = 0; l < n_layers_; ++l) {
        const ExpertLayer& el = m_.pack().experts[(size_t) l];
        grid.layer_offset.push_back(el.offset);
        grid.blob_bytes.push_back(el.blob_bytes);
        if (layer_homes_[l].ram <= 0) continue;
        span_bytes[(size_t) l] = el.blob_bytes * (uint64_t) (layer_homes_[l].ram + 1);
    }
    core::RamLoad load;
    load.fd = m_.experts_fd();
    load.stage_pinned = true;
    load.threads = 8;
    load.progress = ram_load ? &*ram_load : nullptr;
    load.sink = [&](int64_t l, const core::RamRun& r, const uint8_t* bytes, std::string& e) {
        const size_t idx = (size_t) l * n_expert_ + (size_t) r.first;
        const int32_t home = device_home[idx];
        const bool ok = res_[idx] >= 0 ? cudaMemcpy(const_cast<uint8_t*>(slot_ptr(home)), bytes, r.bytes, cudaMemcpyHostToDevice) == cudaSuccess
                                       : peer_->fill_slot(home, bytes, r.bytes);
        if (!ok) {
            e = "cannot place an expert on its card";
            return false;
        }
        (res_[idx] >= 0 ? main_load : peer_load)->add(r.bytes);
        return true;
    };
    core::RamLoadStats read;
    bool loaded = false;
    if constexpr (platform::kDirectHostVmmExchange) {
        if (cover_ > 0) {
            // Map each layer on its reader thread so earlier layers can be filled while
            // Windows backs the rest. No second arena or extra file staging.
            host_vmm_bytes_.resize(n_layers_);
            std::atomic<bool> map_failed{false};
            load.groups.resize(n_layers_);
            load.chunk = std::max(load.chunk, max_blob);
            load.prepare_group = [&](int64_t l, std::vector<core::RamRun>& runs, std::string& e) {
                if (cudaSetDevice(m_.device()) != cudaSuccess ||
                    (span_bytes[l] && !(ram_[l] = core::vmm::alloc_host(span_bytes[l], host_vmm_bytes_[l])))) {
                    map_failed.store(true);
                    e = "could not map a host expert layer";
                    return false;
                }
                runs = core::ram_layer_group(grid, l, ram_[l], ram_idx_, device_home);
                return true;
            };
            loaded = core::read_ram_runs(load, read, err);
            load.prepare_group = {};
            if (!loaded) {
                if (!map_failed.load()) return false;
                // All readers have joined. Drop partial mappings before the ordinary
                // allocation; GPU homes are refilled with the same immutable weights.
                for (int l = 0; l < n_layers_; ++l) {
                    core::vmm::free_host(ram_[l], host_vmm_bytes_[l]);
                    ram_[l] = nullptr;
                }
                host_vmm_bytes_.clear();
                basalt::log::warn("basalt deepseek41", "host VMM unavailable; using normal RAM registration");
                if (ram_load) ram_load.emplace("ram", ram_bytes);
                if (main_load) main_load.emplace(core::LoadProgress::cuda(0), main_bytes);
                if (peer_load) peer_load.emplace(core::LoadProgress::cuda(peer_->device()), peer_bytes);
                err.clear();
            } else if (peer_) {
                const char* profile = std::getenv("BASALT_DS_PROFILE_PEER_EXCHANGE");
                profile_peer_exchange_ = profile && std::strcmp(profile, "1") == 0;
            }
        }
    }
    if (!loaded) {
        for (int l = 0; cover_ > 0 && l < n_layers_; ++l) {
            if (!span_bytes[l]) continue;
            ram_[l] = core::alloc_ram(span_bytes[l]);
            if (!ram_[l]) {
                err = "deepseek41: cannot allocate the RAM expert tier";
                return false;
            }
        }
        load.groups = core::ram_layer_runs(grid, ram_, ram_idx_, device_home);
        if (!core::read_ram_runs(load, read, err)) {
            err = "deepseek41: loading the expert tiers: " + err;
            return false;
        }
    }
    if (!host_vmm_bytes_.empty()) {
        uint64_t n = 0;
        for (auto b : host_vmm_bytes_) n += b;
        basalt::log::info("basalt deepseek41", "RAM expert tier: %.2f GiB CUDA-pinned (host VMM, primary GPU)", n / 1073741824.0);
    } else if (cover_ > 0) {
        ram_pins_ = core::pin_ram_spans(ram_, span_bytes, /*mapped=*/false);
        int unpinned = 0;
        uint64_t registered = 0, locked = 0;
        for (const core::RamPin& p : ram_pins_) {
            unpinned += p.registered_bytes == 0;
            registered += p.registered_bytes;
            locked += p.locked_bytes;
        }
        basalt::log::info("basalt deepseek41", "RAM expert tier: %.1f GiB registered, %.1f GiB locked",
                          (double) registered / 1073741824.0, (double) locked / 1073741824.0);
        if (unpinned > 0)
            basalt::log::warn("basalt deepseek41", "RAM expert tier: %d of %d layers not pinned, locked: %s", unpinned, n_layers_,
                              ram_pins_[(size_t) n_layers_ - 1].lock_note.c_str());

    }
    const auto devices_ready = [&] {
        std::vector<int> cards{-1};
        if (peer_) cards.push_back(peer_->device());
        std::string why;
        if (core::devices_allocate(cards,why)) return true;
        err="deepseek41: the device cannot allocate after pinning the RAM tier (" + why +
            "); pin less with a smaller --resident-budget-gib";
        return false;
    };
    if (!devices_ready()) return false;
    // Only CPU-owned replacements receive persistent RAM; GPU owners share one bounded upload buffer.
    for (int l = 0; l < n_layers_; ++l) for (int e = 0; e < n_expert_; ++e) {
        const auto* v = m_.replacement(l,e);
        if (!v) continue;
        const int s = res_[(size_t)l*n_expert_+e];
        const bool helper = peer_ && peer_->slot_ptr(l,e);
        uint8_t* src = nullptr;
        if (s >= 0 || helper) {
            src = host_.replacement_stage;
        } else {
            src = m_.prepare_replacement_ram(l,e);
            if (!src) { err="deepseek41: allocating initial CPU replacement failed"; return false; }
        }
        if (!m_.read_initial_replacement(l,e,src,err)) return false;
        if (s >= 0) {
            if (cudaMemcpy(const_cast<uint8_t*>(slot_ptr(s)),src,v->layout.bytes,cudaMemcpyHostToDevice) != cudaSuccess) {
                err = "deepseek41: loading a replacement primary expert"; return false;
            }
            main_load->add(v->layout.bytes);
        } else if (helper) {
            const int ps = peer_->residency()[(size_t)l*n_expert_+e];
            if (!peer_->fill_slot(ps,src,v->layout.bytes)) {
                err = "deepseek41: loading a replacement helper expert"; return false;
            }
            peer_load->add(v->layout.bytes);
        } else {
            m_.publish_replacement_ram(l,e);
            ram_load->add(v->layout.bytes);
        }
    }
    if (m_.has_overlay() && !devices_ready()) return false;
    if (ram_load) ram_load->finish();
    if (main_load) main_load->finish();
    if (peer_load) peer_load->finish();
    if (swaps) worker_ = std::thread([this] { swap_worker(); });
    // what the load left in the page cache goes, and the most routed of the experts left in the file take its place
    platform::advise_dontneed(m_.experts_fd(), m_.pack().experts_range.offset, m_.pack().experts_range.size());
    if (swaps) preread_ = std::thread([this] { preread(); });
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    basalt::log::info("basalt deepseek41", "RAM expert tier %.1f GiB (%d of %d experts per layer) loaded in %.1f s",
                 (double) ram_bytes / 1073741824.0, cover_, n_expert_, s);
    if (replaced_bytes)
        basalt::log::info("basalt deepseek41", "replacement base weights excluded from all tiers: %.2f GiB", replaced_bytes / 1073741824.0);
    return true;
}

int ExpertTier::ranked(int layer, int r) const {
    return rank_.empty() ? r : rank_[(size_t) layer * n_expert_ + r];
}

// The routing order from the profile file: per layer, the order its pairs come in.
bool ExpertTier::load_rank(std::string& err) {
    const std::string& path = m_.tiers().expert_profile;
    if (path.empty()) return true;
    uint64_t bytes = 0;
    if (!platform::file_size(path, bytes)) {
        err = "cannot read the expert profile " + path;
        return false;
    }
    std::vector<std::pair<int32_t, int32_t>> pairs;
    int64_t slots = 0;
    if (!core::read_expert_profile(core::FileRange{path, 0, bytes}, n_layers_, n_expert_, pairs, slots, err)) return false;
    if (pairs.size() != (size_t) n_layers_ * n_expert_) {
        err = "the expert profile " + path + " does not rank every expert";
        return false;
    }
    rank_.resize(pairs.size());
    std::vector<int32_t> next((size_t) n_layers_, 0);
    for (const auto& [layer, e] : pairs) rank_[(size_t) layer * n_expert_ + next[(size_t) layer]++] = e;
    return true;
}

// Copies the most routed RAM tier experts the main card does not hold yet into `n` slots of each layer.
void ExpertTier::seed_copies(int64_t n) {
    for (int l = 0; l < n_layers_; ++l) {
        for (int r = 0, placed = 0; r < n_expert_ && placed < n; ++r) {
            const int e = ranked(l,r);
            if (!pinned(l,e) || res_[(size_t)l*n_expert_+e] >= 0 || (peer_ && peer_->slot_ptr(l,e))) continue;
            uint64_t best = 0;
            const int slot = oldest_slot(l,true,best,expert_bytes(l,e));
            if (slot < 0 || owner_[slot] >= 0) continue;
            ++tick_;
            if (claim(l,e,blob_of(l,e),/*homes_too=*/true) >= 0) ++placed;
            if (queue_.size()+pending_.size() >= kMaxPending/2) { land_all(); }
        }
    }
    issue(-1, kAllPieces);
    cudaStreamSynchronize(copy_stream_);
    land_all();
}

const uint8_t* ExpertTier::blob_of(int layer, int e) const {
    if (const auto* v = m_.replacement(layer, e)) return v->host();
    const ExpertLayer& el = m_.pack().experts[(size_t) layer];
    const int32_t r = ram_idx_[(size_t) layer * n_expert_ + e];
    if (r >= 0) return ram_place(layer, r);
    if (!file_.data()) return nullptr;
    return file_.data() + m_.pack().experts_range.offset + el.offset + (uint64_t) e * el.blob_bytes;
}

bool ExpertTier::pinned(int layer, int e) const {
    if (const auto* v = m_.replacement(layer, e)) return v->host() != nullptr;
    return ram_idx_[(size_t) layer * n_expert_ + e] >= 0;
}

const kernels::cpu::NativeFmt& ExpertTier::fmt(int layer, int e) const {
    const auto* v = m_.replacement(layer, e);
    return v ? v->cpu : fmt(layer);
}

uint64_t ExpertTier::expert_bytes(int layer, int e) const {
    const auto* v = m_.replacement(layer, e);
    return v ? v->layout.bytes : m_.pack().experts[layer].blob_bytes;
}

uint8_t* ExpertTier::ram_place(int layer, int place) const {
    const uint64_t bytes = m_.pack().experts[layer].blob_bytes;
    const int base = (int) (ram_bytes_[layer] / bytes);
    return place < base ? ram_[layer] + (size_t) place * bytes : extra_ram_[layer][place - base]->data;
}

bool ExpertTier::ensure_spare(int layer) {
    if (swapping_[layer]) return false;
    if (spare_[layer] >= 0) return true;
    if (!free_places_[layer].empty()) {
        spare_[layer] = free_places_[layer].back(); free_places_[layer].pop_back(); return true;
    }
    if (!m_.has_overlay() && !exclusive_cache_) return false;
    // Mixed-size GPU homes can displace a base expert without releasing a base RAM place.
    // Allocate only that missing place, and never cross the same RAM floor used at startup.
    const uint64_t bytes = m_.pack().experts[layer].blob_bytes;
    auto r = m_.acquire_expert_ram(bytes);
    if (!r || r->registered() < bytes) return false;
    auto& v = extra_ram_[layer];
    size_t i = 0;
    while (i < v.size() && v[i]) ++i;
    if (i == v.size()) v.emplace_back();
    v[i] = std::move(r);
    spare_[layer] = (int) (ram_bytes_[layer] / bytes + i);
    return true;
}

void ExpertTier::return_place(int layer, int place) {
    if (place < 0) return;
    const int base = (int) (ram_bytes_[layer] / m_.pack().experts[layer].blob_bytes);
    if (place >= base) {
        if (spare_[layer] < 0) { spare_[layer] = place; return; }
        m_.recycle_expert_ram(std::move(extra_ram_[layer][place-base]));
    } else if (spare_[layer] < 0) spare_[layer] = place;
    else free_places_[layer].push_back(place);
}

void ExpertTier::land_home(int layer, int in, int out, int place) {
    TierScope profile(m_.tier_profile(),TierProfile::Publish);
    const size_t base = (size_t) layer * n_expert_;
    if (out >= 0 && m_.replacement(layer,out)) m_.publish_replacement_ram(layer,out);
    else if (place >= 0) ram_idx_[base + out] = place;
    if (m_.replacement(layer, in)) m_.release_replacement_ram(layer,in);
    else {
        const int old = ram_idx_[base + in];
        ram_idx_[base + in] = -1;
        return_place(layer, old);
    }
}

uint64_t ExpertTier::resident_ram_bytes() const {
    core::MemoryUsage usage;
    m_.memory_usage(usage);
    memory_usage(usage);
    return usage.host["resident_experts"];
}

void ExpertTier::pack_extra_ram() {
    if (!exclusive_cache_) return;
    // Transfers and CPU jobs have finished. Reuse holes in the compact arena before retaining extra blocks.
    for (int l = 0; l < n_layers_; ++l) {
        const uint64_t blob = m_.pack().experts[l].blob_bytes;
        const int base = (int)(ram_bytes_[l]/blob);
        if (spare_[l] < 0 && !free_places_[l].empty()) {
            spare_[l] = free_places_[l].back(); free_places_[l].pop_back();
        }
        if (spare_[l] >= base && !free_places_[l].empty()) {
            const int old = spare_[l];
            spare_[l] = free_places_[l].back();
            free_places_[l].pop_back();
            return_place(l,old);
        }
        for (int e = 0; e < n_expert_ && !free_places_[l].empty(); ++e) {
            auto& place = ram_idx_[(size_t)l*n_expert_+e];
            if (place < base) continue;
            const int into = free_places_[l].back();
            free_places_[l].pop_back();
            std::memcpy(ram_place(l,into),ram_place(l,place),blob);
            const int old = place;
            place = into;
            return_place(l,old);
        }
    }
}

bool ExpertTier::restore_experts(bool helper, int64_t first, std::string& err, std::vector<int>* backed) {
    settle();
    finish_swaps();
    struct Backing { int layer, expert, place; uint8_t* host; const uint8_t* device; };
    std::vector<Backing> back;
    const auto rollback = [&] {
        for (const auto& b : back) {
            if (m_.replacement(b.layer,b.expert)) m_.release_replacement_ram(b.layer,b.expert);
            else return_place(b.layer,b.place);
        }
    };
    const auto& owners = helper ? peer_owner_ : owner_;
    for (int64_t s = first; s < (int64_t)owners.size(); ++s) {
        const int e = owners[s];
        if (e < 0) continue;
        const int l = helper ? (int)(s/homes_.peer) : slot_layer((int)s);
        if (pinned(l,e)) continue;
        int place = -1;
        uint8_t* dst = nullptr;
        if (m_.replacement(l,e)) dst = m_.prepare_replacement_ram(l,e);
        else if (ensure_spare(l)) {
            place = spare_[l];
            spare_[l] = -1;
            dst = ram_place(l,place);
        }
        if (!dst) {
            rollback();
            err = "not enough RAM above the safety floor to return GPU experts"; return false;
        }
        back.push_back({l,e,place,dst,helper ? peer_->slot_address((int)s) : slot_ptr((int)s)});
    }
    for (const auto& b : back) {
        const bool copied = helper ? peer_->read_slot(b.device,b.host,expert_bytes(b.layer,b.expert),err) :
            cudaMemcpyAsync(b.host,b.device,expert_bytes(b.layer,b.expert),cudaMemcpyDeviceToHost,copy_stream_) == cudaSuccess;
        if (!copied) {
            if (helper) peer_->read_wait(err);
            else cudaStreamSynchronize(copy_stream_);
            rollback();
            err = "copying GPU experts back to RAM failed"; return false;
        }
    }
    if (helper ? !peer_->read_wait(err) : cudaStreamSynchronize(copy_stream_) != cudaSuccess) {
        rollback(); err = "waiting for GPU expert return failed"; return false;
    }
    for (const auto& b : back) {
        if (m_.replacement(b.layer,b.expert)) m_.publish_replacement_ram(b.layer,b.expert);
        else ram_idx_[(size_t)b.layer*n_expert_+b.expert] = b.place;
        if (backed) backed->push_back(b.layer*n_expert_+b.expert);
        ++returned_experts_;
        returned_bytes_ += expert_bytes(b.layer,b.expert);
    }
    return true;
}

void ExpertTier::release_peer_backing() {
    for (int i : peer_workspace_backed_) {
        const int layer = i/n_expert_, expert = i%n_expert_;
        if (m_.replacement(layer,expert)) m_.release_replacement_ram(layer,expert);
        else {
            auto& place = ram_idx_[(size_t)i];
            if (place >= 0) { return_place(layer,place); place = -1; }
        }
    }
    peer_workspace_backed_.clear();
    pack_extra_ram();
    m_.trim_expert_ram_pool(peer_workspace_pool_limit_);
}

bool ExpertTier::borrow_peer_workspace(uint64_t bytes, void*& storage, uint64_t& capacity, std::string& err) {
    storage = nullptr;
    capacity = 0;
    if (peer_workspace_bytes_) { err = "the helper's image workspace is already borrowed"; return false; }
    if (!peer_ || peer_->cache().slots() < 1 || bytes > (uint64_t)peer_->cache().mapped_bytes()) {
        err = "the helper expert tier has too little storage for the image encoder"; return false;
    }
    settle();
    if (peer_compute_ && !peer_compute_->finish(err)) return false;
    if (share_ && !share_->join(copy_stream_,err)) return false;
    if (cudaStreamSynchronize(copy_stream_) != cudaSuccess) { err = "waiting for the helper failed"; return false; }
    {
        const core::OnDevice on(peer_->device());
        if (cudaDeviceSynchronize() != cudaSuccess) { err = "waiting for the image device failed"; return false; }
    }
    const auto* base = peer_->slot_address(0);
    const uint64_t mapped = (uint64_t)peer_->cache().mapped_bytes();
    int32_t first = (int32_t)peer_->cache().slots() - 1;
    while (first > 0 && mapped - (uint64_t)(peer_->slot_address(first)-base) < bytes) --first;
    const uint64_t span = mapped - (uint64_t)(peer_->slot_address(first)-base);
    peer_workspace_pool_limit_ = m_.expert_ram_pool_bytes();
    if (!restore_experts(true,first,err,&peer_workspace_backed_)) {
        m_.trim_expert_ram_pool(peer_workspace_pool_limit_);
        return false;
    }
    if (!peer_->lend(first,peer_workspace_lent_,err)) { release_peer_backing(); return false; }
    storage = const_cast<uint8_t*>(peer_->slot_address(first));
    capacity = peer_workspace_bytes_ = span;
    basalt::log::info("basalt deepseek41", "vision GPU %d borrowed %llu MiB from %zu expert slots", peer_->device(),
                      (unsigned long long)(span >> 20), peer_workspace_lent_.size());
    return true;
}

bool ExpertTier::return_peer_workspace(std::string& err) {
    if (!peer_workspace_bytes_) return true;
    try {
        for (const auto& item : peer_workspace_lent_) {
            const int layer = item.first/n_expert_, expert = item.first%n_expert_;
            const uint8_t* host = blob_of(layer,expert);
            if (!host) { err = "a borrowed helper expert has no RAM backing"; return false; }
            peer_->slot_in(item.second,host,0,expert_bytes(layer,expert));
        }
    } catch (const std::exception& e) {
        peer_->read_wait(err);
        err = e.what();
        return false;
    }
    if (!peer_->read_wait(err)) return false;
    for (const auto& item : peer_workspace_lent_)
        peer_->admit(item.first/n_expert_,item.first%n_expert_,item.second);
    peer_workspace_lent_.clear();
    peer_workspace_bytes_ = 0;
    release_peer_backing();
    return true;
}

const uint8_t* ExpertTier::slot_ptr(int32_t slot) const {
    const int64_t nb = (int64_t) n_layers_ * per_layer_;
    if (slot < 0) return nullptr;
    if (slot < nb) return slot < cache_.slots() ? cache_.device_slot(slot) : nullptr;
    return slot-nb < ext_.slots() ? ext_.device_slot((int32_t)(slot-nb)) : nullptr;
}

int ExpertTier::main_slot(int layer, int rank) const {
    return homes_.main == 0 ? rank*n_layers_+layer : layer*(int)per_layer_+rank;
}

int ExpertTier::slot_layer(int slot) const {
    const int base = (int)(n_layers_*per_layer_);
    return slot < base ? (homes_.main == 0 ? slot%n_layers_ : slot/(int)per_layer_) : (slot-base)%n_layers_;
}

std::vector<int64_t> ExpertTier::cache_sizes(int n, bool extra) const {
    const auto sizes = slot_sizes(n,0,extra);
    if (!extra && homes_.main > 0) return sizes;
    std::vector<int64_t> result(sizes.size());
    for (int r = 0; r < n; ++r) for (int l = 0; l < n_layers_; ++l)
        result[(size_t)r*n_layers_+l] = sizes[(size_t)l*n+r];
    return result;
}

uint64_t ExpertTier::slot_capacity(int32_t slot) const {
    const int64_t nb = (int64_t)n_layers_ * per_layer_;
    const auto& arena = slot < nb ? cache_ : ext_;
    const int s = slot < nb ? slot : (int)(slot-nb);
    return arena.bytes_of(s+1)-arena.bytes_of(s);
}

std::vector<int64_t> ExpertTier::slot_sizes(int n, int rank_start, bool seed, bool conservative) const {
    std::vector<int64_t> result;
    if (n <= 0) return result;
    result.reserve((size_t)n_layers_*n);
    if (!m_.has_overlay()) {
        for (const auto& layer : m_.pack().experts) result.insert(result.end(), n, layer.blob_bytes);
        return result;
    }
    for (int l = 0; l < n_layers_; ++l) {
        std::vector<uint64_t> initial;
        if (conservative) {
            const auto& sorted = sorted_blob_bytes_[l];
            for (int i = 0; i < n; ++i) initial.push_back(sorted[sorted.size()-n+i]);
        } else {
            if (seed && !initial_seeds_.empty()) {
                for (int e : initial_seeds_[l]) {
                    if ((int)initial.size() == n) break;
                    initial.push_back(expert_bytes(l,e));
                }
            } else {
                for (int r = rank_start; r < n_expert_ && (int)initial.size() < n; ++r) {
                    const int e = ranked(l,r);
                    if (seed && (!pinned(l,e) || res_[(size_t)l*n_expert_+e] >= 0 || (peer_ && peer_->slot_ptr(l,e)))) continue;
                    initial.push_back(expert_bytes(l,e));
                }
            }
            initial.resize(n, m_.pack().experts[l].blob_bytes);
        }
        auto sizes = policy::mixed_slot_sizes(sorted_blob_bytes_[l],std::move(initial));
        result.insert(result.end(),sizes.begin(),sizes.end());
    }
    return result;
}

int ExpertTier::slots_fit(uint64_t free, uint64_t keep, int maximum, int rank_start, bool seed, bool conservative) const {
    if (free <= keep) return 0;
    const uint64_t budget = free-keep;
    if (!m_.has_overlay()) {
        uint64_t per_rank = 0;
        for (const auto& layer : m_.pack().experts) per_rank += (layer.blob_bytes + 255) / 256 * 256;
        return policy::uniform_slots_fit(budget, per_rank, maximum);
    }
    int fit = 0;
    for (int n = 1; n <= maximum; ++n) {
        uint64_t bytes = 0;
        for (const auto b : slot_sizes(n,rank_start,seed,conservative)) bytes += ((uint64_t)b+255)/256*256;
        if (bytes <= budget) fit = n;
    }
    return fit;
}

void ExpertTier::apply(const Pending& p) {
    const size_t in = (size_t) p.layer * n_expert_ + p.expert;
    res_[in] = p.slot;
    busy_[(size_t) p.slot] = 0;
    free_events_.push_back(p.event);
    if (p.home) land_home(p.layer, p.expert, p.out, p.place);
}

void ExpertTier::apply(const PeerMove& m) {
    peer_->admit(m.layer, m.out, -1);
    peer_->admit(m.layer, m.in, m.slot);
    land_home(m.layer, m.in, m.out, m.place);
}

// A layer's copies were claimed after its last lookup and before anything else reads its places: they land together,
// in the order they were claimed, before its next lookup.
void ExpertTier::land(int layer) {
    issue(layer, 0);
    size_t keep = 0;
    for (size_t i = 0; i < pending_.size(); ++i) {
        const Pending p = pending_[i];
        if (p.layer != layer) {
            pending_[keep++] = p;
            continue;
        }
        cudaEventSynchronize(events_[(size_t) p.event]);
        apply(p);
    }
    pending_.resize(keep);
    if (peer_moves_.empty()) return;
    std::string err;
    keep = 0;
    for (size_t i = 0; i < peer_moves_.size(); ++i) {
        const PeerMove m = peer_moves_[i];
        if (m.layer != layer) {
            peer_moves_[keep++] = m;
            continue;
        }
        if (peer_submitter_) {
            peer_submitter_->wait();
            if (!peer_->read_wait(err)) throw std::runtime_error(err);
        } else if (!peer_->read_until(m.in_mark, err)) throw std::runtime_error(err);
        apply(m);
    }
    peer_moves_.resize(keep);
}

void ExpertTier::land_all() {
    issue(-1, kAllPieces);
    for (const Pending& p : pending_) {
        { TierScope profile(m_.tier_profile(),TierProfile::DrainWait); cudaEventSynchronize(events_[(size_t)p.event]); }
        apply(p);
    }
    pending_.clear();
    claims_ = 0;
    if (peer_moves_.empty()) return;
    std::string err;
    if (peer_submitter_) peer_submitter_->wait();
    if (!peer_->read_wait(err)) throw std::runtime_error(err);
    for (const PeerMove& m : peer_moves_) apply(m);
    peer_moves_.clear();
}

void ExpertTier::settle() {
    submit_fetch();
    land_all();
    cudaStreamSynchronize(copy_stream_);
    cudaStreamSynchronize(fetch_stream_);
    cudaStreamSynchronize(return_stream_);
    finish_swaps();
    pack_extra_ram();
}

void ExpertTier::begin_token(int n_tokens) {
    if (peer_workspace_bytes_) {
        std::string err;
        if (!return_peer_workspace(err)) throw std::runtime_error(err);
    }
    TierScope profile(m_.tier_profile(),TierProfile::Boundary);
    token_ += (uint64_t) n_tokens;
    claims_ = 0;
    finish_swaps();
    // the swaps read their candidates' RAM places, which the copies of those layers move
    if (homes_.file > 0 && !m_.tiers().diag_tier_frozen)
        for (const auto& w : want_) land(w.first);
    start_swaps();
}

void ExpertTier::swap_worker() {
    std::unique_lock<std::mutex> lk(mu_);
    size_t next = 0;
    for (;;) {
        cv_.wait(lk, [&] { return quit_ || next < jobs_.size(); });
        if (quit_) return;
        const Swap sw = jobs_[next];
        lk.unlock();
        const ExpertLayer& el = m_.pack().experts[(size_t) sw.layer];
        uint8_t* dst = ram_place(sw.layer, sw.place);
        const bool ok = platform::pread_all(m_.experts_fd(), dst, el.blob_bytes,
                                            m_.pack().experts_range.offset + el.offset + (uint64_t) sw.expert * el.blob_bytes);
        lk.lock();
        if (!ok) jobs_[next].expert = -1;
        ++next;
        ++jobs_done_;
        cv_.notify_all();
    }
}

void ExpertTier::preread() {
    uint64_t left = kPrereadBytes;
    for (int r = 0; r < homes_.file; ++r) {
        for (int l = 0; l < n_layers_; ++l) {
            if ((size_t)r >= preread_experts_[l].size()) continue;
            const int e = preread_experts_[l][r];
            if (preread_quit_.load(std::memory_order_relaxed)) return;
            const ExpertLayer& el = m_.pack().experts[(size_t) l];
            if (left < el.blob_bytes) return;
            left -= el.blob_bytes;
            platform::advise_willneed(m_.experts_fd(), m_.pack().experts_range.offset + el.offset +
                                      (uint64_t) e * el.blob_bytes,
                                      el.blob_bytes);
        }
    }
}

void ExpertTier::finish_swaps() {
    if (flight_.empty()) return;
    {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return jobs_done_ == jobs_.size(); });
        for (size_t i = 0; i < flight_.size(); ++i) flight_[i].expert = jobs_[jobs_.size() - flight_.size() + i].expert;
    }
    for (const Swap& sw : flight_) {
        swapping_[(size_t) sw.layer] = 0;
        if (sw.expert < 0) continue;
        const size_t base = (size_t) sw.layer * n_expert_;
        const int32_t v = ram_idx_[base + sw.victim];
        ram_idx_[base + sw.expert] = sw.place;
        ram_idx_[base + sw.victim] = -1;
        spare_[(size_t) sw.layer] = v;
        spare_since_[(size_t) sw.layer] = token_;
        ++swapped_;
    }
    flight_.clear();
}

void ExpertTier::start_swaps() {
    if (homes_.file == 0 || want_.empty() || m_.tiers().diag_tier_frozen) {
        want_.clear();
        return;
    }
    std::vector<Swap> batch;
    for (const auto& w : want_) {
        if ((int) batch.size() >= policy::kSwapsPerToken) break;
        const int l = w.first, e = w.second;
        if (spare_[(size_t) l] < 0 || token_ <= spare_since_[(size_t) l] || pinned(l, e)) continue;
        bool dup = false;
        for (const Swap& b : batch) dup = dup || b.layer == l;
        if (dup) continue;
        const int v = policy::pick_victim(&ram_idx_[(size_t) l * n_expert_], &stamp_[(size_t) l * n_expert_], n_expert_, nullptr, 0);
        // a victim used in the last 16 tokens stays
        if (v < 0 || stamp_[(size_t) l * n_expert_ + v] + 16ull * (uint64_t) n_layers_ >= tick_) continue;
        batch.push_back({l, e, v, spare_[(size_t) l]});
    }
    want_.clear();
    if (batch.empty()) return;
    {
        std::lock_guard<std::mutex> g(mu_);
        jobs_.insert(jobs_.end(), batch.begin(), batch.end());
    }
    cv_.notify_all();
    for (const Swap& sw : batch) swapping_[(size_t) sw.layer] = 1;
    flight_ = std::move(batch);
}

bool ExpertTier::lookup(int layer, const int32_t* ids, int k, Lookup& lk, std::string& err) {
    land(layer);
    const uint64_t now = ++tick_;
    lk.k = k;
    for (int i = 0; i < k; ++i) {
        const int e = ids[i];
        if (e < 0 || e >= n_expert_) {
            err = "deepseek41: router returned a bad expert id";
            return false;
        }
        lk.experts[i] = e;
        lk.recent_misses[i] = 0;
        lk.fetched[i] = false;
        lk.peer[i] = nullptr;
        const size_t idx = (size_t) layer * n_expert_ + e;
        stamp_[idx] = now;
        const int32_t s = res_[idx];
        if (s >= 0) {
            lk.dev[i] = slot_ptr(s);
            lk.host[i] = nullptr;
            ++hits_;
            if (m_.replacement(layer,e)) ++overlay_hits_;
        } else if (peer_ && (lk.peer[i] = peer_->slot_ptr(layer, e))) {
            lk.dev[i] = nullptr;
            lk.host[i] = nullptr;
            ++peer_hits_;
            if (m_.replacement(layer,e)) ++overlay_peer_;
        } else {
            lk.dev[i] = nullptr;
            lk.host[i] = blob_of(layer, e);
            if (!lk.host[i]) {
                err = "deepseek41: no backing for routed expert " + std::to_string(e) +
                      " in layer " + std::to_string(layer);
                return false;
            }
            auto& history = misses_[idx];
            if (pinned(layer, e)) {
                ++ram_hits_;
            } else {
                ++file_hits_;
                if (policy::repeat_miss(history.last, token_, policy::kAdmitWindow) && want_.size() < 64) want_.emplace_back(layer, e);
            }
            lk.recent_misses[i] = history.record(token_);
            ++cpu_;
        }
    }
    return true;
}

ExpertTier::Source ExpertTier::locate(int layer, int e) const {
    Source r;
    const int32_t s = res_[(size_t) layer * n_expert_ + e];
    if (s >= 0) {
        r.dev = slot_ptr(s);
        return r;
    }
    if (peer_) r.peer = peer_->slot_ptr(layer, e);
    if (!r.peer && pinned(layer, e)) r.pinned = blob_of(layer, e);
    return r;
}

void ExpertTier::start_cpu(int layer, const float* x, const Lookup& lk, float* rows) {
    TierScope profile(m_.tier_profile(),TierProfile::CpuStart);
    Use use[kMaxK]{};
    for (int i = 0; i < lk.k; ++i) { use[i].nt = 1; use[i].tok[0] = 0; use[i].row[0] = i; }
    start_cpu_window(layer, x, lk, use, rows);
}

void ExpertTier::start_cpu_window(int layer, const float* x, const Lookup& lk, const Use* use, float* rows) {
    TierScope profile(m_.tier_profile(),TierProfile::CpuStart);
    const size_t n_embd = (size_t) m_.geo().n_embd;
    int types[kMaxK]{};
    bool quantized[kMaxK][kMaxWindow]{};
    int n = 0, ntypes = 0;
    for (int i = 0; i < lk.k; ++i) {
        if (lk.dev[i] || lk.peer[i]) continue;
        const auto& f = fmt(layer, lk.experts[i]);
        int q = 0;
        while (q < ntypes && types[q] != f.gu_act) ++q;
        if (q == ntypes) types[ntypes++] = f.gu_act;
        auto& j = cpu_jobs_[n++];
        j = {};
        j.native_format = &f;
        j.blob = lk.host[i];
        j.nt = use[i].nt;
        if (m_.replacement(layer,lk.experts[i])) overlay_cpu_ += j.nt;
        for (int r = 0; r < j.nt; ++r) {
            const int t = use[i].tok[r];
            auto* act = act_ + ((size_t) q * kMaxWindow + t) * kernels::cpu::kNativeActBytes;
            if (!quantized[q][t]) {
                kernels::cpu::native_quant_act(f, x + (size_t) t * n_embd, act);
                quantized[q][t] = true;
            }
            j.nact[r] = act;
            j.out[r] = rows + (size_t) use[i].row[r] * n_embd;
        }
    }
    if (n) pool_->start_split_multi_native(fmt(layer), cpu_jobs_, n);
}

void ExpertTier::finish_cpu() { TierScope profile(m_.tier_profile(),TierProfile::CpuWait); pool_->finish_split_multi_native(); }

int32_t ExpertTier::oldest_slot(int layer, bool homes_too, uint64_t& best, uint64_t bytes) const {
    int32_t slot = -1;
    best = ~0ull;
    uint64_t best_capacity = ~0ull;
    bool best_empty = false;
    auto scan = [&](int64_t lo, int64_t n, int stride) {
        for (int64_t r = 0; r < n; ++r) {
            const int64_t s = lo+r*stride;
            if (!slot_ptr((int)s)) break;
            if (busy_[(size_t) s]) continue;
            const uint64_t capacity = slot_capacity((int)s);
            if (capacity < bytes) continue;
            const int32_t o = owner_[(size_t) s];
            const uint64_t st = o < 0 ? 0 : stamp_[(size_t) layer * n_expert_ + o];
            const bool empty = o < 0;
            if (st != tick_ && ((empty && !best_empty) || (empty == best_empty &&
                (capacity < best_capacity || (capacity == best_capacity && st < best))))) {
                best = st;
                best_capacity = capacity;
                best_empty = empty;
                slot = (int32_t) s;
            }
        }
    };
    if (homes_.main == 0 || (homes_too && spare_[(size_t) layer] >= 0 && !swapping_[(size_t) layer]))
        scan(main_slot(layer,0),per_layer_,homes_.main == 0 ? n_layers_ : 1);
    if (ext_per_layer_) scan((int64_t)n_layers_*per_layer_+layer,ext_per_layer_,n_layers_);
    return slot;
}

bool ExpertTier::move_to_peer(int layer, int e, const uint8_t* src) {
    if (peer_workspace_bytes_) return false;
    TierScope profile(m_.tier_profile(),TierProfile::PeerSubmit);
    if (homes_.peer == 0 || swapping_[(size_t) layer]) return false;
    const size_t base = (size_t) layer * n_expert_;
    int32_t slot = -1;
    uint64_t oldest = ~0ull;
    uint64_t best_capacity = ~0ull;
    const uint64_t bytes = expert_bytes(layer,e);
    for (int64_t s = (int64_t) layer * homes_.peer; s < (int64_t) (layer + 1) * homes_.peer; ++s) {
        const uint64_t st = stamp_[base + peer_owner_[(size_t) s]];
        const auto& arena = peer_->cache();
        const uint64_t capacity = arena.bytes_of(s+1)-arena.bytes_of(s);
        if (capacity < bytes) continue;
        if (st != tick_ && (capacity < best_capacity || (capacity == best_capacity && st < oldest))) {
            oldest = st;
            best_capacity = capacity;
            slot = (int32_t) s;
        }
    }
    uint64_t on_main = ~0ull;
    oldest_slot(layer, /*homes_too=*/true, on_main, bytes);
    if (slot < 0 || oldest + policy::kPeerMoveAge >= on_main) return false;
    const int victim = peer_owner_[(size_t) slot];
    const bool needs_home = !pinned(layer, victim);
    const bool replacement_home = needs_home && m_.replacement(layer,victim);
    uint8_t* out = replacement_home ? m_.prepare_replacement_ram(layer,victim) : nullptr;
    if (replacement_home && !out) return false;
    if (needs_home && !replacement_home && !ensure_spare(layer)) return false;
    const int32_t place = needs_home && !replacement_home ? spare_[layer] : -1;
    if (place >= 0) { spare_[layer] = -1; out = ram_place(layer,place); }
    const uint64_t out_bytes = needs_home ? expert_bytes(layer,victim) : 0;
    PeerMove move{layer, e, victim, slot, place, out, src, out_bytes, bytes};
    if (profile_peer_exchange_ && !host_vmm_bytes_.empty()) {
        using Clock = std::chrono::steady_clock;
        const auto stamp = [this]() { return profile_peer_exchange_ ? Clock::now() : Clock::time_point{}; };
        const auto start = stamp();
        int prev;
        if (cudaGetDevice(&prev) != cudaSuccess || cudaSetDevice(peer_->device()) != cudaSuccess)
            throw std::runtime_error("deepseek41: cannot select helper for expert exchange");
        const auto read = stamp();
        auto rc = out_bytes ? cudaMemcpy(out, peer_->slot_address(slot), out_bytes, cudaMemcpyDeviceToHost) : cudaSuccess;
        const auto write = stamp();
        if (rc == cudaSuccess)
            rc = cudaMemcpy(const_cast<uint8_t*>(peer_->slot_address(slot)), src, bytes, cudaMemcpyHostToDevice);
        const auto restore = stamp();
        auto restored = cudaSetDevice(prev);
        if (rc != cudaSuccess || restored != cudaSuccess)
            throw std::runtime_error("deepseek41: direct helper exchange failed");
        if (profile_peer_exchange_) {
            const auto ns = [](Clock::duration d) { return std::chrono::duration_cast<std::chrono::nanoseconds>(d).count(); };
            peer_exchange_ns_ += ns(stamp() - start);
            peer_read_ns_ += ns(write - read);
            peer_write_ns_ += ns(restore - write);
            ++peer_exchanges_;
        }
        move.out_off = out_bytes; move.in_off = bytes; move.in_mark = 0;
    }
    peer_moves_.push_back(move);
    peer_owner_[(size_t) slot] = e;
    ++promoted_;
    if (m_.replacement(layer,e) || m_.replacement(layer,victim)) ++overlay_swaps_;
    return true;
}

int32_t ExpertTier::claim(int layer, int e, const uint8_t* src, bool homes_too, bool device_source) {
    TierScope profile(m_.tier_profile(),TierProfile::Claim);
    const auto* replacement = m_.replacement(layer, e);
    if (homes_too && homes_.main > 0) ensure_spare(layer);
    if (claims_ >= (int) kMaxPending || free_events_.empty()) {
        ++dropped_;
        return -1;
    }
    uint64_t best = ~0ull;
    const int32_t slot = oldest_slot(layer, homes_too, best, expert_bytes(layer,e));
    if (slot < 0 || best == tick_) return -1;   // every slot is in use by this very lookup
    const int32_t o = owner_[(size_t) slot];
    const bool home = exclusive_cache_ || (homes_.main > 0 && slot < (int64_t) n_layers_ * per_layer_);
    const bool needs_home = home && o >= 0 && !pinned(layer,o);
    const bool replacement_home = needs_home && m_.replacement(layer,o);
    uint8_t* home_to = replacement_home ? m_.prepare_replacement_ram(layer,o) : nullptr;
    if (replacement_home && !home_to) return -1;
    if (needs_home && !replacement_home && !ensure_spare(layer)) return -1;
    if (o >= 0) res_[(size_t) layer * n_expert_ + o] = -1;
    owner_[(size_t) slot] = e;
    busy_[(size_t) slot] = 1;
    stamp_[(size_t) layer * n_expert_ + e] = tick_;
    const uint64_t bytes = replacement ? replacement->layout.bytes : m_.pack().experts[(size_t) layer].blob_bytes;
    uint8_t* dst = const_cast<uint8_t*>(slot_ptr(slot));
    const uint64_t out_bytes = needs_home ? expert_bytes(layer,o) : 0;
    int32_t place = -1;
    if (needs_home && !replacement_home) {
        place = spare_[(size_t) layer];
        spare_[(size_t) layer] = -1;
        home_to = ram_place(layer,place);
    }
    const int ev = free_events_.back();
    free_events_.pop_back();
    ++promoted_;
    ++claims_;
    if (replacement) ++overlay_copies_;
    Queued q{{layer,e,slot,ev,o,place,home},dst,src,home_to,bytes};
    q.device_source=device_source;
    if(device_source) fetch_promotions_=true;
    q.home_bytes = out_bytes;
    queue_.push_back(q);
    return slot;
}

void ExpertTier::mark_gap(cudaStream_t main) {
    bool left = !queue_.empty();
    for (const PeerMove& m : peer_moves_) left = left || m.in_off < m.in_bytes;
    gap_marked_ = left && cudaEventRecord(gap_ev_, main) == cudaSuccess;
    if (fetch_scratch_used_) {
        if (fetch_promotions_) {
            issue(-1, kAllPieces);
            cudaEventRecord(fetch_scratch_done_[fetch_ring_],copy_stream_);
            fetch_promotions_=false;
        } else cudaEventRecord(fetch_scratch_done_[fetch_ring_],main);
        fetch_scratch_used_=false; fetch_scratch_fenced_[fetch_ring_]=true;
        fetch_ring_=(fetch_ring_+1)%kFetchRings;
    }
}
void ExpertTier::issue_peer(int layer, int pieces) {
    const uint64_t P = policy::kPieceBytes;
    const auto piece = [&](PeerMove& m) {
        if (m.out_off < m.out_bytes) {
            const auto bytes = std::min(P, m.out_bytes - m.out_off);
            if (peer_submitter_) peer_submitter_->submit({m.slot, m.to, nullptr, m.out_off, bytes});
            else peer_->slot_out(m.slot, m.to, m.out_off, bytes);
            m.out_off += P;
            return;
        }
        const auto bytes = std::min(P, m.in_bytes - m.in_off);
        if (peer_submitter_) peer_submitter_->submit({m.slot, nullptr, m.from, m.in_off, bytes});
        else peer_->slot_in(m.slot, m.from, m.in_off, bytes);
        m.in_off += P;
        if (m.in_off >= m.in_bytes) m.in_mark = peer_submitter_ ? 0 : peer_->mark_in();
    };
    for (PeerMove& m : peer_moves_) if (m.layer == layer)
        while (m.in_off < m.in_bytes) piece(m);
    for (PeerMove& m : peer_moves_)
        while (m.in_off < m.in_bytes && pieces > 0) { piece(m); --pieces; }
}

void ExpertTier::issue(int layer, int pieces) {
    TierScope profile(m_.tier_profile(),TierProfile::Pump);
    if (gap_marked_) {
        cudaStreamWaitEvent(copy_stream_, gap_ev_, 0);
        cudaStreamWaitEvent(return_stream_, gap_ev_, 0);
        if (!peer_moves_.empty()) peer_->copies_after(gap_ev_);
    }
    gap_marked_ = false;
    // the second card's own bus: what one head leaves it (the rest at the window's end)
    if (!peer_moves_.empty()) issue_peer(layer, pieces == kAllPieces ? pieces : std::min(pieces, policy::kPeerGapPieces));
    if (queue_.empty()) return;
    const uint64_t P = policy::kPieceBytes;
    // A copy engine takes copies in the order they were queued, whatever their stream: a copy queued later (a layer's
    // bus fetch, a window's rows, the drafter's result) would wait for every piece queued before it. A small operation
    // between two pieces lets it go first, after one piece at most. The home's bytes go out on a stream of their own, so
    // the pieces in do not wait for them, nor they for the copies in.
    const auto out = [&](Queued& q) {   // every piece out of the home q replaces, and the event behind them
        if (!q.home_to || q.home_off >= q.home_bytes) return;
        for (; q.home_off < q.home_bytes; q.home_off += P) {
            cudaMemcpyAsync(q.home_to + q.home_off, q.dst + q.home_off, std::min(P, q.home_bytes - q.home_off), cudaMemcpyDeviceToHost,
                            return_stream_);
            cudaMemsetAsync(q.dst + q.home_off, 0, 4, return_stream_);   // out already; the pieces in overwrite it
        }
        cudaEventRecord(return_events_[(size_t) q.p.event], return_stream_);
    };
    const auto piece = [&](Queued& q) {   // one piece in, the first once the home's bytes are out
        if (q.home_to && q.off == 0) {
            out(q);
            cudaStreamWaitEvent(copy_stream_, return_events_[(size_t) q.p.event], 0);
        }
        if (q.device_source) {
            cudaMemcpyAsync(q.dst, q.src, q.bytes, cudaMemcpyDeviceToDevice, copy_stream_);
            q.off = q.bytes;
        } else {
            cudaMemsetAsync(q.dst + q.off, 0, 4, copy_stream_);   // the piece overwrites it
            cudaMemcpyAsync(q.dst + q.off, q.src + q.off, std::min(P, q.bytes - q.off), cudaMemcpyHostToDevice, copy_stream_);
            q.off += P;
        }
        if (q.off >= q.bytes) {
            cudaEventRecord(events_[q.p.event], copy_stream_);
            pending_.push_back(q.p);
        }
    };
    for (Queued& q : queue_) if (q.p.layer == layer)
        while (q.off < q.bytes) piece(q);
    for (size_t i = 0; i < queue_.size() && pieces > 0; ++i) {
        if (queue_[i].off < queue_[i].bytes)   // the next home goes out while this promotion comes in
            for (size_t j = i + 1; j < queue_.size(); ++j)
                if (queue_[j].home_to && queue_[j].off == 0) {
                    out(queue_[j]);
                    break;
                }
        for (Queued& q = queue_[i]; q.off < q.bytes && pieces > 0; --pieces) piece(q);
    }
    queue_.erase(std::remove_if(queue_.begin(), queue_.end(), [](const Queued& q) { return q.off >= q.bytes; }), queue_.end());
}

void ExpertTier::pump(int layer, int pieces) {
    issue(layer, pieces);
    if (layer >= 0) land(layer);
}

int ExpertTier::fetch(int layer, Lookup& lk) {
    TierScope profile(m_.tier_profile(),TierProfile::Fetch);
    if (m_.tiers().diag_tier_frozen) return 0;
    int np = 0, nf = 0;
    for (int i = 0; i < lk.k; ++i)
        if (!lk.dev[i] && !lk.peer[i]) ++(pinned(layer, lk.experts[i]) ? np : nf);
    int left = policy::bus_count(np,nf,m_.tiers().gpu_single_miss);
    int far = peer_ && !m_.has_overlay() ? std::min(policy::peer_bus_count(np, nf), peer_->scratch_free()) : 0;
    int started = 0, scratch = 0;
    for (int i = lk.k - 1; i >= 0 && left > 0; --i) {
        if (lk.dev[i] || lk.peer[i] || !pinned(layer, lk.experts[i])) continue;   // RAM sources only
        --left;
        if (far > 0) {
            --far;
            lk.peer[i] = peer_->fetch(lk.experts[i], lk.host[i], expert_bytes(layer, lk.experts[i]));
            if (lk.peer[i]) {
                lk.fetched[i] = true;
                --cpu_;
                ++fetched_;
                continue;
            }
        }
        if (scratch==fetch_slots_) continue;
        ++started;
        uint8_t* dst=fetch_scratch_+((uint64_t)fetch_ring_*fetch_slots_+scratch++)*fetch_stride_;
        fetch_queue_.push_back({dst,lk.host[i],expert_bytes(layer,lk.experts[i])});
        lk.dev[i]=dst; fetch_scratch_used_=true;
        lk.host[i] = nullptr;
        lk.fetched[i] = true;
        --cpu_;
        ++fetched_;
    }
    return started;
}

void ExpertTier::submit_fetch() {
    if (fetch_queue_.empty()) return;
    if (fetch_scratch_fenced_[fetch_ring_]) cudaStreamWaitEvent(fetch_stream_,fetch_scratch_done_[fetch_ring_],0);
    for (const auto& q:fetch_queue_) {
        cudaMemcpyAsync(q.dst,q.src,q.bytes,cudaMemcpyHostToDevice,fetch_stream_);
    }
    cudaEventRecord(fetch_ev_,fetch_stream_);
    fetch_queue_.clear();
}

void ExpertTier::await_fetch(cudaStream_t main) { cudaStreamWaitEvent(main, fetch_ev_, 0); }

void ExpertTier::promote(int layer, const Lookup& lk) {
    if (m_.tiers().diag_tier_frozen) return;
    const int main_misses = m_.tiers().main_promote_misses;
    for (int i = 0; i < lk.k; ++i) {
        if ((lk.dev[i] && !lk.fetched[i]) || (lk.peer[i] && !lk.fetched[i]) ||
            lk.recent_misses[i] < 2 || !pinned(layer,lk.experts[i])) continue;
        if (!lk.fetched[i] && move_to_peer(layer,lk.experts[i],lk.host[i])) continue;
        if (lk.recent_misses[i] < main_misses) continue;
        if (lk.fetched[i] && lk.dev[i])
            claim(layer,lk.experts[i],lk.dev[i],/*homes_too=*/false,/*device_source=*/true);
        else  // A temporary peer fetch retains its host source.
            claim(layer,lk.experts[i],lk.host[i],/*homes_too=*/true);
    }
}

void ExpertTier::prompt_workspace(const void* owner, uint64_t bytes, uint64_t margin) {
    if (bytes) prompt_workspaces_[owner] = {bytes,margin};
    else prompt_workspaces_.erase(owner);
}

uint64_t ExpertTier::refill_reserve() const {
    uint64_t held = 0, margin = prefillp::kShortMargin;
    for (const auto& entry : prompt_workspaces_) { held += entry.second.first; margin = std::max(margin,entry.second.second); }
    // This arena releases its workspace at the end of every prompt. Re-lend on the next read.
    const bool loan_on_next_prompt = prompt_workspaces_.empty() && malloc_slab_backed();
    return policy::refill_reserve(loan_on_next_prompt ? 0 : reserve_,held,margin,free_floor_,token_ ? 0 : modules_);
}

bool ExpertTier::shrink_cache(core::ExpertCache& arena, uint64_t offset, uint64_t keep, std::string& err) {
    const int64_t before = arena.slots();
    const int64_t after = arena.slots_after_shrink((int64_t)keep);
    if (after >= before) return true;
    if (!restore_experts(false,(int64_t)offset+after,err)) return false;
    const bool ok = arena.shrink((int64_t)keep,err);
    // Returned weights are safe in RAM even if a driver releases only part of the requested tail.
    for (int64_t s = (int64_t)offset+after; s < (int64_t)offset+before; ++s) {
        const int e = owner_[s];
        if (e >= 0) res_[(size_t)slot_layer((int)s)*n_expert_+e] = -1;
        owner_[s] = -1;
    }
    return ok;
}

bool ExpertTier::lend(uint64_t wanted, std::string& err) {
    settle();
    uint64_t free = 0, total = 0;
    std::string why;
    if (!core::device_memory(-1, free, total, why)) { err="cannot query free memory before lending experts: " + why; return false; }
    const uint64_t before = free;
    const auto base_before = cache_.mapped_bytes(), extra_before = ext_.mapped_bytes();
    const auto loan = [&](core::ExpertCache& arena, uint64_t offset) {
        while (free < wanted && arena.segmented() && arena.mapped_bytes()) {
            const auto mapped = (uint64_t) arena.mapped_bytes();
            const uint64_t missing = wanted-free;
            const uint64_t keep = arena.malloc_slab_backed()
                ? arena.malloc_slab_boundary(missing < mapped ? mapped-missing : 0)
                : policy::loan_keep(mapped,missing);
            if (!shrink_cache(arena,offset,keep,err)) return false;
            if (!core::device_memory(-1, free, total, err)) return false;
            // Released mappings and allocator free bytes can differ; require mapping progress before another loan.
            if ((uint64_t) arena.mapped_bytes() >= mapped) break;
        }
        return true;
    };
    if (!loan(ext_,(uint64_t)n_layers_*per_layer_) || !loan(cache_,0)) return false;
    if (free > before) basalt::log::info("basalt deepseek41","expert VRAM loan: %.2f GiB released, %.2f GiB free",(free-before)/1073741824.,free/1073741824.);
    if (free >= wanted) return true;
    err = "expert lending leaves " + std::to_string(free >> 20) + " MiB free, needs " +
        std::to_string((wanted + (1ull << 20) - 1) >> 20) + " MiB (before " +
        std::to_string(before >> 20) + ", total " + std::to_string(total >> 20) +
        "; base " + std::to_string(base_before >> 20) + " -> " + std::to_string(cache_.mapped_bytes() >> 20) +
        ", extra " + std::to_string(extra_before >> 20) + " -> " + std::to_string(ext_.mapped_bytes() >> 20) +
        ", base slots " + std::to_string(cache_.slots()) + ", segmented " + std::to_string(cache_.segmented()) + ")";
    return false;
}

void ExpertTier::shrink() {
    std::string err;
    if (ext_.segmented() && !shrink_cache(ext_,(uint64_t)n_layers_*per_layer_,0,err))
        basalt::log::warn("basalt deepseek41","expert VRAM stays resident: %s",err.c_str());
}

void ExpertTier::lend_all() {
    shrink();
    std::string err;
    if (ext_.slots() == 0 && homes_.main == 0 && cache_.valid() && !shrink_cache(cache_,0,0,err))
        basalt::log::warn("basalt deepseek41","expert VRAM stays resident: %s",err.c_str());
}

bool ExpertTier::reclaim(std::string& err) {
    uint64_t free = 0, total = 0;
    std::string why;
    if (!core::device_memory(-1, free, total, why)) { err="cannot query free memory before reclaiming experts: " + why; return false; }
    const uint64_t keep = refill_reserve();
    if (cache_.valid() && free > keep) {
        const auto before = cache_.slots();
        if (!cache_.grow(cache_.mapped_bytes()+(int64_t)(free-keep),err)) return false;
        if (cache_.slots() > before) {
            seed_copies(n_expert_); finish_swaps(); pack_extra_ram();
            if (malloc_slab_backed()) {
                const auto filled = std::count_if(owner_.begin(), owner_.begin() + cache_.slots(), [](int e) { return e >= 0; });
                basalt::log::info("basalt deepseek41", "expert malloc slabs: %lld -> %lld of %lld slots; %lld filled",
                    (long long) before, (long long) cache_.slots(), (long long) cache_.full_slots(), (long long) filled);
            }
        }
    }
    return true;
}

void ExpertTier::grow() {
    if (!cache_.valid()) return;
    std::string err;
    if (!reclaim(err)) { basalt::log::warn("basalt deepseek41","expert cache refill: %s",err.c_str()); return; }
    uint64_t free = 0, total = 0;
    std::string why;
    if (!core::device_memory(-1, free, total, why)) {
        if (!unread_said_) basalt::log::warn("basalt deepseek41", "VRAM expert tier cannot grow: %s", why.c_str());
        unread_said_ = true;
        return;
    }
    const uint64_t keep = refill_reserve();
    if (ext_.segmented() && free > keep) {
        const auto before = ext_.slots();
        if (!ext_.grow(ext_.mapped_bytes()+(int64_t)(free-keep),err))
            basalt::log::warn("basalt deepseek41","expert cache refill: %s",err.c_str());
        if (ext_.slots() > before) {
            seed_copies(n_expert_); finish_swaps(); pack_extra_ram();
            if (ext_.malloc_slab_backed())
                basalt::log::info("basalt deepseek41", "extra expert slabs: %lld -> %lld of %lld slots (%.2f GiB backed)",
                    (long long)before, (long long)ext_.slots(), (long long)ext_.full_slots(),
                    (double)ext_.mapped_bytes()/1073741824.0);
        }
    }
}

void ExpertTier::set_free_floor(int64_t mib) { free_floor_ = mib < 0 ? start_floor_ : (uint64_t) mib << 20; }

void ExpertTier::report(const char* tag) const {
    if (m_.has_overlay()) {
        basalt::log::info("basalt deepseek41", "overlay %s: primary=%lld peer=%lld cpu_rows=%lld admissions=%lld peer_swaps=%lld", tag,
                         (long long) overlay_hits_, (long long) overlay_peer_, (long long) overlay_cpu_,
                         (long long) overlay_copies_, (long long) overlay_swaps_);
        uint64_t primary = 0, helper = 0;
        uint64_t primary_duplicates = 0, helper_duplicates = 0;
        const int64_t base = (int64_t)n_layers_*per_layer_;
        for (size_t s = 0; s < owner_.size(); ++s) if (owner_[s] >= 0) {
            const int layer = slot_layer((int)s);
            primary += expert_bytes(layer,owner_[s]);
            primary_duplicates += pinned(layer,owner_[s]);
        }
        for (size_t s = 0; s < peer_owner_.size(); ++s) if (peer_owner_[s] >= 0) {
            helper += expert_bytes((int)(s/homes_.peer),peer_owner_[s]);
            helper_duplicates += pinned((int)(s/homes_.peer),peer_owner_[s]);
        }
        basalt::log::info("basalt deepseek41","expert backing %s: RAM duplicates primary=%llu helper=%llu; returned %llu experts, %.3f GiB from VRAM",tag,
                         (unsigned long long)primary_duplicates,(unsigned long long)helper_duplicates,
                         (unsigned long long)returned_experts_,returned_bytes_/1073741824.);
        basalt::log::info("basalt deepseek41", "slot storage %s: primary %.3f/%.3f GiB, helper %.3f/%.3f GiB (weight/allocated)", tag,
                          primary/1073741824., (cache_.mapped_bytes()+ext_.mapped_bytes())/1073741824.,
                          helper/1073741824., (peer_ ? peer_->cache().mapped_bytes() : 0)/1073741824.);
    }
    if (profile_peer_exchange_)
        basalt::log::info("basalt deepseek41", "peer exchange %s: count=%llu host_elapsed_ms=%.3f d2h_ms=%.3f memcpy_ms=%.3f h2d_ms=%.3f",
                         tag, (unsigned long long) peer_exchanges_, peer_exchange_ns_ / 1e6,
                         peer_read_ns_ / 1e6, peer_host_ns_ / 1e6, peer_write_ns_ / 1e6);
    const int64_t n = hits_ + cpu_ + fetched_ + peer_hits_;
    if (n == 0) return;
    basalt::log::info("basalt deepseek41", "TIER %s: %lld lookups: %.1f%% VRAM, %.1f%% second card, %.1f%% bus copy, %.1f%% CPU (file %.1f%%); promoted %lld, dropped %lld, swapped %lld", tag,
                 (long long) n, 100.0 * (double) hits_ / (double) n, 100.0 * (double) peer_hits_ / (double) n, 100.0 * (double) fetched_ / (double) n,
                 100.0 * (double) cpu_ / (double) n, 100.0 * (double) file_hits_ / (double) n, (long long) promoted_, (long long) dropped_, (long long) swapped_);
}

}  // namespace basalt::models::deepseek41
