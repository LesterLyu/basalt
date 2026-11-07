// src/models/deepseek41/tier.cpp - the DeepSeek-V4.1 routed-expert tiers.
#include "basalt/log.hpp"
#include "models/deepseek41/tier.hpp"

#include "basalt/core/load_progress.hpp"
#include "basalt/core/vmm.hpp"
#include <optional>
#include <stdexcept>

#include "basalt/core/expert_source.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "models/deepseek41/model.hpp"
#include "models/deepseek41/prefill_plan.hpp"
#include "models/deepseek41/tier_slots.hpp"

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
    if (peer_) usage.device(peer_->device(), "expert_helper", peer_->cache().mapped_bytes());
    for (size_t l = 0; l < ram_.size(); ++l)
        if (ram_[l]) usage.host["resident_experts"] += ram_bytes_[l];
    for (uint64_t n : host_vmm_bytes_) usage.host["pinned_experts"] += n;
    for (const auto& pin : ram_pins_) {
        usage.host["pinned_experts"] += pin.registered_bytes;
        usage.host["locked_experts"] += pin.locked_bytes;
    }
    for (size_t l = 0; l < extra_ram_.size(); ++l) for (const auto& r : extra_ram_[l]) if (r.ptr) {
        usage.host["resident_experts"] += m_.pack().experts[l].blob_bytes;
        usage.host["pinned_experts"] += r.mapped ? r.mapped : r.pin.registered_bytes;
        usage.host["locked_experts"] += r.pin.locked_bytes;
    }
}

namespace {

constexpr size_t kMaxPending = 128;              // promotions in flight
constexpr uint64_t kPrereadBytes = 12ull << 30;  // file-backed experts read ahead into the page cache

// The usual routing order of this model's experts: the RAM and VRAM tiers start from the head of it.
constexpr int kRankLayers = 40, kRankExperts = 384;
}  // namespace

// The helper's async copies can block the submitting Windows thread. Queue the submissions,
// then fence the refill stream before publishing the new expert homes.
struct ExpertTier::PeerSubmitter {
    struct Move { int slot; uint8_t* out; const uint8_t* in; uint64_t out_bytes, in_bytes; };
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
            try { peer.exchange_slot(job.slot, job.out, job.out_bytes, job.in, job.in_bytes); }
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

ExpertTier::ExpertTier(const DsModel& model) : m_(model) {}

ExpertTier::~ExpertTier() {
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
    if (peer_ && !peer_moves_.empty()) {
        std::string e;
        peer_->read_wait(e);
    }
    if (copy_stream_) {
        flush_queue();
        cudaStreamSynchronize(copy_stream_);
    }
    if (fetch_stream_) cudaStreamSynchronize(fetch_stream_);
    for (cudaEvent_t e : events_) cudaEventDestroy(e);
    if (fetch_ev_) cudaEventDestroy(fetch_ev_);
    if (gap_ev_) cudaEventDestroy(gap_ev_);
    if (copy_stream_) cudaStreamDestroy(copy_stream_);
    if (fetch_stream_) cudaStreamDestroy(fetch_stream_);
    for (size_t l = 0; l < ram_.size(); ++l) {
        if (ram_[l] == nullptr) continue;
        if (l < ram_pins_.size()) core::unpin_ram(ram_[l], ram_pins_[l]);
        if (!host_vmm_bytes_.empty()) core::vmm::free_host(ram_[l], host_vmm_bytes_[l]);
        else platform::free_aligned(ram_[l]);
    }
    platform::free_aligned(act_);
    for (auto& layer : extra_ram_) for (auto& r : layer) if (r.ptr) {
        if (r.mapped) core::vmm::free_host(r.ptr, r.mapped);
        else { core::unpin_ram(r.ptr, r.pin); platform::free_aligned(r.ptr); }
    }
}

bool ExpertTier::open(const Reserve& reserve, float* h_rows, std::string& err) {
    const Geometry& g = m_.geo();
    n_layers_ = g.n_layers;
    n_expert_ = g.n_expert;
    uint64_t one_each_ram = 0;
    for (int l = 0; l < n_layers_; ++l) {
        const auto& e = m_.pack().experts[l];
        one_each_ram += e.blob_bytes;
        std::vector<uint64_t> sizes;
        for (int expert = 0; expert < n_expert_; ++expert) sizes.push_back(std::max<uint64_t>(e.blob_bytes, expert_bytes(l,expert)));
        std::sort(sizes.begin(), sizes.end());
        sorted_blob_bytes_.push_back(std::move(sizes));
    }
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        err = "deepseek41: cannot query the free device memory";
        return false;
    }
    auto slots_in = [&](uint64_t room, uint64_t left_free) {
        return slots_fit(room, left_free, n_expert_);
    };
    if (!load_rank(err)) return false;
    base_count_.resize(n_layers_);
    for (int l = 0; l < n_layers_; ++l) {
        base_count_[l] = n_expert_;
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
        if (peer_->open_plain(m_.peer_device(), n_layers_, n_expert_, g.n_embd, g.n_ff_exp, h_rows, pe)) {
            const uint64_t reserve = (uint64_t) m_.tiers().peer_reserve_mib << 20;
            peer_room = peer_->room();
            peer_slots = slots_fit(peer_room, reserve, n_expert_, 0, false, true);
        } else {
            basalt::log::warn("basalt deepseek41", "second card not used: %s", pe.c_str());
            peer_.reset();
        }
    }

    if (m_.tiers().diag_tier_peer >= 0) peer_slots = std::min(peer_slots, m_.tiers().diag_tier_peer);
    main_homes = std::min(main_homes, n_expert_ - peer_slots);
    copies = std::min(copies, n_expert_ - peer_slots);

    // The prompt and image buffers first, so the free memory the tier reads already lacks them. The ring holds five layers
    // of the second card's homes, or its cap when experts may stay in the file.
    auto available = [] {
        core::detail::HostMemory hm;
        return core::detail::host_available_memory(hm) ? hm.available : 0;
    };
    const uint64_t ram_reserve = platform::ram_reserve();
    const int peer_homes = std::min(peer_slots, n_expert_);
    if (!host_.open(m_, std::min(prefillp::kRingPieces, prefillp::kRing * peer_homes), m_.sessions(), m_.tiers().vision, err))
        return false;
    uint64_t avail = available();
    auto ram_in = [&] {   // places per layer above the reserve, less the spare each layer keeps
        const uint64_t places = core::ram_tier_room(avail, ram_reserve, true) / one_each_ram;
        int slots = (int) std::min<uint64_t>(places > 0 ? places - 1 : 0, (uint64_t) n_expert_);
        if (m_.tiers().resident_budget > 0)
            slots = std::min(slots, (int) std::min<uint64_t>(m_.tiers().resident_budget / one_each_ram, (uint64_t) n_expert_));
        if (m_.diag_tier_ram() >= 0) slots = std::min(m_.diag_tier_ram(), n_expert_);
        return slots;
    };
    int ram_slots = ram_in();
    if (ram_slots < (m_.has_overlay() ? max_base : n_expert_ - peer_homes)) {
        if (!host_.set_ring(prefillp::kRingPieces, err)) return false;
        avail = available();
        ram_slots = ram_in();
    }
    // Rank replacements with ordinary experts. Their immutable RAM bank is already allocated;
    // count only the remaining base weights against the complement's RAM budget.
    uint64_t planned_ram = 0;
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
        planned_ram = 0;
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
            const int count = h.ram;
            if (count > 0) planned_ram += (uint64_t) (count + 1) * m_.pack().experts[l].blob_bytes;
        }
        homes_.ram = max_ram;
        homes_.file = max_file;
        if (!m_.has_overlay()) {
            homes_ = policy::plan_homes(n_expert_, main_homes, peer_slots, ram_slots);
            std::fill(layer_homes_.begin(), layer_homes_.end(), homes_);
            planned_ram = homes_.ram > 0 ? (uint64_t)(homes_.ram + 1) * one_each_ram : 0;
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
    const uint64_t ram_floor = ram_reserve + core::pin_tracking_bytes(planned_ram);
    if (homes_.peer == 0) peer_.reset();

    // the base arena (homes, or copies a long prompt borrows) leaves its reserve free; grow() fills it beside a short prompt
    reserve_ = reserve.beside_homes;
    modules_ = reserve.modules;
    per_layer_ = homes_.main > 0 ? homes_.main : copies;
    if (homes_.main == 0 && per_layer_ < g.n_used && !m_.has_overlay()) {
        err = "deepseek41: not enough device memory for the expert tier";
        return false;
    }
    const auto slot_bytes = slot_sizes((int)per_layer_);
    if (!slot_bytes.empty() && !cache_.open_sized(slot_bytes, n_layers_, n_expert_, err)) return false;
    res_.assign((size_t) n_layers_ * n_expert_, -1);
    stamp_.assign((size_t) n_layers_ * n_expert_, 0);
    owner_.assign(slot_bytes.size(), -1);
    busy_.assign(slot_bytes.size(), 0);
    int lo_prio = 0, hi_prio = 0;
    cudaDeviceGetStreamPriorityRange(&lo_prio, &hi_prio);
    if (cudaEventCreateWithFlags(&fetch_ev_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&gap_ev_, cudaEventDisableTiming) != cudaSuccess ||
        cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithPriority(&fetch_stream_, cudaStreamNonBlocking, hi_prio) != cudaSuccess) {
        err = "deepseek41: cannot create the promotion stream";
        return false;
    }
    for (size_t i = 0; i < kMaxPending; ++i) {
        cudaEvent_t e;
        if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
            err = "deepseek41: cannot create a promotion event";
            return false;
        }
        events_.push_back(e);
        free_events_.push_back((int) i);
    }

    // the CPU side: the file view, the per-layer formats, the pool
    if (homes_.file > 0) {
        std::optional<platform::MappedFile> mf = platform::MappedFile::open_ro(m_.path(), &err);
        if (!mf) return false;
        file_ = std::move(*mf);
    }
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
    if constexpr (platform::kDirectHostVmmExchange) {
        if (peer_ && !host_vmm_bytes_.empty() && !profile_peer_exchange_) {
            peer_submitter_ = std::make_unique<PeerSubmitter>(*peer_);
            basalt::log::info("basalt deepseek41", "helper expert exchange: dedicated submission thread");
        }
    }
    if (homes_.main == 0) seed_copies(per_layer_);
    promoted_ = 0;
    grow();
    // what the start picked by itself, on the server's auto line; the RAM floor the tier left free
    std::printf("auto flags: resident %.1f GB, pool-workers %d\n",
                (double) resident_bytes_ / 1e9, pool_->workers());   // the server keeps the words after the name
    std::fflush(stdout);
    basalt::log::info("basalt generate", "RAM floor %.1f GiB of %.1f GiB free", (double) ram_floor / 1073741824.0,
                      (double) avail / 1073741824.0);
    if (peer_) basalt::log::info("basalt deepseek41", "peer GPU %d: %lld experts", peer_->device(), (long long) n_layers_ * homes_.peer);
    basalt::log::info("basalt deepseek41", "expert homes of %d per layer: main card %d, second card %d, RAM %d, file %d; main card copies %lld",
                 n_expert_, homes_.main, homes_.peer, homes_.ram, homes_.file, (long long) (per_layer_ + ext_per_layer_ - homes_.main));
    program::warn_spill(spill(), peer_ ? "" : ", or --gpu 0,1");
    basalt::log::info("basalt deepseek41", "VRAM expert tier %.2f GiB, %lld of %d experts per layer (%.0f%%); CPU pool %d workers",
                 cache_.gib(), (long long) per_layer_, n_expert_, 100.0 * (double) per_layer_ / n_expert_, pool_->workers());
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

// The second card's real path: launches over the homes of the layer it holds most of, one token at a time.
class PeerPath final : public program::ProbePeerPath {
public:
    PeerPath(core::PeerExperts& peer, int top_k, const DsModel& model) : peer_(peer), k_(top_k), model_(model) {}
    bool setup(const program::ProbeEnv& env, const kernels::cpu::NativeFmt& f, std::string& why) override {
        L_ = kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
        for (int64_t e = 0; e < env.n_expert && (int) ids_.size() < 8; ++e)
            if (peer_.slot_ptr(0, e) != nullptr && !model_.replacement(0,(int)e)) ids_.push_back((int32_t) e);
        if (ids_.empty()) { why = "no_experts_on_peer"; return false; }
        if (!x_.alloc((size_t) f.n_embd * sizeof(float), cudaHostAllocPortable)) { why = "alloc_failed"; return false; }
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
            const auto t0 = program::ProbeClock::now();
            ok = peer_.launch_plain(0, L_, (const float*) x_.p, ids, k, kind, err) && peer_.finish(nullptr, err);
            rt.push_back(program::us_since(t0));
        }
        return ok;
    }

private:
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
    last_miss_.assign((size_t) n_layers_ * n_expert_, 0);
    preread_experts_.resize(n_layers_);
    const int64_t copies = homes_.main > 0 ? 0 : per_layer_;
    std::vector<int32_t> device_home;   // [layer * n_expert + e] the slot of a home on either card, or -1
    if (homes_.main > 0 || homes_.peer > 0) device_home.assign((size_t) n_layers_ * n_expert_, -1);
    if (peer_) {
        peer_owner_.assign((size_t) n_layers_ * homes_.peer, -1);
        const auto sizes = slot_sizes(homes_.peer, homes_.main > 0 ? homes_.main : (int)copies);
        if (!peer_->open_slots(sizes, err)) return false;
    }
    for (int l = 0; l < n_layers_; ++l) {
        std::vector<int> held;
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
                    const int32_t slot = empty_slot(false, expert_bytes(l,e));
                    if (slot < 0) { err="deepseek41: primary slot layout cannot hold its ranked experts"; return false; }
                    res_[idx] = slot;
                    owner_[(size_t) slot] = e;
                    if (!replacement) device_home[idx] = slot;
                    break;
                }
                case policy::Place::Peer: {
                    const int32_t slot = empty_slot(true, expert_bytes(l,e));
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
    if (cover_ <= 0 && device_home.empty()) return true;
    const bool swaps = homes_.file > 0 && cover_ > 0;   // the file's experts swap into RAM from a thread
    for (int l = 0; l < n_layers_; ++l)
        if (layer_homes_[l].ram > 0) spare_[(size_t) l] = layer_homes_[l].ram;
    const auto t0 = std::chrono::steady_clock::now();
    // what the start places from here on: the file's share never loads, a second card and the RAM tier have their bars
    uint64_t ram_bytes = 0, replaced_bytes = 0;
    for (int l = 0; l < n_layers_; ++l) {
        ram_bytes += (uint64_t) layer_homes_[l].ram * m_.pack().experts[l].blob_bytes;
        replaced_bytes += (uint64_t) (n_expert_ - base_count_[l]) * m_.pack().experts[l].blob_bytes;
    }
    resident_bytes_ = ram_bytes;
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
        const bool ok = res_[idx] >= 0 ? cudaMemcpy(cache_.device_slot(home), bytes, r.bytes, cudaMemcpyHostToDevice) == cudaSuccess
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
    // Base-file runs deliberately omit replaced IDs. Fill their GPU homes from the
    // original donor bytes, with the same slot ownership as every other expert.
    for (int l = 0; l < n_layers_; ++l) for (int e = 0; e < n_expert_; ++e) {
        const auto* v = m_.replacement(l,e);
        if (!v) continue;
        const int s = res_[(size_t)l*n_expert_+e];
        if (s >= 0) {
            if (cudaMemcpy(cache_.device_slot(s), v->blob, v->layout.bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                err = "deepseek41: loading a replacement primary expert"; return false;
            }
            main_load->add(v->layout.bytes);
        } else if (peer_ && peer_->slot_ptr(l,e)) {
            const int ps = peer_->residency()[(size_t)l*n_expert_+e];
            if (!peer_->fill_slot(ps, (const uint8_t*)v->blob, v->layout.bytes)) {
                err = "deepseek41: loading a replacement helper expert"; return false;
            }
            peer_load->add(v->layout.bytes);
        }
    }
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
        for (int64_t r = 0, placed = 0; r < n_expert_ && placed < n; ++r) {
            const int e = ranked(l, (int) r);
            if (!pinned(l, e) || res_[(size_t) l * n_expert_ + e] >= 0 || (peer_ && peer_->slot_ptr(l,e))) continue;
            ++tick_;
            if (claim(l, e, blob_of(l, e), copy_stream_) < 0) {
                cudaStreamSynchronize(copy_stream_);
                drain();
                ++tick_;
                if (claim(l, e, blob_of(l, e), copy_stream_) < 0) return;
            }
            ++placed;
        }
    }
    cudaStreamSynchronize(copy_stream_);
    drain();
}

const uint8_t* ExpertTier::blob_of(int layer, int e) const {
    if (const auto* v = m_.replacement(layer, e)) return static_cast<const uint8_t*>(v->blob);
    const ExpertLayer& el = m_.pack().experts[(size_t) layer];
    const int32_t r = ram_idx_[(size_t) layer * n_expert_ + e];
    if (r >= 0) return ram_place(layer, r);
    if (!file_.data()) return nullptr;
    return file_.data() + m_.pack().experts_range.offset + el.offset + (uint64_t) e * el.blob_bytes;
}

bool ExpertTier::pinned(int layer, int e) const {
    return m_.replacement(layer, e) || ram_idx_[(size_t) layer * n_expert_ + e] >= 0;
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
    return place < base ? ram_[layer] + (size_t) place * bytes : extra_ram_[layer][place - base].ptr;
}

bool ExpertTier::ensure_spare(int layer) {
    if (swapping_[layer]) return false;
    if (spare_[layer] >= 0) return true;
    if (!free_places_[layer].empty()) {
        spare_[layer] = free_places_[layer].back(); free_places_[layer].pop_back(); return true;
    }
    if (!m_.has_overlay()) return false;
    // Mixed-size GPU homes can displace a base expert without releasing a base RAM place.
    // Allocate only that missing place, and never cross the same RAM floor used at startup.
    const uint64_t bytes = m_.pack().experts[layer].blob_bytes;
    core::detail::HostMemory hm;
    if (!core::detail::host_available_memory(hm) || bytes > core::ram_tier_room(hm.available, platform::ram_reserve(), true))
        return false;
    ExtraRam r;
    if constexpr (platform::kDirectHostVmmExchange) r.ptr = core::vmm::alloc_host(bytes, r.mapped);
    if (!r.ptr) {
        r.ptr = core::alloc_ram(bytes);
        if (!r.ptr) return false;
        r.pin = core::pin_ram(r.ptr, bytes, bytes, {}, false);
        if (r.pin.registered_bytes < bytes) {
            core::unpin_ram(r.ptr, r.pin); platform::free_aligned(r.ptr); return false;
        }
    }
    auto& v = extra_ram_[layer];
    size_t i = 0;
    while (i < v.size() && v[i].ptr) ++i;
    if (i == v.size()) v.emplace_back();
    v[i] = std::move(r);
    spare_[layer] = (int) (ram_bytes_[layer] / bytes + i);
    return true;
}

void ExpertTier::return_place(int layer, int place) {
    if (place < 0) return;
    const int base = (int) (ram_bytes_[layer] / m_.pack().experts[layer].blob_bytes);
    if (place >= base) {
        auto& r = extra_ram_[layer][place - base];
        if (r.mapped) core::vmm::free_host(r.ptr, r.mapped);
        else { core::unpin_ram(r.ptr, r.pin); platform::free_aligned(r.ptr); }
        r = {};
    } else if (spare_[layer] < 0) spare_[layer] = place;
    else free_places_[layer].push_back(place);
}

void ExpertTier::land_home(int layer, int in, int out, int place) {
    const size_t base = (size_t) layer * n_expert_;
    if (place >= 0) ram_idx_[base + out] = place;
    if (!m_.replacement(layer, in)) {
        const int old = ram_idx_[base + in];
        ram_idx_[base + in] = -1;
        return_place(layer, old);
    }
}

const uint8_t* ExpertTier::slot_ptr(int32_t slot) const {
    const int64_t nb = (int64_t) n_layers_ * per_layer_;
    return slot < nb ? cache_.device_slot(slot) : ext_.device_slot((int32_t) (slot - nb));
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
    for (int l = 0; l < n_layers_; ++l) {
        std::vector<uint64_t> initial;
        if (conservative) {
            const auto& sorted = sorted_blob_bytes_[l];
            for (int i = 0; i < n; ++i) initial.push_back(sorted[sorted.size()-n+i]);
        } else {
            for (int r = rank_start; r < n_expert_ && (int)initial.size() < n; ++r) {
                const int e = ranked(l,r);
                if (seed && (!pinned(l,e) || res_[(size_t)l*n_expert_+e] >= 0 || (peer_ && peer_->slot_ptr(l,e)))) continue;
                initial.push_back(expert_bytes(l,e));
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
    int fit = 0;
    for (int n = 1; n <= maximum; ++n) {
        uint64_t bytes = 0;
        for (const auto b : slot_sizes(n,rank_start,seed,conservative)) bytes += ((uint64_t)b+255)/256*256;
        if (bytes <= budget) fit = n;
    }
    return fit;
}

void ExpertTier::drain() {
    for (const Pending& p : pending_) {
        cudaEventSynchronize(events_[(size_t) p.event]);
        const size_t in = (size_t) p.layer * n_expert_ + p.expert;
        res_[in] = p.slot;
        busy_[(size_t) p.slot] = 0;
        free_events_.push_back(p.event);
        if (p.home) land_home(p.layer, p.expert, p.out, p.place);
    }
    pending_.clear();
    if (peer_moves_.empty()) return;
    std::string err;
    if (peer_submitter_) peer_submitter_->wait();
    if (!peer_->read_wait(err)) throw std::runtime_error(err);
    for (const PeerMove& m : peer_moves_) {
        peer_->admit(m.layer, m.out, -1);
        peer_->admit(m.layer, m.in, m.slot);
        land_home(m.layer, m.in, m.out, m.place);
    }
    peer_moves_.clear();
}

void ExpertTier::settle() {
    flush_queue();
    cudaStreamSynchronize(copy_stream_);
    cudaStreamSynchronize(fetch_stream_);
    drain();
}

void ExpertTier::begin_token(int n_tokens) {
    token_ += (uint64_t) n_tokens;
    flush_queue();
    drain();
    finish_swaps();
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
    const uint64_t now = ++tick_;
    lk.k = k;
    for (int i = 0; i < k; ++i) {
        const int e = ids[i];
        if (e < 0 || e >= n_expert_) {
            err = "deepseek41: router returned a bad expert id";
            return false;
        }
        lk.experts[i] = e;
        lk.again[i] = false;
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
            lk.again[i] = policy::repeat_miss(last_miss_[idx], token_, policy::kPromoteWindow);
            if (pinned(layer, e)) {
                ++ram_hits_;
            } else {
                ++file_hits_;
                if (policy::repeat_miss(last_miss_[idx], token_, policy::kAdmitWindow) && want_.size() < 64) want_.emplace_back(layer, e);
            }
            last_miss_[idx] = token_ + 1;
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
    Use use[kMaxK]{};
    for (int i = 0; i < lk.k; ++i) { use[i].nt = 1; use[i].tok[0] = 0; use[i].row[0] = i; }
    start_cpu_window(layer, x, lk, use, rows);
}

void ExpertTier::start_cpu_window(int layer, const float* x, const Lookup& lk, const Use* use, float* rows) {
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

void ExpertTier::finish_cpu() { pool_->finish_split_multi_native(); }

int32_t ExpertTier::oldest_slot(int layer, bool homes_too, uint64_t& best, uint64_t bytes) const {
    int32_t slot = -1;
    best = ~0ull;
    uint64_t best_capacity = ~0ull;
    bool best_empty = false;
    auto scan = [&](int64_t lo, int64_t n) {
        for (int64_t s = lo; s < lo + n; ++s) {
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
        scan((int64_t) layer * per_layer_, per_layer_);
    if (ext_per_layer_) scan((int64_t) n_layers_ * per_layer_ + (int64_t) layer * ext_per_layer_, ext_per_layer_);
    return slot;
}

bool ExpertTier::move_to_peer(int layer, int e, const uint8_t* src) {
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
    if (slot < 0 || oldest >= on_main) return false;
    const int victim = peer_owner_[(size_t) slot];
    const bool needs_home = !pinned(layer, victim);
    if (needs_home && !ensure_spare(layer)) return false;
    const int32_t place = needs_home ? spare_[layer] : -1;
    if (needs_home) spare_[layer] = -1;
    const uint64_t out_bytes = needs_home ? expert_bytes(layer,victim) : 0;
    auto* out = needs_home ? ram_place(layer,place) : nullptr;
    if (peer_submitter_) {
        peer_submitter_->submit({slot, out, src, out_bytes, bytes});
    } else if (!host_vmm_bytes_.empty()) {
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
    } else peer_->exchange_slot(slot, out, out_bytes, src, bytes);
    peer_moves_.push_back({layer, e, peer_owner_[(size_t) slot], slot, place});
    peer_owner_[(size_t) slot] = e;
    ++promoted_;
    if (m_.replacement(layer,e) || m_.replacement(layer,victim)) ++overlay_swaps_;
    return true;
}

int32_t ExpertTier::claim(int layer, int e, const uint8_t* src, cudaStream_t stream, bool homes_too, bool whole, bool later) {
    const auto* replacement = m_.replacement(layer, e);
    if (homes_too && homes_.main > 0) ensure_spare(layer);
    if (pending_.size() + queue_.size() >= kMaxPending || free_events_.empty()) {
        ++dropped_;
        return -1;
    }
    uint64_t best = ~0ull;
    const int32_t slot = oldest_slot(layer, homes_too, best, expert_bytes(layer,e));
    if (slot < 0 || best == tick_) return -1;   // every slot is in use by this very lookup
    const int32_t o = owner_[(size_t) slot];
    if (o >= 0) res_[(size_t) layer * n_expert_ + o] = -1;
    owner_[(size_t) slot] = e;
    busy_[(size_t) slot] = 1;
    stamp_[(size_t) layer * n_expert_ + e] = tick_;
    const uint64_t bytes = replacement ? replacement->layout.bytes : m_.pack().experts[(size_t) layer].blob_bytes;
    uint8_t* dst = const_cast<uint8_t*>(slot_ptr(slot));
    const bool home = homes_.main > 0 && slot < (int64_t) n_layers_ * per_layer_;
    const bool needs_home = home && o >= 0 && !pinned(layer,o);
    const uint64_t out_bytes = needs_home ? expert_bytes(layer,o) : 0;
    int32_t place = -1;
    const uint64_t piece = whole ? bytes : policy::kPieceBytes;
    const auto pieces = [&](uint8_t* to, const uint8_t* from, uint64_t size, cudaMemcpyKind kind) {
        for (uint64_t off = 0; off < size; off += piece)
            cudaMemcpyAsync(to + off, from + off, std::min(piece, size - off), kind, stream);
    };
    if (needs_home) {   // the home's bytes first: the stream keeps the order
        place = spare_[(size_t) layer];
        spare_[(size_t) layer] = -1;
    }
    const int ev = free_events_.back();
    free_events_.pop_back();
    ++promoted_;
    if (replacement) ++overlay_copies_;
    if (later) {
        Queued q{{layer, e, slot, ev, o, place, home}, dst, src, needs_home ? ram_place(layer,place) : nullptr, bytes};
        q.home_bytes = out_bytes;
        queue_.push_back(q);
        return slot;
    }
    if (needs_home) pieces(ram_place(layer,place), dst, out_bytes, cudaMemcpyDeviceToHost);
    pieces(dst, src, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(events_[(size_t) ev], stream);
    pending_.push_back({layer, e, slot, ev, o, place, home});
    return slot;
}

void ExpertTier::mark_gap(cudaStream_t main) {
    gap_marked_ = !queue_.empty() && cudaEventRecord(gap_ev_, main) == cudaSuccess;
}

void ExpertTier::pump(int pieces) {
    if (gap_marked_) cudaStreamWaitEvent(copy_stream_, gap_ev_, 0);
    gap_marked_ = false;
    if (queue_.empty()) return;
    size_t done = 0;
    for (; done < queue_.size() && pieces > 0; ++done) {
        Queued& q = queue_[done];
        for (; q.home_to && q.home_off < q.home_bytes && pieces > 0; q.home_off += policy::kPieceBytes, --pieces)
            cudaMemcpyAsync(q.home_to + q.home_off, q.dst + q.home_off, std::min(policy::kPieceBytes, q.home_bytes - q.home_off),
                            cudaMemcpyDeviceToHost, copy_stream_);
        for (; (!q.home_to || q.home_off >= q.home_bytes) && q.off < q.bytes && pieces > 0; q.off += policy::kPieceBytes, --pieces)
            cudaMemcpyAsync(q.dst + q.off, q.src + q.off, std::min(policy::kPieceBytes, q.bytes - q.off), cudaMemcpyHostToDevice,
                            copy_stream_);
        if (q.off < q.bytes) break;
        cudaEventRecord(events_[(size_t) q.p.event], copy_stream_);
        pending_.push_back(q.p);
    }
    queue_.erase(queue_.begin(), queue_.begin() + (std::ptrdiff_t) done);
}

void ExpertTier::flush_queue() { pump(kAllPieces); }

int ExpertTier::fetch(int layer, Lookup& lk) {
    if (m_.tiers().diag_tier_frozen) return 0;
    int np = 0, nf = 0;
    for (int i = 0; i < lk.k; ++i)
        if (!lk.dev[i] && !lk.peer[i]) ++(pinned(layer, lk.experts[i]) ? np : nf);
    int left = policy::bus_count(np, nf, m_.tiers().gpu_single_miss);
    int started = 0;
    for (int i = lk.k - 1; i >= 0 && left > 0; --i) {
        if (lk.dev[i] || lk.peer[i] || !pinned(layer, lk.experts[i])) continue;   // RAM sources only
        --left;
        const int32_t slot = claim(layer, lk.experts[i], lk.host[i], fetch_stream_, false, /*whole=*/true);
        if (slot < 0) continue;
        lk.dev[i] = slot_ptr(slot);
        lk.host[i] = nullptr;
        lk.fetched[i] = true;
        --cpu_;
        ++fetched_;
        ++started;
    }
    if (started) cudaEventRecord(fetch_ev_, fetch_stream_);
    return started;
}

void ExpertTier::await_fetch(cudaStream_t main) { cudaStreamWaitEvent(main, fetch_ev_, 0); }

void ExpertTier::promote(int layer, const Lookup& lk) {
    if (m_.tiers().diag_tier_frozen) return;
    for (int i = 0; i < lk.k; ++i) {
        if (lk.dev[i] || lk.peer[i] || !lk.again[i] || !pinned(layer, lk.experts[i])) continue;   // RAM sources only
        if (!move_to_peer(layer, lk.experts[i], lk.host[i]))
            claim(layer, lk.experts[i], lk.host[i], copy_stream_, /*homes_too=*/true, /*whole=*/false, /*later=*/true);
    }
}

void ExpertTier::shrink() {
    if (ext_per_layer_ == 0) return;
    flush_queue();
    cudaStreamSynchronize(copy_stream_);
    cudaStreamSynchronize(fetch_stream_);
    drain();
    const int64_t nb = (int64_t) n_layers_ * per_layer_;
    for (int64_t s = nb; s < (int64_t) owner_.size(); ++s) {
        const int32_t o = owner_[(size_t) s];
        if (o >= 0) res_[(size_t) ((s - nb) / ext_per_layer_) * n_expert_ + o] = -1;
    }
    owner_.resize((size_t) nb);
    busy_.resize((size_t) nb);
    ext_.close();
    ext_per_layer_ = 0;
}

void ExpertTier::lend_all() {
    shrink();
    if (lent_per_layer_ > 0 || !cache_.valid() || homes_.main > 0) return;
    flush_queue();
    cudaStreamSynchronize(copy_stream_);
    cudaStreamSynchronize(fetch_stream_);
    drain();
    std::fill(res_.begin(), res_.end(), -1);
    owner_.clear();
    busy_.clear();
    lent_per_layer_ = per_layer_;
    per_layer_ = 0;
    cache_.close();
}

bool ExpertTier::reclaim(std::string& err) {
    if (lent_per_layer_ == 0) return true;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        err = "deepseek41: cannot query the free device memory";
        return false;
    }
    int64_t n = slots_fit(free_b, free_floor_, (int)lent_per_layer_);
    if (n < lent_per_layer_) n = slots_fit(free_b, reserve_ + free_floor_, (int)lent_per_layer_);
    if (n < 1) return true;   // nothing fits above the reserve: the slots stay lent
    const auto slot_bytes = slot_sizes((int)n);
    if (!cache_.open_sized(slot_bytes, n_layers_, n_expert_, err)) return false;
    per_layer_ = n;
    lent_per_layer_ = 0;
    owner_.assign(slot_bytes.size(), -1);
    busy_.assign(slot_bytes.size(), 0);
    return true;
}

void ExpertTier::grow() {
    if (ext_per_layer_ > 0 || per_layer_ >= n_expert_ || !cache_.valid()) return;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return;
    const int limit = (int)(cap_per_layer_ >= 0 ? std::min<int64_t>(n_expert_,cap_per_layer_) : n_expert_);
    const int64_t n = slots_fit(free_b, reserve_ + (token_ > 0 ? 0 : modules_) + free_floor_, limit-(int)per_layer_, 0, true);
    if (n < 1) return;
    const auto slot_bytes = slot_sizes((int)n,0,true);
    std::string err;
    if (!ext_.open_sized(slot_bytes, n_layers_, n_expert_, err)) {
        basalt::log::warn("basalt deepseek41", "VRAM expert tier cannot grow: %s", err.c_str());
        return;
    }
    ext_per_layer_ = n;
    owner_.resize((size_t) slot_total(), -1);
    busy_.resize((size_t) slot_total(), 0);
    basalt::log::info("basalt deepseek41", "VRAM expert tier %lld of %d experts per layer (%lld beside the prompt workspaces)",
                 (long long) (per_layer_ + n), n_expert_, (long long) per_layer_);
    cudaDeviceSynchronize();   // the arena's zeroing is not ordered against the copy stream
    const int64_t promoted = promoted_;
    seed_copies(n);
    promoted_ = promoted;
}

void ExpertTier::set_free_floor(int64_t mib) { free_floor_ = mib < 0 ? start_floor_ : (uint64_t) mib << 20; }

void ExpertTier::report(const char* tag) const {
    if (m_.has_overlay()) {
        basalt::log::info("basalt deepseek41", "overlay %s: primary=%lld peer=%lld cpu_rows=%lld admissions=%lld peer_swaps=%lld", tag,
                         (long long) overlay_hits_, (long long) overlay_peer_, (long long) overlay_cpu_,
                         (long long) overlay_copies_, (long long) overlay_swaps_);
        uint64_t primary = 0, helper = 0;
        const int64_t base = (int64_t)n_layers_*per_layer_;
        for (size_t s = 0; s < owner_.size(); ++s) if (owner_[s] >= 0) {
            const int layer = s < (size_t)base ? (int)(s/per_layer_) : (int)((s-base)/ext_per_layer_);
            primary += expert_bytes(layer,owner_[s]);
        }
        for (size_t s = 0; s < peer_owner_.size(); ++s) if (peer_owner_[s] >= 0)
            helper += expert_bytes((int)(s/homes_.peer),peer_owner_[s]);
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
