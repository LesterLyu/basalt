// src/models/deepseek41/tier.hpp - where the routed experts of a DeepSeek-V4.1 pack live while a token runs.
//
// An expert's home is a slot of the main card, a slot of a second card, the pinned RAM tier or the pack file mapped
// read-only (policy::Homes); the routing order deals them out at the start. The main card's reclaimable cache and its
// permanent homes share a per-layer LRU. Both GPUs own resident weights exclusively; an evicted expert goes to RAM
// before its GPU slot is overwritten. The GPUs compute the experts they
// hold; the CPU pool computes the rest from RAM or the mapped file while the GPUs work, and the pinned experts it
// computed are copied into the main card's LRU on a side stream so the next token finds them there.
#pragma once

#include "basalt/core/expert_cache.hpp"
#include "basalt/core/memory_usage.hpp"
#include "basalt/core/ram_load.hpp"
#include "basalt/program/expert_spill.hpp"
#include "basalt/program/probe_rates.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/prefill/peer_share.hpp"
#include "basalt/platform/mapped_file.hpp"
#include "models/deepseek41/prompt_host.hpp"
#include "models/deepseek41/expert_ram.hpp"
#include "models/deepseek41/tier_policy.hpp"

#include <cuda_runtime.h>
#include <map>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace basalt::models::deepseek41 {

class DsModel;

class ExpertTier {
public:
    static constexpr int kMaxK = 48;      ///< distinct experts one lookup holds (a window of up to 8 rows, 6 each)
    static constexpr int kMaxWindow = 8;

    /// The tier lookup of one layer's routed experts.
    struct Lookup {
        int k = 0;
        const uint8_t* dev[kMaxK] = {};    ///< device blob of an expert the VRAM tier holds, else null
        const uint8_t* peer[kMaxK] = {};   ///< device blob of an expert the second card holds or fetched, else null
        const uint8_t* host[kMaxK] = {};   ///< host blob of the others (and of those the second card fetched)
        int experts[kMaxK] = {};
        bool again[kMaxK] = {};            ///< a miss that repeats one from the last kPromoteWindow tokens
        bool fetched[kMaxK] = {};          ///< `dev` or `peer` is being copied in by `fetch`
    };

    /// The window rows that use one looked-up expert: row i of token tok[i] is `row[i]` of the output rows.
    struct Use {
        int nt = 0;
        int tok[kMaxWindow] = {};
        int row[kMaxWindow] = {};
    };

    explicit ExpertTier(const DsModel& model);
    ~ExpertTier();
    ExpertTier(const ExpertTier&) = delete;
    ExpertTier& operator=(const ExpertTier&) = delete;

    /// Main card bytes each arrangement leaves free.
    struct Reserve {
        uint64_t copies = 0;       ///< beside slots that are all copies, which a long prompt borrows
        uint64_t homes = 0;        ///< beside homes, which stay through a prompt at the full context
        uint64_t beside_homes = 0; ///< beside homes and the copies next to them, where a short prompt's workspaces fit
        uint64_t modules = 0;      ///< ... until a first token has loaded the kernel modules
    };
    /// Sizes the tiers and loads them. `h_rows`: the mapped rows a second card shares with the CPU pool.
    bool open(const Reserve& reserve, float* h_rows, std::string& err);

    /// Gives the extra arena's physical segments back; weights return to RAM before release.
    /// Prompt workspaces use lend() for a partial loan, and grow() refills the free segments.
    void shrink();
    void grow();
    /// Make this many bytes free by returning only the needed cache segments to RAM.
    bool lend(uint64_t free_bytes, std::string& err);
    bool malloc_slab_backed() const { return cache_.malloc_slab_backed(); }
    uint64_t loanable_bytes() const {
        return (ext_.segmented() ? ext_.mapped_bytes() : 0) + (homes_.main == 0 ? cache_.mapped_bytes() : 0);
    }
    void prompt_workspace(const void* owner, uint64_t bytes, uint64_t margin);
    /// A long prompt streams every expert to the main card and needs the card for its chunks: the slots that are
    /// reclaimable segments go back whole, and reclaim() maps them again at the same addresses. Homes stay.
    void lend_all();
    bool reclaim(std::string& err);
    /// Temporarily use the helper's existing slot storage; GPU-only weights return to RAM first.
    bool borrow_peer_workspace(uint64_t bytes, void*& storage, uint64_t& capacity, std::string& err);
    bool return_peer_workspace(std::string& err);
    /// Main card bytes the tier, the caches and the prompt path leave free: --vram-reserve-mib, or `mib` once set
    /// (`mib` < 0: back to the start's). The arena is not resized here.
    void set_free_floor(int64_t mib);
    uint64_t free_floor() const { return free_floor_; }

    /// Called before each token or window. The copies queued during the previous one (promotions, second card exchanges)
    /// land layer by layer: each before its layer's next lookup (`pump`), so what a lookup finds depends only on the
    /// routing history, as if every copy had landed at the window's start.
    void begin_token(int n_tokens = 1);
    /// Looks up `ids[0..k)` of `layer` and stamps them as used; `layer`'s queued copies land first.
    bool lookup(int layer, const int32_t* ids, int k, Lookup& lk, std::string& err);
    /// The CPU's share: the experts of `lk` without a device blob, over the activation `x` (n_embd floats, host).
    /// Row i of `rows` (n_embd floats each) gets expert i's output. Started here, it runs on the pool's workers while
    /// the caller launches the GPUs' shares; finish_cpu joins it. `lk` and `rows` stay as they are until then.
    void start_cpu(int layer, const float* x, const Lookup& lk, float* rows);
    /// The same over a window: `x` holds a row per token and each expert i runs over the rows `use[i]` names, into those
    /// rows of `rows`.
    void start_cpu_window(int layer, const float* x, const Lookup& lk, const Use* use, float* rows);
    void finish_cpu();
    /// The promotions' copies, and the second card's exchanges, wait for the bus to idle: mark_gap notes the point of
    /// `main` (a layer's sum) they start behind, and pump issues up to `pieces` promotion pieces there (and at most
    /// policy::kPeerGapPieces exchange pieces on the second card's bus), after the next launches went out (the next
    /// layer's head; at the window's end, its output), so they do not take the host memory's bandwidth from the CPU's
    /// share and the bus copies of a layer's misses. A copy queued after them on another stream (a layer's bus fetch)
    /// waits for one piece at most. With `layer` >= 0 (called before the layer's lookup) that layer's queued copies first
    /// go out whole and land.
    void mark_gap(cudaStream_t main);
    void pump(int layer, int pieces);
    static constexpr int kAllPieces = 1 << 20;
    /// Plans the GPU share of RAM misses in transient slots, leaving the rest for the CPU. submit_fetch() starts
    /// their copies after the other shares launch; await_fetch() gates their GPU consumers. Repeat misses may be cached.
    int fetch(int layer, Lookup& lk);
    /// Submit the planned misses after the CPU, helper and resident GPU experts have started.
    void submit_fetch();
    void await_fetch(cudaStream_t main);
    /// Queues the pinned misses the CPU computed that missed before within kPromoteWindow tokens for a copy into the
    /// VRAM tier (side stream, off the token path, in pieces).
    void promote(int layer, const Lookup& lk);

    /// The copies under way land: every expert is in its place. A prompt reads the places, so it comes first.
    void settle();
    /// Where a resident copy of (layer, e) lives: a main card slot, a second card slot or the pinned RAM tier; all null
    /// when only the file has it. Read-only: no stamp, no promotion.
    struct Source {
        const uint8_t* dev = nullptr;
        const uint8_t* peer = nullptr;
        const uint8_t* pinned = nullptr;
    };
    Source locate(int layer, int e) const;
    core::PeerExperts* peer() const { return peer_.get(); }
    /// The second card's share of a prompt chunk's experts; null without a second card.
    prefill::PeerShare* peer_share() const { return share_.get(); }
    /// What the split of a layer's experts between the cards prices them at.
    const prefill::ShareRates& share_rates(int layer) const { return rates_[(size_t) layer]; }
    bool launch_peer(int layer, const kernels::NativeExpertLayout& layout, const float* x, const int32_t* ids,
                     int tokens, int k, const int32_t* kind, std::string& err,
                     const kernels::NativeExpertLayout* const* layouts);
    bool finish_peer(std::string& err);
    /// The prompt path's and the images' host buffers, allocated before the RAM tier is sized.
    PromptHost& host() { return host_; }
    kernels::cpu::ExpertPool* pool() const { return pool_.get(); }
    const kernels::cpu::NativeFmt& fmt(int layer) const { return fmt_[(size_t) layer]; }
    const kernels::cpu::NativeFmt& fmt(int layer, int expert) const;

    /// The VRAM slots open now, every layer's.
    int64_t slots() const { return cache_.slots()+ext_.slots(); }

    void memory_usage(core::MemoryUsage& usage) const;
    void report(const char* tag) const;
    /// The routed experts so far: hits of the main card's slots, those plus the CPU's, a bus copy's or the second card's.
    int64_t hits() const { return hits_; }
    int64_t lookups() const { return hits_ + cpu_; }
    int64_t offloaded() const { return fetched_ + peer_hits_; }
    /// Of the CPU's, those read from the file.
    int64_t file_lookups() const { return file_hits_; }
    /// The experts only the file holds.
    program::ExpertSpill spill() const;
    /// What `PROBE rates` measures: this tier's pool, RAM blobs, formats and second card.
    void probe_env(program::ProbeEnv& env);

private:
    /// A slot's copy on its way. `out` >= 0: the home it replaces is on its way to RAM place `place`.
    struct Pending {
        int layer, expert, slot, event;
        int out = -1, place = -1;
        bool home = false;
    };
    /// A promotion whose copies wait for an idle bus: the home's bytes out to `home_to` (on the out stream), then `src`
    /// into `dst` once they are out.
    struct Queued {
        Pending p;
        uint8_t* dst;
        const uint8_t* src;
        uint8_t* home_to;
        uint64_t bytes, off = 0, home_off = 0, home_bytes = 0;
        bool device_source = false;
    };
    struct Fetch { uint8_t* dst; const uint8_t* src; uint64_t bytes; };
    /// `in` on its way into the second card's `slot`, its pieces waiting for an idle bus; `out`, which held it, on its
    /// way to RAM place `place`: the slot's bytes out to `to` first, then `from` in.
    struct PeerMove {
        int layer, in, out, slot, place;
        uint8_t* to;
        const uint8_t* from;
        uint64_t out_bytes, in_bytes, out_off = 0, in_off = 0;
        int64_t in_mark = -1;   ///< the second card mark behind its last piece
    };
    /// Issues what is left of `layer`'s queued copies (>= 0), then up to `pieces` pieces of the others, oldest first:
    /// the main card's and the second card's.
    void issue(int layer, int pieces);
    void issue_peer(int layer, int pieces);
    /// The copies queued for `layer` (or every layer) land: the experts take their places.
    void land(int layer);
    void land_all();
    void apply(const Pending& p);
    void apply(const PeerMove& m);
    /// A cold pinned expert (victim) replaced by a hot file-backed one (expert), read into the layer's spare slot.
    struct Swap {
        int layer, expert, victim, place;
    };
    bool load_ram(std::string& err);
    bool open_share(std::string& err);
    int ranked(int layer, int r) const;
    bool load_rank(std::string& err);
    void seed_copies(int64_t n);
    void preread();
    bool pinned(int layer, int e) const;
    uint64_t expert_bytes(int layer, int e) const;
    uint8_t* ram_place(int layer, int place) const;
    bool ensure_spare(int layer);
    void return_place(int layer, int place);
    void pack_extra_ram();
    uint64_t resident_ram_bytes() const;
    bool restore_experts(bool helper, int64_t first, std::string& err, std::vector<int>* backed = nullptr);
    void release_peer_backing();
    bool shrink_cache(core::ExpertCache& cache, uint64_t offset, uint64_t keep, std::string& err);
    uint64_t refill_reserve() const;
    int main_slot(int layer, int rank) const;
    int slot_layer(int slot) const;
    std::vector<int64_t> cache_sizes(int n, bool extra) const;
    void land_home(int layer, int in, int out, int place);
    const uint8_t* blob_of(int layer, int e) const;
    const uint8_t* slot_ptr(int32_t slot) const;
    uint64_t slot_capacity(int32_t slot) const;
    std::vector<int64_t> slot_sizes(int n, int rank_start = 0, bool seed = false, bool conservative = false) const;
    int slots_fit(uint64_t free, uint64_t keep, int maximum, int rank_start = 0, bool seed = false, bool conservative = false) const;
    uint64_t slot_total() const { return (uint64_t) n_layers_ * (uint64_t) (per_layer_ + ext_per_layer_); }
    void finish_swaps();
    void start_swaps();
    void swap_worker();
    /// Reserves the least recently used fitting slot, or -1, for a background copy. Homes require a RAM place
    /// for their displaced expert. Device sources remain live until the fetch ring's completion event.
    int32_t claim(int layer, int e, const uint8_t* src, bool homes_too, bool device_source = false);
    /// The least recently used free slot of the layer's main card share, or -1; `best` gets its last use.
    int32_t oldest_slot(int layer, bool homes_too, uint64_t& best, uint64_t bytes = 0) const;
    /// Moves into the helper only when its victim is policy::kPeerMoveAge older than the main card's.
    bool move_to_peer(int layer, int e, const uint8_t* src);

    const DsModel& m_;
    core::ExpertCache cache_;
    int n_layers_ = 0, n_expert_ = 0;
    int64_t per_layer_ = 0;                 ///< VRAM slots each layer owns in the base arena
    core::ExpertCache ext_;                 ///< slots beside the base arena, open only while no prompt workspace is
    int64_t ext_per_layer_ = 0;             ///< slots each layer owns in `ext_`
    uint64_t ram_file_keep_ = 0;
    bool unread_said_ = false;
    std::map<const void*, std::pair<uint64_t,uint64_t>> prompt_workspaces_;
    bool exclusive_cache_ = false;         ///< primary cache entries own their weights without RAM copies
    uint64_t returned_experts_ = 0, returned_bytes_ = 0;
    std::vector<int32_t> res_;              ///< [layer * n_expert + e] VRAM slot or -1
    std::vector<uint64_t> stamp_;           ///< [layer * n_expert + e] last use
    std::vector<int32_t> owner_;            ///< [slot] expert it holds or is receiving, or -1
    std::vector<uint8_t> busy_;             ///< [slot] a promotion is copying into it
    std::vector<Pending> pending_;          ///< issued copies not landed yet
    std::vector<Queued> queue_;             ///< promotions not yet fully issued, oldest first
    std::vector<Fetch> fetch_queue_;
    std::vector<PeerMove> peer_moves_;      ///< second card exchanges not landed yet, oldest first
    std::vector<int32_t> peer_owner_;       ///< [second card slot] expert it holds or is receiving
    std::vector<cudaEvent_t> events_, return_events_;
    std::vector<int> free_events_;
    int claims_ = 0;                        ///< copies claimed since the last begin_token (at most kMaxPending)
    cudaStream_t copy_stream_ = nullptr;    ///< promotions, off the token path
    cudaStream_t return_stream_ = nullptr;
    cudaStream_t fetch_stream_ = nullptr;   ///< copies a layer waits for (high priority)
    uint8_t* fetch_scratch_ = nullptr;
    uint64_t fetch_stride_ = 0;
    int fetch_slots_ = 0;
    static constexpr int kFetchRings = 3;
    int fetch_ring_ = 0;
    bool fetch_scratch_used_ = false, fetch_promotions_ = false;
    bool fetch_scratch_fenced_[kFetchRings]{};
    cudaEvent_t fetch_scratch_done_[kFetchRings]{};
    cudaEvent_t fetch_ev_ = nullptr;
    cudaEvent_t gap_ev_ = nullptr;          ///< the main stream's point the pumped pieces start behind
    bool gap_marked_ = false;               ///< gap_ev_ was recorded for the next pump
    uint64_t tick_ = 0;
    uint64_t reserve_ = 0, modules_ = 0;
    std::vector<std::vector<uint64_t>> sorted_blob_bytes_;
    int64_t overlay_hits_ = 0, overlay_copies_ = 0, overlay_cpu_ = 0, overlay_peer_ = 0, overlay_swaps_ = 0;
    std::vector<std::vector<std::unique_ptr<ExpertRam>>> extra_ram_;
    std::vector<std::vector<int>> free_places_;
    uint64_t start_floor_ = 0, free_floor_ = 0;
    int64_t cap_per_layer_ = -1;            ///< --expert-cache or the diagnostic's slots per layer, -1 when neither

    std::vector<uint64_t> host_vmm_bytes_;  ///< nonempty only for primary-only pinned system allocations
    bool profile_peer_exchange_ = false;
    uint64_t peer_exchanges_ = 0, peer_exchange_ns_ = 0, peer_read_ns_ = 0, peer_host_ns_ = 0, peer_write_ns_ = 0;
    std::vector<core::RamPin> ram_pins_;    ///< [layer] what each span's pin got
    std::vector<uint8_t*> ram_;             ///< [layer] compact pinned experts and a spare when the layer has RAM homes
    std::vector<uint64_t> ram_bytes_;       ///< actual allocated span per layer, including its spare
    policy::Homes homes_;
    std::vector<policy::Homes> layer_homes_;
    std::vector<int> base_count_;          ///< effective base experts; replacement weights have separate RAM backing
    std::vector<int32_t> rank_;             ///< [layer * n_expert + rank] expert; empty: the file's order
    std::vector<std::vector<int>> initial_seeds_; ///< ranked host candidates while initial placement is planned
    std::vector<std::vector<int>> preread_experts_; ///< immutable initial file-backed ranking for the reader thread
    int cover_ = 0;                         ///< experts per layer in the RAM tier
    std::vector<int32_t> ram_idx_;          ///< [layer * n_expert + e] pinned slot or -1
    std::vector<int32_t> spare_;            ///< [layer] pinned slot free for a swap, or -1
    std::vector<uint8_t> swapping_;         ///< [layer] a file expert is being read into its spare
    std::vector<uint64_t> spare_since_;     ///< [layer] token at which the spare became free
    std::vector<uint64_t> last_miss_;       ///< [layer * n_expert + e] token + 1 of the last miss
    std::vector<std::pair<int, int>> want_; ///< file-backed experts that missed twice, in lookup order
    std::vector<Swap> flight_;              ///< swaps started at the previous begin_token
    std::thread worker_;
    std::thread preread_;
    std::atomic<bool> preread_quit_{false};
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<Swap> jobs_;
    size_t jobs_done_ = 0;
    bool quit_ = false;
    uint64_t token_ = 0;
    int64_t swapped_ = 0;
    platform::MappedFile file_;             ///< the experts section
    PromptHost host_;
    struct PeerSubmitter;
    struct PeerCompute;
    std::unique_ptr<core::PeerExperts> peer_;
    uint64_t peer_workspace_bytes_ = 0;
    static constexpr size_t kSpareBlocks = 16;   // free RAM places kept pinned for promotions
    uint64_t peer_workspace_pool_limit_ = 0;
    std::vector<std::pair<int32_t, int32_t>> peer_workspace_lent_;
    std::vector<int> peer_workspace_backed_;
    std::unique_ptr<prefill::PeerShare> share_;
    std::vector<prefill::ShareRates> rates_;    ///< [layer]
    std::unique_ptr<PeerSubmitter> peer_submitter_;
    std::unique_ptr<PeerCompute> peer_compute_;
    std::vector<kernels::cpu::NativeFmt> fmt_;   ///< [layer]
    std::unique_ptr<kernels::cpu::ExpertPool> pool_;
    uint8_t* act_ = nullptr;                ///< the quantized activation the CPU jobs share
    kernels::cpu::ExpertJobMulti cpu_jobs_[kMaxK];   ///< the CPU share started and not yet joined

    int64_t hits_ = 0, cpu_ = 0, ram_hits_ = 0, promoted_ = 0, fetched_ = 0, dropped_ = 0, peer_hits_ = 0, file_hits_ = 0;
};

}  // namespace basalt::models::deepseek41
