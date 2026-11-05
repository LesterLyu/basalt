#pragma once
#include <span>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "basalt/kernels/sampler.hpp"

namespace basalt::deepseek41 {
inline constexpr int kPromptWindow=32768;
inline constexpr int kConcurrentPrefill=512;
inline constexpr uint64_t kExpertRamFloor=4ull<<30;

struct RunOptions {
    int context = 262144;
    int slots = 1;
    int draft_device = -1; // automatic: secondary card when available
    int helper_device = -1;
    int workers = 0;
    int reserve_mib = 200;
    int helper_reserve_mib = 200;
    bool dspark = true;
    bool resident_experts = true;
    // Development comparisons pin the placement and can cap the otherwise automatic tier sizes.
    bool freeze_placement = false;
    bool stream_prefill_experts = false; // diagnostic: stage otherwise CPU-served prompt experts on the primary
    bool tensor_prefill = false;
    int tensor_prefill_tokens = 512;
    uint64_t cache_limit_bytes = 0;
    bool elastic_kv = false;
    uint64_t engram_cache_rows = 1 << 20;
    int engram_inflight = 256;
    std::string draft_path;
};
enum class LogitsMode { none, last, all };
struct PrefillCancelled {};
struct ForwardResult {
    int rows = 0;
    std::vector<float> logits;
};
struct DraftResult {
    std::vector<int32_t> tokens;
    std::vector<float> confidence;
};
struct DecodeResult {
    std::vector<int32_t> tokens;
    int32_t next = -1;
    int proposed = 0, accepted = 0;
};
// Returns an allowed draw, or -1 if the constraint has no valid next token.
using TokenFilter = std::function<int32_t(int row, int32_t token)>;
struct DecodeRequest {
    int slot=0;
    int32_t anchor=-1;
    kernels::SamplerParams parameters;
    int max_tokens=6;
    int32_t stop_token=-1;
    std::span<const int32_t> proposals;
    TokenFilter filter{};
    std::span<const int32_t> penalty_tail{};
};
struct ExecutionStats { uint64_t primary_entries=0, cpu_entries=0, helper_entries=0, draft_rounds=0, helper_swaps=0, primary_swaps=0, streamed_entries=0, streamed_bytes=0, peer_streamed_bytes=0, expert_file_reads=0, peer_promotions=0, batch_rounds=0; };
struct MemoryUsage { int device; std::string component; uint64_t bytes; };

class Model;
class PrefixSnapshot {
    friend class Model;
    const Model* owner_ = nullptr;
    int position_ = 0;
    std::array<uint32_t,3> history_{};
    std::unique_ptr<uint8_t[]> data_;
    size_t bytes_ = 0;
public:
    size_t bytes() const { return bytes_; }
};

// One execution owner. A slot's forward window stays provisional until its prefix is committed.
class Model {
public:
    Model(const std::string& path, RunOptions options = {});
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    void reset();
    void resume_turn();
    bool prepare_kv(int end_position, std::string& error, bool during_prefill = false);
    uint64_t trim_kv(bool replenish=true);
    uint64_t release_idle_kv();
    void replenish_expert_cache();
    // Captures only committed state. Restoring is followed by prefill of a nonempty suffix.
    size_t prefix_snapshot_bytes() const;
    PrefixSnapshot save_prefix();
    void restore_prefix(const PrefixSnapshot& snapshot);
    void reset_metrics();
    int position() const;
    int context() const;
    int slots() const;
    int current_slot() const;
    // One scheduler owns the model. Slots isolate sequence state while sharing weights and expert tiers.
    void select_slot(int slot);
    // Known prompt tokens commit in bounded chunks; chunk also permits comparisons with smaller windows.
    bool accepts_images() const;
    ForwardResult prefill_image(std::span<const uint16_t> rows);
    ForwardResult prefill(const std::vector<int32_t>& tokens, int chunk = 0, LogitsMode mode = LogitsMode::last,
                          const std::function<void()>& between_steps = {},
                          const std::function<void()>& check_stop = {});
    // All-logits windows above 128 rows return host logits; GPU sampling remains available for smaller windows.
    ForwardResult forward(const std::vector<int32_t>& tokens, LogitsMode mode = LogitsMode::last);
    void commit(int accepted);
    DraftResult draft(int32_t anchor);
    // Verify proposals against the target's own greedy picks; only the accepted prefix changes persistent state.
    DecodeResult decode_greedy(int32_t anchor, int max_tokens = 6, int32_t stop_token = -1);
    std::vector<int32_t> sample(const kernels::SamplerParams& parameters, int rows = 1,
                                std::span<const int32_t> tail = {}, std::span<const int32_t> window = {});
    int32_t resample(const kernels::SamplerParams& parameters, int row, std::span<const float> bias);
    static DecodeResult verify_picks(std::span<const int32_t> input, std::span<const int32_t> picks,
                                     int32_t stop_token, const TokenFilter& filter);
    // counter names the draw after anchor; the caller advances it by the number of committed result tokens.
    DecodeResult decode(int32_t anchor, const kernels::SamplerParams& parameters, int max_tokens = 6,
                        int32_t stop_token = -1, std::span<const int32_t> proposals = {}, const TokenFilter& filter = {},
                        std::span<const int32_t> penalty_tail = {});
    std::vector<DecodeResult> decode_batch(const std::vector<DecodeRequest>& requests);
    ExecutionStats stats() const;
    std::array<uint64_t,3> decode_routes(int slot) const; // RAM miss, primary hit, helper hit
    std::vector<MemoryUsage> memory() const;
    std::vector<MemoryUsage> host_memory(uint64_t checkpoint_bytes) const;
    void log_host_memory(const char* phase, uint64_t rid, uint64_t checkpoint_bytes) const;
    void watch_memory(std::function<void()> changed);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace basalt::deepseek41
