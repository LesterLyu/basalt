// Sizes the RAM tier after both GPUs have exclusive ownership of their initial experts.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace basalt::models::deepseek41::policy {

inline uint64_t ram_aligned(uint64_t bytes, uint64_t grain) {
    return grain ? (bytes + grain - 1) / grain * grain : bytes;
}

struct RamLayer {
    int experts = 0;
    uint64_t blob_bytes = 0;
};

struct RamPlan {
    std::vector<int> counts;
    uint64_t bytes = 0;
    bool fits = false;
};

// Mandatory CPU replacements first, then ranked base experts, including one swap spare
// per occupied layer. GPU-owned weights are absent from the input and consume no RAM.
inline RamPlan plan_ram(const std::vector<RamLayer>& layers, uint64_t budget,
                        uint64_t mandatory, uint64_t grain) {
    RamPlan plan;
    plan.counts.resize(layers.size(), 0);
    if (mandatory > budget) return plan;
    plan.fits = true;
    plan.bytes = mandatory;
    int ranks = 0;
    for (const auto& l : layers) ranks = std::max(ranks, l.experts);
    for (int rank = 0; rank < ranks; ++rank) {
        for (size_t l = 0; l < layers.size(); ++l) {
            if (rank >= layers[l].experts || plan.counts[l] != rank) continue;
            const auto blob = layers[l].blob_bytes;
            const auto before = rank ? ram_aligned((uint64_t)(rank + 1) * blob, grain) : 0;
            const auto after = ram_aligned((uint64_t)(rank + 2) * blob, grain);
            if (after - before > budget - plan.bytes) continue;
            ++plan.counts[l];
            plan.bytes += after - before;
        }
    }
    return plan;
}

}  // namespace basalt::models::deepseek41::policy
