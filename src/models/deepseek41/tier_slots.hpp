#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace basalt::models::deepseek41::policy {
// Uniform layer formats need one equal-sized slot per rank in every layer.
inline int uniform_slots_fit(uint64_t budget, uint64_t per_rank, int maximum) {
    if (!per_rank || maximum <= 0) return 0;
    return (int)std::min<uint64_t>(budget / per_rank, (uint64_t)maximum);
}

// Match the population's size classes while retaining enough room for the initial ranked set.
// The final quantile always keeps a place for the largest expert; promotion checks actual capacity.
inline std::vector<int64_t> mixed_slot_sizes(const std::vector<uint64_t>& sorted, std::vector<uint64_t> initial) {
    std::sort(initial.begin(), initial.end());
    const std::size_t n = initial.size();
    std::vector<int64_t> sizes(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t q = ((i + 1) * sorted.size() + n - 1) / n - 1;
        sizes[i] = (int64_t)std::max(initial[i], sorted[q]);
    }
    return sizes;
}
}
