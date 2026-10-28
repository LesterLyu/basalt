#include "basalt/core/expert_source.hpp"

#include <algorithm>
#include <cstdio>
#include <iterator>
#include <vector>

namespace {
class UnusedSource final : public basalt::core::ExpertSource {
public:
    const uint8_t* blob(int64_t, int64_t) override { ++reads; return nullptr; }
    int reads = 0;
};
}

int main() {
    using namespace basalt::core;
    UnusedSource source;
    ExpertDispatch dispatch;
    dispatch.src = &source;
    dispatch.n_expert = 16;
    int32_t ids[] = {2, 4, 5, 4, 2, 5};
    int32_t kinds[] = {1, 0, 1, 0, 1, 1};
    GpuPlanSink sink;
    sink.ext_kind = kinds;
    sink.cap = 6;
    dispatch.plan = &sink;
    std::vector<float> input(2 * 2560, 0.0f), output(6 * 2560, 7.0f);
    auto run = [&] { expert_pool_dispatch_multi(dispatch, input.data(), ids, 2, 3, output.data()); };
    run();
    if (dispatch.failed || dispatch.pcie_experts != 2 || dispatch.offload_entries != 4 || dispatch.cache_hits != 2)
        return std::puts("device-planned staging must count distinct experts and all offloaded entries"), 1;
    std::fill(std::begin(kinds), std::end(kinds), 0);
    run();
    if (dispatch.failed || dispatch.pcie_experts != 2 || dispatch.offload_entries != 4 || dispatch.cache_hits != 8)
        return std::puts("VRAM-only routing must not add staged experts"), 1;
    std::fill(std::begin(kinds), std::end(kinds), 1);
    run();
    if (dispatch.failed || dispatch.pcie_experts != 5 || dispatch.offload_entries != 10 || dispatch.layers != 3)
        return std::puts("staged expert counts must reset between layers"), 1;
    if (source.reads != 0 || dispatch.multi_misses != 0 ||
        !std::all_of(output.begin(), output.end(), [](float value) { return value == 7.0f; }) ||
        !std::all_of(dispatch.job_of.begin(), dispatch.job_of.end(), [](int16_t value) { return value == -1; }))
        return std::puts("staging statistics must preserve GPU rows, scratch state and source access"), 1;
    std::puts("expert_stage_stats: distinct groups, entries, layer reset and GPU rows pass");
    return 0;
}
