// Native prompt products reuse the decode reductions over indexed rows.
#pragma once
#include "basalt/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <array>
#include <algorithm>

namespace basalt::models::deepseek41 {
constexpr int kNativePromptRows = 256;
constexpr int kNativePromptGroup = 8;
constexpr int kNativePromptGroups = kNativePromptRows / kNativePromptGroup;
struct NativePromptPlan {
    std::array<unsigned long long,kNativePromptGroups> ptr;
    std::array<int32_t,kNativePromptGroups+1> bounds;
    int32_t groups;
};

inline void native_prompt_rows(const kernels::NativeExpertLayout& L, float limit, const void* blob, const void* xq,
                               const int32_t* tok, const int32_t* dst, int rows, void* scratch, float* out, void* stream) {
    auto* meta = (uint8_t*)scratch + kernels::native_expert_scratch_bytes(kNativePromptRows, L.n_ff);
    NativePromptPlan plan{};
    plan.groups = (rows+kNativePromptGroup-1)/kNativePromptGroup;
    plan.ptr.fill((unsigned long long)(uintptr_t)blob);
    for (int i=0;i<=plan.groups;++i) plan.bounds[i]=std::min(i*kNativePromptGroup,rows);
    cudaMemcpyAsync(meta,&plan,sizeof plan,cudaMemcpyHostToDevice,(cudaStream_t)stream);
    kernels::native_grouped_set_mma_hint(false);
    kernels::native_expert_grouped_clamped(L, limit, (const unsigned long long*)meta,
        (const int32_t*)(meta + offsetof(NativePromptPlan,bounds)), (const int32_t*)(meta + offsetof(NativePromptPlan,groups)),
        dst, tok, plan.groups, rows, xq, scratch, out, stream);
}
} // namespace basalt::models::deepseek41
