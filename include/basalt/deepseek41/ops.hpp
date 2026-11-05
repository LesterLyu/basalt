#pragma once
#include <cstdint>

namespace basalt::deepseek41 {

// Streams are [tokens,4,hidden]; mixes [tokens,24], pre/post [tokens,4], residual matrices [tokens,4,4].
void hc_project(const float* streams, const float* fn, float* mixes, int hidden, int tokens, float epsilon, void* stream);
void hc_project_grouped(const float* streams, const float* fn, float* mixes, int hidden, int tokens, float epsilon, void* stream);
void hc_split(const float* mixes, const float* scale, const float* base, float* pre, float* post, float* residual,
              int tokens, int iterations, float epsilon, void* stream);
void hc_reduce(const float* streams, const float* pre, const float* norm_weight, float* out,
               int hidden, int tokens, float epsilon, void* stream);
// out may alias streams; all four source streams are read before any destination is written.
void hc_expand(const float* branch, const float* streams, const float* post, const float* residual, float* out,
               int hidden, int tokens, void* stream);

// Bias changes the picks; normalized unbiased scores determine their weights. Ties prefer the smaller id.
void route(const float* logits, const float* bias, int32_t* ids, float* weights,
           int experts, int top_k, int tokens, float scale, void* stream);

}  // namespace basalt::deepseek41
