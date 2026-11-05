#pragma once
#include "basalt/deepseek41/attention.hpp"
#include "basalt/prefill/gemm.hpp"
#include <stdexcept>

namespace basalt::deepseek41 {
struct AttentionMatrices {
    float *kv,*kt,*scores,*temporary;
    explicit AttentionMatrices(float* scratch) {
        kv=scratch;kt=kv+640*512;scores=kt+640*512;temporary=scores+64*640;
    }
};
// Reuse the bounded attention workspace; keys are padded for tensor-core products.
inline void attention_matrix(prefill::Gemm& gemm,const float* q,PackedRows window,PackedRows compressed,
                             const int32_t* selection,const AttentionRows& rows,const float* sinks,float scale,
                             float* out,float* scratch,void* stream) {
    const int count=rows.window_count[0]+rows.selected_count[0],stride=(count+31)/32*32;
    if(count<1 || count>640)throw std::invalid_argument("invalid attention matrix key count");
    const AttentionMatrices b(scratch);
    static_assert(2*640*512+64*640+64*512<=kAttentionScratch);
    prepare_attention_matrix(window,compressed,selection,rows,stride,b.kv,b.kt,stream);
    gemm.f32_emulated(q,b.kv,b.scores,64,stride,512);
    softmax_attention_matrix(b.scores,sinks,selection,rows,stride,scale,stream);
    gemm.f32_emulated(b.scores,b.kt,out,64,512,stride);
}
}
