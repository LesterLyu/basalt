#pragma once
#include "basalt/deepseek41/execution_ops.hpp"
#include <functional>
#include <memory>

namespace basalt::deepseek41 {
// Quantized index queries/keys are exact BF16; key blocks bound the temporary QK matrix.
class PrefillIndex {
public:
    PrefillIndex(void* stream,const std::function<uint8_t*(size_t)>& allocate);
    ~PrefillIndex();
    void prepare_keys(PackedRows keys,int count);
    void run(const float* queries,const float* weights,PackedRows keys,const uint8_t* candidates,
              int candidate_stride,float* scores,int stride,int tokens,int key_count,const IndexRows& rows);
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
}
