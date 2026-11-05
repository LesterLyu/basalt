#include "prefill_index.hpp"
#include "basalt/prefill/gemm.hpp"
#include "basalt/prefill/kernels.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/deepseek41/weights.hpp"
#include <cuda_runtime.h>
#include <algorithm>

namespace basalt::deepseek41 {
struct PrefillIndex::Impl {
    prefill::Gemm gemm;void* stream;int device=0;
    static constexpr int key_tile=4096;
    uint16_t *q,*keys;
    float *key_float,*dots;
    static constexpr int cached_capacity=32768;
    uint16_t* cached=nullptr;
    const uint8_t* cached_source=nullptr;
    int cached_count=0;
    Impl(void* s,const std::function<uint8_t*(size_t)>& allocate):stream(s) {
        expect(cudaGetDevice(&device)==cudaSuccess,"index tensor device");
        q=reinterpret_cast<uint16_t*>(allocate(size_t(kAttentionWindow)*32*128*2));
        keys=reinterpret_cast<uint16_t*>(allocate(size_t(key_tile)*128*2));
        if(core::exp_flag("ds-index-cache-keys")!=0)
            cached=reinterpret_cast<uint16_t*>(allocate(size_t(cached_capacity)*128*2));
        key_float=reinterpret_cast<float*>(allocate(size_t(key_tile)*128*4));
        dots=reinterpret_cast<float*>(allocate(size_t(kAttentionWindow)*32*key_tile*4));
        auto* workspace=allocate(32ull<<20);std::string error;
        expect(gemm.init_external(s,nullptr,0,workspace,32ull<<20,error),error);
    }
};
PrefillIndex::PrefillIndex(void* stream,const std::function<uint8_t*(size_t)>& allocate)
    :impl_(std::make_unique<Impl>(stream,allocate)){}
PrefillIndex::~PrefillIndex(){const core::OnDevice on(impl_->device);impl_.reset();}
void PrefillIndex::prepare_keys(PackedRows keys,int count) {
    auto& m=*impl_;
    m.cached_count=0;m.cached_source=nullptr;
    if(!m.cached || core::exp_flag("ds-index-cache-keys")==0 || count<1 || count>m.cached_capacity)return;
    const core::OnDevice on(m.device);
    for(int first=0;first<count;first+=m.key_tile) {
        const int n=std::min(m.key_tile,count-first);
        unpack_rows(keys.data+size_t(first)*keys.stride,n,128,keys.stride,keys.format,m.key_float,m.stream);
        prefill::to_bf16(m.key_float,m.cached+size_t(first)*128,size_t(n)*128,m.stream);
    }
    m.cached_count=count;m.cached_source=keys.data;
}
void PrefillIndex::run(const float* q,const float* weights,PackedRows keys,const uint8_t* candidates,
                       int candidate_stride,float* scores,int stride,int tokens,int key_count,const IndexRows& rows) {
    auto& m=*impl_;const core::OnDevice on(m.device);
    expect(tokens>0 && tokens<=kAttentionWindow && key_count>0 && stride>=key_count,"invalid tensor index geometry");
    expect(keys.format==CacheFormat::index_fp4 && keys.stride>=68,"tensor index requires packed index keys");
    prefill::to_bf16(q,m.q,size_t(tokens)*32*128,m.stream);
    for(int first=0;first<key_count;first+=Impl::key_tile) {
        const int n=std::min(Impl::key_tile,key_count-first);
        const uint16_t* input=m.keys;
        if(m.cached_source==keys.data && key_count<=m.cached_count)input=m.cached+size_t(first)*128;
        else {
            unpack_rows(keys.data+size_t(first)*keys.stride,n,128,keys.stride,keys.format,m.key_float,m.stream);
            prefill::to_bf16(m.key_float,m.keys,size_t(n)*128,m.stream);
        }
        m.gemm.bf16(m.q,input,m.dots,tokens*32,n,128);
        reduce_index_tensor(m.dots,weights,candidates,candidate_stride,scores,stride,tokens,n,first,rows,m.stream);
    }
}
}
