#include "basalt/deepseek41/execution_ops.hpp"
#include "basalt/core/diag.hpp"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <stdexcept>
#include <string>
#include <climits>

namespace basalt::deepseek41 {
namespace {
__device__ float warp_sum(float x) {
    for (int delta = 16; delta; delta >>= 1) x += __shfl_down_sync(0xffffffff, x, delta);
    return x;
}
template<bool bf16>
__device__ void linear_stage(const void* matrix, const float* x, float* y, int inputs, int outputs,
                             size_t input_stride, size_t output_stride, int block, int token) {
    const int row = block * 4 + threadIdx.x / 32, lane = threadIdx.x & 31;
    float value = 0;
    if (row < outputs) for (int c = lane; c < inputs; c += 32) {
        const size_t index = size_t(row) * inputs + c;
        const float weight = bf16 ? __uint_as_float(uint32_t(static_cast<const uint16_t*>(matrix)[index]) << 16) :
                                    static_cast<const float*>(matrix)[index];
        value += weight * x[size_t(token) * input_stride + c];
    }
    value = warp_sum(value);
    if (!lane && row < outputs) y[size_t(token) * output_stride + row] = value;
}
template<bool bf16>
__global__ void linear_kernel(const void* m, const float* x, float* y, int in, int out, size_t xs, size_t ys) {
    linear_stage<bf16>(m, x, y, in, out, xs, ys, blockIdx.x, blockIdx.y);
}
// Keep the reference's FP32 lane accumulations; reuse a weight across prompt rows.
template<int rows,int experts=1>
__global__ void router_linear_tiled_kernel(const float* matrix,const float* input,float* out,int tokens) {
    constexpr int in=5120,outputs=384;
    const int expert=(blockIdx.x*4+threadIdx.x/32)*experts, lane=threadIdx.x%32;
    const int first=blockIdx.y*rows;
    const float* w=matrix+size_t(expert)*in;
    float value[rows][experts]{};
    const bool full=first+rows<=tokens;
    for(int c=lane;c<in;c+=32) {
        float weight[experts];
        #pragma unroll
        for(int e=0;e<experts;++e)weight[e]=w[size_t(e)*in+c];
        #pragma unroll
        for(int t=0;t<rows;++t)if(full || first+t<tokens) {
            const float x=input[size_t(first+t)*in+c];
            #pragma unroll
            for(int e=0;e<experts;++e)value[t][e]+=weight[e]*x;
        }
    }
    #pragma unroll
    for(int t=0;t<rows;++t) {
        #pragma unroll
        for(int e=0;e<experts;++e) {
            const float dot=warp_sum(value[t][e]);
            if(!lane && (full || first+t<tokens))out[size_t(first+t)*outputs+expert+e]=dot;
        }
    }
}
constexpr int markov_vocab=129280,markov_inputs=256,markov_rows=32;
__global__ void cached_markov_kernel(const uint16_t* matrix,const float* input,const int32_t* previous,
                                     const int32_t* keys,float* cache,float* logits) {
    const int id=*previous,slot=static_cast<unsigned>(id)%kMarkovCacheSlots;
    float* cached=cache+size_t(slot)*markov_vocab;
    const int first=blockIdx.x*markov_rows;
    if(keys[slot]==id) {
        if(threadIdx.x<markov_rows) {
            const int row=first+threadIdx.x;
            logits[row]+=cached[row];
        }
        return;
    }
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    float x[8];
    #pragma unroll
    for(int k=0;k<8;++k)x[k]=input[lane+32*k];
    for(int j=warp;j<markov_rows;j+=4) {
        const int row=first+j;
        const uint16_t* w=matrix+size_t(row)*markov_inputs;
        float sum=0.f;
        #pragma unroll
        for(int k=0;k<8;++k)sum+=__uint_as_float(uint32_t(w[lane+32*k])<<16)*x[k];
        sum=warp_sum(sum);
        if(!lane) {
            cached[row]=sum;
            logits[row]+=sum;
        }
    }
}
__global__ void publish_markov_key(const int32_t* previous,int32_t* keys) {
    const int id=*previous;
    keys[static_cast<unsigned>(id)%kMarkovCacheSlots]=id;
}
__device__ void embedding_stage(const float* x, float* h, float* pre, int hidden, int token, int c) {
    if (c < hidden) for (int s = 0; s < 4; ++s) h[(size_t(token)*4+s)*hidden+c] = x[size_t(token)*hidden+c];
    if (c < 4) pre[token*4+c] = c == 0 ? 1.f : 0.f;
}
__global__ void embedding_kernel(const float* x, float* h, float* p, int dim) {
    embedding_stage(x,h,p,dim,blockIdx.y,blockIdx.x*blockDim.x+threadIdx.x);
}
__device__ void mean_stage(const float* h, float* out, int dim, size_t stride, size_t offset, int token, int c) {
    if (c < dim) {
        float sum = 0; for (int s = 0; s < 4; ++s) sum += h[(size_t(token)*4+s)*dim+c];
        out[size_t(token)*stride+offset+c] = sum * .25f;
    }
}
__global__ void mean_kernel(const float* h, float* out, int dim, size_t stride, size_t offset) {
    mean_stage(h,out,dim,stride,offset,blockIdx.y,blockIdx.x*blockDim.x+threadIdx.x);
}
template<bool masked=false>
__device__ void combine_stage(const float* gpu, const float* cpu, const float* weights, const float* shared,
                              float* out, int dim, int k, int token, int c,uint64_t host_rows=~uint64_t{0}) {
    if (c >= dim) return;
    float value = shared ? shared[size_t(token)*dim+c] : 0.f;
    for (int e = 0; e < k; ++e) {
        const size_t index = (size_t(token)*k+e)*dim+c;
        value += weights[token*k+e] * ((gpu ? gpu[index] : 0.f) + (cpu && (!masked || ((host_rows>>(token*k+e))&1u)) ? cpu[index] : 0.f));
    }
    out[size_t(token)*dim+c] = value;
}
__global__ void combine_kernel(const float* g, const float* c, const float* w, const float* s, float* out, int dim, int k) {
    combine_stage(g,c,w,s,out,dim,k,blockIdx.y,blockIdx.x*blockDim.x+threadIdx.x);
}
__global__ void combine_masked_kernel(const float* g,const float* host,const float* w,const float* shared,
                                      float* out,int dim,int k,uint64_t host_rows) {
    combine_stage<true>(g,host,w,shared,out,dim,k,blockIdx.y,blockIdx.x*blockDim.x+threadIdx.x,host_rows);
}
__device__ void add_stage(float* x, const float* y, size_t n, size_t i) { if (i<n) x[i] += y[i]; }
__global__ void add_kernel(float* x,const float* y,size_t n) { add_stage(x,y,n,size_t(blockIdx.x)*blockDim.x+threadIdx.x); }
struct FiniteMaximum { float value;int32_t index,invalid; };
constexpr int pick_tile=1024;
__device__ FiniteMaximum pick_max(FiniteMaximum a,FiniteMaximum b) {
    a.invalid|=b.invalid;
    if(b.index>=0 && (a.index<0 || b.value>a.value || (b.value==a.value && b.index<a.index))) {
        a.value=b.value;a.index=b.index;
    }
    return a;
}
__device__ FiniteMaximum pick_warp(FiniteMaximum a) {
    for(int delta=16;delta;delta>>=1) {
        FiniteMaximum b{__shfl_down_sync(0xffffffff,a.value,delta),
                        __shfl_down_sync(0xffffffff,a.index,delta),__shfl_down_sync(0xffffffff,a.invalid,delta)};
        a=pick_max(a,b);
    }
    return a;
}
__device__ FiniteMaximum pick_block(FiniteMaximum value,FiniteMaximum* scratch) {
    value=pick_warp(value);
    if(!(threadIdx.x&31))scratch[threadIdx.x/32]=value;
    __syncthreads();
    if(threadIdx.x<32) {
        value=threadIdx.x<8?scratch[threadIdx.x]:FiniteMaximum{-CUDART_INF_F,-1,0};
        value=pick_warp(value);
    }
    return value;
}
__device__ void finite_maximum_stage(const float* x,int count,FiniteMaximum* out,int block,FiniteMaximum* scratch) {
    FiniteMaximum value{-CUDART_INF_F,-1,0};
    for(int i=block*pick_tile+threadIdx.x;i<min(count,(block+1)*pick_tile);i+=256) {
        const float next=x[i];
        if(isfinite(next))value=pick_max(value,{next,i,0});else value.invalid=1;
    }
    value=pick_block(value,scratch);
    if(!threadIdx.x)out[block]=value;
}
__global__ void finite_maximum_kernel(const float* x,int count,FiniteMaximum* out) {
    __shared__ FiniteMaximum scratch[8];finite_maximum_stage(x,count,out,blockIdx.x,scratch);
}
__device__ void finite_pick_stage(const FiniteMaximum* x,int count,int32_t* token,int32_t* invalid,FiniteMaximum* scratch) {
    FiniteMaximum value{-CUDART_INF_F,-1,0};
    for(int i=threadIdx.x;i<count;i+=256)value=pick_max(value,x[i]);
    value=pick_block(value,scratch);
    if(!threadIdx.x) {*token=value.index;if(value.invalid)atomicOr(invalid,1);}
}
__global__ void finite_pick_kernel(const FiniteMaximum* x,int count,int32_t* token,int32_t* invalid) {
    __shared__ FiniteMaximum scratch[8];finite_pick_stage(x,count,token,invalid,scratch);
}
__device__ void bf16_gather_stage(const uint16_t* table,const int32_t* index,int rows,int width,float* out,int col) {
    if(col>=width)return;
    const int row=*index;
    out[col]=row>=0 && row<rows?__uint_as_float(uint32_t(table[size_t(row)*width+col])<<16):0.f;
}
__global__ void bf16_gather_kernel(const uint16_t* table,const int32_t* index,int rows,int width,float* out) {
    bf16_gather_stage(table,index,rows,width,out,blockIdx.x*blockDim.x+threadIdx.x);
}
__device__ void primary_plan_stage(const unsigned long long* lookup,const int32_t* ids,int experts,int top,int tokens,
                                   unsigned long long* ptr,int32_t* starts,int32_t* groups,int32_t* tok,int32_t* dst) {
    __shared__ int expert[36],representative[36];
    __shared__ unsigned long long address[36];
    const int j=threadIdx.x,entries=top*tokens;
    if(j<entries) {
        const int e=ids[j];expert[j]=e;
        address[j]=e>=0 && e<experts?lookup[e]:0;
    }
    __syncthreads();
    if(j<entries) {
        int first=j;
        for(int k=0;k<j;++k)if(expert[k]==expert[j]) {first=k;break;}
        representative[j]=address[j] && first==j;
    }
    __syncthreads();
    if(j<entries && address[j]) {
        int rank=0,row=0,offset=0;
        for(int k=0;k<entries;++k)if(address[k]) {
            if(expert[k]<expert[j]) {row+=1;rank+=representative[k];}
            else if(k<j && expert[k]==expert[j])offset+=1;
        }
        if(representative[j]) {ptr[rank]=address[j];starts[rank]=row;}
        tok[row+offset]=j/top;dst[row+offset]=j;
    }
    if(j==0) {
        int ng=0,rows=0;
        for(int k=0;k<entries;++k) {ng+=representative[k];rows+=address[k]!=0;}
        starts[ng]=rows;*groups=ng;
    }
}
__global__ void primary_plan_kernel(const unsigned long long* lookup,const int32_t* ids,int experts,int top,int tokens,
                                     unsigned long long* ptr,int32_t* starts,int32_t* groups,int32_t* tok,int32_t* dst) {
    primary_plan_stage(lookup,ids,experts,top,tokens,ptr,starts,groups,tok,dst);
}
__device__ void seed_f64_stage(const float* x,double* out,size_t count,size_t i) {if(i<count)out[i]=double(x[i]);}
__global__ void seed_f64_kernel(const float* x,double* out,size_t n) {seed_f64_stage(x,out,n,size_t(blockIdx.x)*blockDim.x+threadIdx.x);}
__device__ void accumulate_f64_stage(const float* parts,const int32_t* entries,const float* weights,double* out,
                                     int hidden,int top,size_t i) {
    const int row=int(i/hidden),col=int(i%hidden),entry=entries[row];
    atomicAdd(out+size_t(entry/top)*hidden+col,double(parts[i])*double(weights[entry]));
}
__global__ void accumulate_f64_kernel(const float* p,const int32_t* ids,const float* weights,double* out,int rows,int h,int top) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<size_t(rows)*h)accumulate_f64_stage(p,ids,weights,out,h,top,i);
}
__device__ void finish_f64_stage(const double* x,float* out,size_t count,size_t i) {if(i<count)out[i]=float(x[i]);}
__global__ void finish_f64_kernel(const double* x,float* out,size_t n) {finish_f64_stage(x,out,n,size_t(blockIdx.x)*blockDim.x+threadIdx.x);}
__device__ bool score_better(float a,int ai,float b,int bi) {return a!=b?a>b:ai<bi;}
template<int cap,bool blocks,bool merge=false>
__device__ void top_rows_stage(const float* input,int stride,IndexRows visible,void* result,int result_stride,int row) {
    __shared__ float score[2*cap];__shared__ int ids[2*cap];
    const int tid=threadIdx.x,nkeys=visible.visible[row],count=blocks?(nkeys+7)/8:nkeys;
    const float* x=input+size_t(row)*stride;
    if constexpr(blocks)for(int i=tid;i<count;i+=blockDim.x)static_cast<uint8_t*>(result)[size_t(row)*result_stride+i]=0;
    for(int i=tid;i<cap;i+=blockDim.x) {score[i]=-CUDART_INF_F;ids[i]=INT_MAX;}
    __syncthreads();
    for(int start=0;start<count;start+=cap) {
        for(int i=tid;i<cap;i+=blockDim.x) {
            const int id=start+i;float value=-CUDART_INF_F;
            if(id<count) {
                if constexpr(blocks) {
                    for(int j=id*8;j<min(nkeys,id*8+8);++j)value=fmaxf(value,x[j]);
                    if(id==count-1)value=CUDART_INF_F;
                } else value=x[id];
            }
            score[cap+i]=value;ids[cap+i]=id<count?id:INT_MAX;
        }
        __syncthreads();
        for(int k=2;k<=2*cap;k*=2)for(int j=k/2;j;j/=2) {
            // The previous winners are already sorted. Sort only the new half
            // before the cross-half compare, then finish only the winning half.
            const int begin=merge && k<=cap?cap/2:0;
            const int end=merge && k>cap && j<cap?cap/2:cap;
            for(int p=begin+tid;p<end;p+=blockDim.x) {
                const int i=((p&~(j-1))<<1)|(p&(j-1)),b=i+j;
                const bool reverse=(i&k)!=0;
                if(reverse?score_better(score[i],ids[i],score[b],ids[b]):score_better(score[b],ids[b],score[i],ids[i])) {
                    const float v=score[i];score[i]=score[b];score[b]=v;
                    const int id=ids[i];ids[i]=ids[b];ids[b]=id;
                }
            }
            __syncthreads();
        }
    }
    if constexpr(blocks) {
        for(int i=tid;i<cap;i+=blockDim.x)if(ids[i]!=INT_MAX && score[i]!=-CUDART_INF_F)
            static_cast<uint8_t*>(result)[size_t(row)*result_stride+ids[i]]=1;
    } else {
        for(int i=tid;i<cap;i+=blockDim.x)if(score[i]==-CUDART_INF_F)ids[i]=INT_MAX;
        __syncthreads();
        for(int k=2;k<=cap;k*=2)for(int j=k/2;j;j/=2) {
            for(int p=tid;p<cap/2;p+=blockDim.x) {
                const int i=((p&~(j-1))<<1)|(p&(j-1)),b=i+j;
                if((i&k)?ids[i]<ids[b]:ids[i]>ids[b]) {const int id=ids[i];ids[i]=ids[b];ids[b]=id;}
            }
            __syncthreads();
        }
        for(int i=tid;i<cap;i+=blockDim.x)static_cast<int32_t*>(result)[size_t(row)*result_stride+i]=ids[i]==INT_MAX?-1:ids[i];
    }
}
template<int cap,bool blocks,bool merge=false>
__global__ void top_rows_kernel(const float* x,int stride,IndexRows rows,void* out,int out_stride) {
    top_rows_stage<cap,blocks,merge>(x,stride,rows,out,out_stride,blockIdx.x);
}
__device__ uint32_t ordered_score(float x) {
    uint32_t bits=__float_as_uint(x);
    if((bits<<1)==0)bits=0; // Signed zero is a tie in the original comparator.
    return bits^(bits>>31?0xffffffffu:0x80000000u);
}
__global__ void radix_top_rows_kernel(const float* input,int stride,IndexRows visible,int32_t* output,int out_stride) {
    constexpr int cap=512,threads=256;
    const int row=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid/32,count=visible.visible[row];
    const float* x=input+size_t(row)*stride;
    int32_t* out=output+size_t(row)*out_stride;
    using Scan=cub::BlockScan<uint32_t,threads>;
    __shared__ typename Scan::TempStorage scan;
    __shared__ uint32_t histogram[threads/32][256],prefix;
    __shared__ int rank,taken,eq_seen;
    if(tid==0) {prefix=0;rank=cap;taken=eq_seen=0;}
    __syncthreads();
    bool all=count<=cap;
    if(all && __syncthreads_or((tid<count && isnan(x[tid])) ||
                               (tid+threads<count && isnan(x[tid+threads])))) {
        top_rows_stage<cap,false>(input,stride,visible,output,out_stride,row);
        return;
    }
    uint32_t mask=0;
    for(int shift=24;shift>=0 && !all;shift-=8) {
        for(int w=0;w<threads/32;++w)histogram[w][tid]=0;
        __syncthreads();
        bool nan=false;
        const uint32_t prior=prefix;
        for(int first=0;first<count;first+=threads) {
            const int i=first+tid;
            const float value=i<count?x[i]:-CUDART_INF_F;
            const uint32_t bits=ordered_score(value),bin=(bits>>shift)&255;
            if(shift==24)nan|=isnan(value);
            const unsigned active=__ballot_sync(0xffffffffu,value!=-CUDART_INF_F && (bits&mask)==prior);
            if(active&(1u<<lane)) {
                const unsigned same=__match_any_sync(active,bin);
                if(lane==__ffs(same)-1)atomicAdd(&histogram[warp][bin],__popc(same));
            }
        }
        if(__syncthreads_or(nan)) {
            // Preserve the existing selection behavior for unordered scores.
            top_rows_stage<cap,false>(input,stride,visible,output,out_stride,row);
            return;
        }
        uint32_t local=0,greater=0,total=0;
        for(int w=0;w<threads/32;++w)local+=histogram[w][255-tid];
        Scan(scan).ExclusiveSum(local,greater,total);
        const int desired=rank;
        __syncthreads();
        all=shift==24 && total<cap;
        if(!all && greater<uint32_t(desired) && greater+local>=uint32_t(desired)) {
            prefix=prior|(uint32_t(255-tid)<<shift);
            rank=desired-int(greater);
        }
        __syncthreads();
        mask|=255u<<shift;
    }
    const uint32_t threshold=prefix;
    const int ties=rank;
    // A block scan tracks counts above the threshold and ties independently.
    // Take the earliest tied positions and emit all winners in source order.
    for(int first=0;first<count;first+=threads) {
        const int i=first+tid;
        const float value=i<count?x[i]:-CUDART_INF_F;
        const uint32_t bits=ordered_score(value);
        const bool valid=value!=-CUDART_INF_F;
        const bool greater=valid && (all || bits>threshold);
        const bool equal=valid && !all && bits==threshold;
        uint32_t prior=0,total=0;
        Scan(scan).ExclusiveSum(uint32_t(greater)|(uint32_t(equal)<<16),prior,total);
        const int base=taken,seen=eq_seen,quota=max(0,ties-seen);
        const int eq_prior=prior>>16;
        if(greater || (equal && eq_prior<quota))
            out[base+int(prior&65535)+min(eq_prior,quota)]=i;
        __syncthreads();
        if(tid==0) {taken=base+int(total&65535)+min(int(total>>16),quota);eq_seen=seen+int(total>>16);}
        __syncthreads();
        if(taken==cap)break;
    }
    for(int i=taken+tid;i<cap;i+=threads)out[i]=-1;
}
__device__ float index_value(PackedRows keys, int row, int c, int width) {
    const uint8_t* p = keys.data+size_t(row)*keys.stride;
    const unsigned code = (p[c/2]>>(4*(c&1)))&15;
    const float values[8] = {0,.5f,1,1.5f,2,3,4,6};
    return ((code&8)?-values[code&7]:values[code&7])*__uint_as_float(uint32_t(p[width/2+c/32])<<23);
}
__device__ void index_stage(const float* q,const float* weights,PackedRows keys,const uint8_t* candidates,
                            int candidate_stride,int block_size,float* scores,int score_stride,int heads,int width,
                            IndexRows rows,int token,int key,float weight_scale) {
    __shared__ float key_values[512], partial[8];
    const int lane=threadIdx.x&31,warp=threadIdx.x/32;
    if (key>=rows.visible[token] || (candidates && !candidates[token*candidate_stride+key/block_size])) {
        if (!threadIdx.x) scores[size_t(token)*score_stride+key]=-CUDART_INF_F;
        return;
    }
    for (int c=threadIdx.x;c<width;c+=blockDim.x) key_values[c]=index_value(keys,key,c,width);
    __syncthreads();
    float sum=0;
    for (int h=warp;h<heads;h+=8) {
        float dot=0;
        for (int c=lane;c<width;c+=32) dot+=q[(size_t(token)*heads+h)*width+c]*key_values[c];
        dot=warp_sum(dot);
        if (!lane) sum+=fmaxf(0.f,dot)*(weights[token*heads+h]*weight_scale);
    }
    if (!lane) partial[warp]=sum;
    __syncthreads();
    if (!threadIdx.x) { float value=0;for(int i=0;i<8;++i)value+=partial[i];scores[size_t(token)*score_stride+key]=value; }
}
__global__ void index_kernel(const float* q,const float* w,PackedRows keys,const uint8_t* mask,int ms,int bs,
                             float* out,int stride,int heads,int width,IndexRows rows,float scale) {
    index_stage(q,w,keys,mask,ms,bs,out,stride,heads,width,rows,blockIdx.y,blockIdx.x,scale);
}
__device__ void reduce_index_stage(const float* dots,const float* weights,const uint8_t* mask,int ms,float* scores,
                                    int stride,int keys,int first,IndexRows rows,int t,int key) {
    if(key>=keys)return;
    const int position=first+key;
    if(position>=rows.visible[t] || (mask && !mask[size_t(t)*ms+position/8])) {
        scores[size_t(t)*stride+position]=-CUDART_INF_F;return;
    }
    float value=0;
    for(int w=0;w<8;++w) {
        float part=0;
        for(int h=w;h<32;h+=8)part+=fmaxf(0.f,dots[(size_t(t)*32+h)*keys+key])*(weights[t*32+h]*(1.f/64));
        value+=part;
    }
    scores[size_t(t)*stride+position]=value;
}
__global__ void reduce_index_kernel(const float* dots,const float* weights,const uint8_t* mask,int ms,float* scores,
                                     int stride,int keys,int first,IndexRows rows) {
    reduce_index_stage(dots,weights,mask,ms,scores,stride,keys,first,rows,blockIdx.y,blockIdx.x*blockDim.x+threadIdx.x);
}
__device__ void iota_stage(int32_t* ids,int n,int i) { if(i<n)ids[i]=i; }
__global__ void iota_kernel(int32_t* ids,int n) { iota_stage(ids,n,blockIdx.x*blockDim.x+threadIdx.x); }
// Keeping every candidate needs only reachable positions in source order.
__device__ void all_selected_stage(const float* scores,int32_t* out,int count,int tid,
                                  cub::BlockScan<int,256>::TempStorage& scratch,int& valid_count) {
    int valid[8],offset[8];
    for(int j=0;j<8;++j) {const int i=tid*8+j;valid[j]=i<count && scores[i]!=-CUDART_INF_F;}
    int total=0;cub::BlockScan<int,256>(scratch).ExclusiveSum(valid,offset,total);
    if(tid==0)valid_count=total;
    __syncthreads();
    for(int j=0;j<8;++j)if(valid[j])out[offset[j]]=tid*8+j;
    for(int i=tid;i<count;i+=256)if(i>=valid_count)out[i]=-1;
}
__global__ void all_selected_kernel(const float* scores,int32_t* out,int count) {
    __shared__ cub::BlockScan<int,256>::TempStorage scratch;
    __shared__ int valid_count;
    all_selected_stage(scores,out,count,threadIdx.x,scratch,valid_count);
}
__device__ void sorted_stage(const float* scores,const int32_t* ids,int32_t* out,int selected,int padded) {
    __shared__ int32_t positions[2048];
    for (int i=threadIdx.x;i<padded;i+=blockDim.x)
        positions[i]=i<selected && scores[i]!=-CUDART_INF_F?ids[i]:0x7fffffff;
    __syncthreads();
    for (int k=2;k<=padded;k*=2)for(int j=k/2;j;j/=2) {
        for (int i=threadIdx.x;i<padded;i+=blockDim.x) {
            const int other=i^j;
            if (other>i) { const int a=positions[i],b=positions[other]; if ((i&k)?a<b:a>b) {positions[i]=b;positions[other]=a;} }
        }
        __syncthreads();
    }
    for (int i=threadIdx.x;i<selected;i+=blockDim.x)out[i]=positions[i]==0x7fffffff?-1:positions[i];
}
__global__ void sorted_kernel(const float* s,const int32_t* ids,int32_t* out,int n,int padded) { sorted_stage(s,ids,out,n,padded); }
__device__ void block_score_stage(const float* score,int n,int size,float* blocks,int b) {
    const int count=(n+size-1)/size;
    if (b>=count)return;
    float maximum=-CUDART_INF_F;
    for(int i=b*size;i<n && i<(b+1)*size;++i)maximum=fmaxf(maximum,score[i]);
    blocks[b]=b==count-1?CUDART_INF_F:maximum;
}
__global__ void block_score_kernel(const float* s,int n,int b,float* out) { block_score_stage(s,n,b,out,blockIdx.x*blockDim.x+threadIdx.x); }
__device__ void mask_stage(const int32_t* ids,int n,uint8_t* mask,int blocks,int i) {if(i<n && ids[i]>=0 && ids[i]<blocks)mask[ids[i]]=1;}
__global__ void mask_kernel(const int32_t* ids,int n,uint8_t* mask,int blocks) {mask_stage(ids,n,mask,blocks,blockIdx.x*blockDim.x+threadIdx.x);}
void cuda_check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(std::string("deepseek41 selection: ")+cudaGetErrorString(e));}
}
void cached_markov_add(const uint16_t* matrix,const float* input,const int32_t* previous,
                       int32_t* keys,float* cache,float* logits,void* stream) {
    static_assert(markov_vocab%markov_rows==0);
    const auto s=static_cast<cudaStream_t>(stream);
    cached_markov_kernel<<<markov_vocab/markov_rows,128,0,s>>>(matrix,input,previous,keys,cache,logits);
    publish_markov_key<<<1,1,0,s>>>(previous,keys);
}
void router_linear_tiled(const float* weights,const float* input,float* output,int tokens,void* stream,int rows,int expert_tile) {
    if(tokens<1 || tokens>32768)throw std::invalid_argument("deepseek41: invalid tiled router window");
    const auto s=static_cast<cudaStream_t>(stream);
    if(expert_tile==4)router_linear_tiled_kernel<8,4><<<dim3(24,(tokens+7)/8),128,0,s>>>(weights,input,output,tokens);
    else if(expert_tile==2)router_linear_tiled_kernel<8,2><<<dim3(48,(tokens+7)/8),128,0,s>>>(weights,input,output,tokens);
    else if(rows==8)router_linear_tiled_kernel<8><<<dim3(96,(tokens+7)/8),128,0,s>>>(weights,input,output,tokens);
    else router_linear_tiled_kernel<4><<<dim3(96,(tokens+3)/4),128,0,s>>>(weights,input,output,tokens);
}
void dense_linear(int type,const void* w,const float* x,float* y,int inputs,int outputs,int tokens,size_t xs,size_t ys,void* stream) {
    if(inputs<1 || outputs<1 || tokens<1 || xs<size_t(inputs) || ys<size_t(outputs) || (type!=0 && type!=30))
        throw std::invalid_argument("deepseek41 dense: invalid geometry or format");
    const dim3 grid((outputs+3)/4,tokens);const auto s=static_cast<cudaStream_t>(stream);
    if(type==0)linear_kernel<false><<<grid,128,0,s>>>(w,x,y,inputs,outputs,xs,ys);
    else linear_kernel<true><<<grid,128,0,s>>>(w,x,y,inputs,outputs,xs,ys);
}
void expand_embedding(const float* x,float* h,float* pre,int hidden,int tokens,void* stream) {
    if(hidden<4 || tokens<1)throw std::invalid_argument("deepseek41 embedding: invalid geometry");
    embedding_kernel<<<dim3((hidden+255)/256,tokens),256,0,static_cast<cudaStream_t>(stream)>>>(x,h,pre,hidden);
}
void stream_mean(const float* h,float* out,int hidden,int tokens,size_t stride,size_t offset,void* stream) {
    if(hidden<1 || tokens<1 || stride<offset+size_t(hidden))throw std::invalid_argument("deepseek41 mean: invalid geometry");
    mean_kernel<<<dim3((hidden+255)/256,tokens),256,0,static_cast<cudaStream_t>(stream)>>>(h,out,hidden,stride,offset);
}
void combine_experts(const float* g,const float* c,const float* w,const float* sh,float* out,int hidden,int k,int tokens,void* stream) {
    if(hidden<1 || k<1 || k>6 || tokens<1)throw std::invalid_argument("deepseek41 expert combine: invalid geometry");
    combine_kernel<<<dim3((hidden+255)/256,tokens),256,0,static_cast<cudaStream_t>(stream)>>>(g,c,w,sh,out,hidden,k);
}
void combine_experts_masked(const float* gpu,const float* host,const float* weights,const float* shared,
                            float* output,int hidden,int top,int tokens,uint64_t host_rows,void* stream) {
    if(hidden<1 || top<1 || top>6 || tokens<1 || tokens>6 || !host)
        throw std::invalid_argument("deepseek41 masked expert combine: invalid geometry");
    combine_masked_kernel<<<dim3((hidden+255)/256,tokens),256,0,static_cast<cudaStream_t>(stream)>>>(
        gpu,host,weights,shared,output,hidden,top,host_rows);
}
void add_rows(float* x,const float* y,size_t count,void* stream) {
    if(count)add_kernel<<<(count+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(x,y,count);
}
void plan_primary_experts(const unsigned long long* lookup,const int32_t* ids,int experts,int top,
                           unsigned long long* ptr,int32_t* starts,int32_t* groups,int32_t* tok,int32_t* dst,void* stream,int tokens) {
    if(experts<1 || top<1 || top>6 || tokens<1 || tokens>6)throw std::invalid_argument("invalid device expert plan geometry");
    primary_plan_kernel<<<1,64,0,static_cast<cudaStream_t>(stream)>>>(lookup,ids,experts,top,tokens,ptr,starts,groups,tok,dst);
}
void seed_experts_f64(const float* shared,double* out,size_t count,void* stream) {
    if(count)seed_f64_kernel<<<(count+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(shared,out,count);
}
void accumulate_experts_f64(const float* p,const int32_t* ids,const float* weights,double* out,int rows,int h,int top,void* stream) {
    if(rows<0 || h<1 || top<1 || top>6)throw std::invalid_argument("deepseek41 expert accumulator: invalid geometry");
    if(rows)accumulate_f64_kernel<<<(size_t(rows)*h+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(p,ids,weights,out,rows,h,top);
}
void finish_experts_f64(const double* accumulator,float* out,size_t count,void* stream) {
    if(count)finish_f64_kernel<<<(count+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(accumulator,out,count);
}
void index_scores(const float* q,const float* w,PackedRows keys,const uint8_t* candidates,int ms,int bs,float* scores,
                   int stride,int heads,int width,int tokens,int count,const IndexRows& rows,void* stream,float scale) {
    if(tokens<1 || tokens>kAttentionWindow || count<1 || stride<count || heads<1 || width<32 || width>512 || width%32 ||
        keys.format!=CacheFormat::index_fp4 || keys.stride<cache_row_bytes(width,keys.format) ||
        (candidates && (bs<1 || ms<(count+bs-1)/bs)))throw std::invalid_argument("deepseek41 index: invalid geometry");
    for(int t=0;t<tokens;++t)if(rows.visible[t]<0 || rows.visible[t]>count)throw std::invalid_argument("deepseek41 index: invalid visible length");
    index_kernel<<<dim3(count,tokens),256,0,static_cast<cudaStream_t>(stream)>>>(q,w,keys,candidates,ms,bs,scores,stride,heads,width,rows,scale);
}
void gather_bf16_row(const uint16_t* table,const int32_t* index,int rows,int width,float* out,void* stream) {
    if(!table || !index || !out || rows<1 || width<1 || width>32768)throw std::invalid_argument("invalid BF16 row gather");
    bf16_gather_kernel<<<(width+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(table,index,rows,width,out);
}
size_t finite_argmax_workspace_bytes(int count) {
    if(count<1 || count>262144)throw std::invalid_argument("invalid finite argmax size");
    return size_t((count+pick_tile-1)/pick_tile)*sizeof(FiniteMaximum);
}
void finite_argmax(const float* x,int count,int32_t* token,int32_t* invalid,void* workspace,size_t bytes,void* stream) {
    const size_t need=finite_argmax_workspace_bytes(count);
    if(!x || !token || !invalid || !workspace || bytes<need)throw std::invalid_argument("invalid finite argmax buffers");
    const auto s=static_cast<cudaStream_t>(stream);const int blocks=(count+pick_tile-1)/pick_tile;
    auto* partial=static_cast<FiniteMaximum*>(workspace);
    finite_maximum_kernel<<<blocks,256,0,s>>>(x,count,partial);
    finite_pick_kernel<<<1,256,0,s>>>(partial,blocks,token,invalid);
}
void select_topk_rows(const float* scores,int stride,const IndexRows& rows,int tokens,int32_t* out,int out_stride,void* stream) {
    if(tokens<1 || tokens>kAttentionWindow || out_stride<512)throw std::invalid_argument("deepseek41 row selection: invalid geometry");
    for(int t=0;t<tokens;++t)if(rows.visible[t]<0 || rows.visible[t]>stride)throw std::invalid_argument("deepseek41 row selection: invalid visible length");
    const auto s=static_cast<cudaStream_t>(stream);
    if(core::exp_flag("ds-index-topk-merge")!=0)
        top_rows_kernel<512,false,true><<<tokens,256,0,s>>>(scores,stride,rows,out,out_stride);
    else if(core::exp_flag("ds-index-topk-radix",1)!=0)
        radix_top_rows_kernel<<<tokens,256,0,s>>>(scores,stride,rows,out,out_stride);
    else top_rows_kernel<512,false><<<tokens,512,0,s>>>(scores,stride,rows,out,out_stride);
}
void reduce_index_tensor(const float* dots,const float* weights,const uint8_t* mask,int ms,float* scores,int stride,
                          int tokens,int keys,int first,const IndexRows& rows,void* stream) {
    if(tokens<1 || tokens>kAttentionWindow || keys<1 || first<0 || stride<first+keys)
        throw std::invalid_argument("deepseek41 index tensor reduction: invalid geometry");
    if(mask && ms<(first+keys+7)/8)throw std::invalid_argument("deepseek41 index tensor reduction: invalid candidate stride");
    for(int t=0;t<tokens;++t)if(rows.visible[t]<0 || rows.visible[t]>stride)
        throw std::invalid_argument("deepseek41 index tensor reduction: invalid visible length");
    reduce_index_kernel<<<dim3((keys+255)/256,tokens),256,0,static_cast<cudaStream_t>(stream)>>>(dots,weights,mask,ms,scores,stride,keys,first,rows);
}
void candidate_topk_rows(const float* scores,int stride,const IndexRows& rows,int tokens,uint8_t* mask,int mask_stride,void* stream) {
    if(tokens<1 || tokens>kAttentionWindow)throw std::invalid_argument("deepseek41 candidate rows: invalid geometry");
    for(int t=0;t<tokens;++t)if(rows.visible[t]<0 || rows.visible[t]>stride || mask_stride<(rows.visible[t]+7)/8)
        throw std::invalid_argument("deepseek41 candidate rows: invalid visible length");
    top_rows_kernel<2048,true><<<tokens,512,0,static_cast<cudaStream_t>(stream)>>>(scores,stride,rows,mask,mask_stride);
}
size_t topk_workspace_bytes(int count) {
    if(count<1)throw std::invalid_argument("deepseek41 selection: empty score row");
    size_t size=0;
    cuda_check(cub::DeviceRadixSort::SortPairsDescending(nullptr,size,static_cast<const float*>(nullptr),static_cast<float*>(nullptr),
               static_cast<const int32_t*>(nullptr),static_cast<int32_t*>(nullptr),count));
    return size;
}
void select_topk(const float* scores,int count,int selected,int32_t* input_ids,float* sorted_scores,int32_t* sorted_ids,
                  int32_t* result,void* workspace,size_t bytes,void* stream) {
    if(count<1 || selected<1 || selected>count || selected>2048)
        throw std::invalid_argument("deepseek41 selection: invalid count or workspace");
    const auto s=static_cast<cudaStream_t>(stream);
    if(count==selected && core::exp_flag("ds-all-select",1)!=0) {
        all_selected_kernel<<<1,256,0,s>>>(scores,result,count);return;
    }
    if(!workspace || bytes<topk_workspace_bytes(count))throw std::invalid_argument("deepseek41 selection: invalid workspace");
    iota_kernel<<<(count+255)/256,256,0,s>>>(input_ids,count);
    cuda_check(cub::DeviceRadixSort::SortPairsDescending(workspace,bytes,scores,sorted_scores,input_ids,sorted_ids,count,0,32,s));
    int padded=1;while(padded<selected)padded*=2;
    sorted_kernel<<<1,256,0,s>>>(sorted_scores,sorted_ids,result,selected,padded);
}
void candidate_block_scores(const float* scores,int count,int size,float* blocks,void* stream) {
    if(count<1 || size<1)throw std::invalid_argument("deepseek41 candidates: invalid geometry");
    const int n=(count+size-1)/size;
    block_score_kernel<<<(n+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(scores,count,size,blocks);
}
void candidate_mask(const int32_t* ids,int count,uint8_t* mask,int blocks,void* stream) {
    if(count<1 || blocks<1)throw std::invalid_argument("deepseek41 candidate mask: invalid geometry");
    const auto s=static_cast<cudaStream_t>(stream);
    cuda_check(cudaMemsetAsync(mask,0,blocks,s));mask_kernel<<<(count+255)/256,256,0,s>>>(ids,count,mask,blocks);
}
} // namespace basalt::deepseek41
