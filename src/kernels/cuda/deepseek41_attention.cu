// Cache quantization and latent attention follow DeepSeek's inference contract (MIT; THIRD_PARTY_LICENSES).
#include "basalt/deepseek41/attention.hpp"
#include <cuda_runtime.h>
#include "basalt/core/diag.hpp"
#include <cuda_fp8.h>
#include <math_constants.h>
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace basalt::deepseek41 {
namespace {
constexpr int threads = 256;
__device__ float sum_block(float value) {
    __shared__ float warps[8];
    for (int d = 16; d; d >>= 1) value += __shfl_down_sync(0xffffffff, value, d);
    if (!(threadIdx.x & 31)) warps[threadIdx.x / 32] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? warps[threadIdx.x] : 0.f;
    if (threadIdx.x < 32) for (int d = 16; d; d >>= 1) value += __shfl_down_sync(0xffffffff, value, d);
    if (!threadIdx.x) warps[0] = value;
    __syncthreads();
    const float result = warps[0];
    __syncthreads();
    return result;
}
__device__ float max_block(float value) {
    __shared__ float warps[8];
    for (int d = 16; d; d >>= 1) value = fmaxf(value, __shfl_down_sync(0xffffffff, value, d));
    if (!(threadIdx.x & 31)) warps[threadIdx.x / 32] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? warps[threadIdx.x] : -CUDART_INF_F;
    if (threadIdx.x < 32) for (int d = 16; d; d >>= 1) value = fmaxf(value, __shfl_down_sync(0xffffffff, value, d));
    if (!threadIdx.x) warps[0] = value;
    __syncthreads();
    const float result = warps[0];
    __syncthreads();
    return result;
}
__device__ float e4_value(uint8_t code) {
    const uint32_t sign=uint32_t(code&128)<<24;
    if(code&0x78)return __uint_as_float(sign | (uint32_t((code>>3)&15)+120u)<<23 | uint32_t(code&7)<<20);
    return __uint_as_float(sign | __float_as_uint(float(code&7)*0x1p-9f));
}
__device__ uint8_t e4_code(float value) {
    return __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
}
__device__ uint8_t e2_code(float value) {
    const float a = fabsf(value);
    const int m = a <= .25f ? 0 : a < .75f ? 1 : a <= 1.25f ? 2 : a < 1.75f ? 3 :
                  a <= 2.5f ? 4 : a < 3.5f ? 5 : a <= 5.f ? 6 : 7;
    return uint8_t(m | ((__float_as_uint(value) >> 31) << 3));
}
__device__ float e2_value(uint8_t code) {
    const uint32_t m=code&7,sign=uint32_t(code&8)<<28;
    const uint32_t bits=m<2?(m?0x3f000000u:0u):(((m>>1)+126u)<<23)|((m&1)<<22);
    return __uint_as_float(sign|bits);
}
__device__ float pow2_ceil(float value) {
    const uint32_t bits = __float_as_uint(value);
    return __uint_as_float((bits & 0x7f800000u) + ((bits & 0x7fffffu) ? 0x800000u : 0));
}
__device__ float row_value(PackedRows matrix, int row, int col, int width) {
    const uint8_t* p = matrix.data + size_t(row) * matrix.stride;
    if (matrix.format == CacheFormat::window_fp8)
        return e4_value(p[col]) * __uint_as_float(uint32_t(p[width + col / 32]) << 23);
    const uint8_t c = (p[col / 2] >> (4 * (col & 1))) & 15;
    const bool e4 = matrix.format == CacheFormat::compressed_fp4;
    const uint8_t scale = p[width / 2 + col / (e4 ? 16 : 32)];
    return e2_value(c) * (e4 ? e4_value(scale) : __uint_as_float(uint32_t(scale) << 23));
}
__device__ void finish_stage(float* rows, int width, size_t stride, RowFinish f,
                             uint8_t* packed, size_t packed_stride, int vector) {
    float values[32];
    float ss = 0.f;
    for (int j = 0; j < (width + threads - 1) / threads; ++j) {
        const int col = threadIdx.x + j * threads;
        values[j] = col < width ? rows[size_t(vector) * stride + col] : 0.f;
        ss += values[j] * values[j];
    }
    const float inv = f.norm ? rsqrtf(sum_block(ss) / width + f.epsilon) : 1.f;
    for (int j = 0; j < (width + threads - 1) / threads; ++j) {
        const int col = threadIdx.x + j * threads;
        if (col < width && f.norm) values[j] *= inv * f.norm[col];
        const float other = __shfl_xor_sync(0xffffffff, values[j], 1);
        if (col >= width - f.rotary && col < width) {
            const int pair = (col - width + f.rotary) / 2;
            const float angle = float(f.first_position + vector / f.vectors_per_token * f.position_step) * f.frequencies[pair];
            float sn, cs;
            if(f.rotary_pairs) {
                const float2 sc=reinterpret_cast<const float2*>(f.rotary_pairs)[size_t(vector/f.vectors_per_token)*32+pair];
                sn=sc.x;cs=sc.y;
            } else sincosf(f.inverse ? -angle : angle, &sn, &cs);
            values[j] = (col & 1) ? other * sn + values[j] * cs : values[j] * cs - other * sn;
        }
        float result = values[j];
        if (f.quant != CacheFormat::none) {
            const bool e4 = f.quant == CacheFormat::compressed_fp4;
            const bool fp8 = f.quant == CacheFormat::window_fp8;
            const int group = e4 ? 16 : 32;
            float amax = col < width ? fabsf(result) : 0.f;
            for (int offset = group / 2; offset; offset >>= 1)
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
            const float scale = fp8 ? pow2_ceil(fmaxf(amax, 1e-4f) / 448.f) :
                                e4 ? e4_value(e4_code(fmaxf(amax, 6.f * 0x1p-9f) / 6.f)) :
                                     pow2_ceil(fmaxf(amax, 6.f * 0x1p-126f) / 6.f);
            const uint8_t code = fp8 ? e4_code(fminf(fmaxf(result / scale, -448.f), 448.f)) : e2_code(result / scale);
            result = (fp8 ? e4_value(code) : e2_value(code)) * scale;
            if (packed) {
                uint8_t* p = packed + size_t(vector) * packed_stride;
                const uint8_t next = uint8_t(__shfl_down_sync(0xffffffff, unsigned(code), 1));
                if (col < width && (fp8 || !(col & 1)))
                    p[fp8 ? col : col / 2] = fp8 ? code : uint8_t(code | (next << 4));
                if (col < width && col % group == 0)
                    p[(fp8 ? width : width / 2) + col / group] = e4 ? e4_code(scale) : uint8_t(__float_as_uint(scale) >> 23);
            }
        }
        if (col < width && !packed) rows[size_t(vector) * stride + col] = result;
    }
}
__global__ void finish_kernel(float* rows, int width, size_t stride, RowFinish f, uint8_t* packed, size_t ps) {
    finish_stage(rows, width, stride, f, packed, ps, blockIdx.x);
}
__device__ void unpack_stage(PackedRows rows, int width, float* out, int row) {
    for (int c = threadIdx.x; c < width; c += blockDim.x) out[size_t(row) * width + c] = row_value(rows, row, c, width);
}
__global__ void unpack_kernel(PackedRows rows, int width, float* out) { unpack_stage(rows, width, out, blockIdx.x); }
__device__ void compress_stage(const float* kv, const float* score, float* kh, float* sh, const float* norm,
                               float* pooled, int width, int ratio, int pos, float epsilon) {
    const int slot = pos % ratio;
    const bool complete = (pos + 1) % ratio == 0;
    float values[32], sum = 0.f;
    for (int j = 0; j < (width + threads - 1) / threads; ++j) {
        const int c = threadIdx.x + j * threads;
        float v = 0.f;
        if (c < width) {
            if (ratio == 1) v = kv[c];
            else {
                kh[slot * width + c] = kv[c]; sh[slot * width + c] = score[c];
                if (complete) {
                    const float a = expf(sh[c] - fmaxf(sh[c], sh[width + c]));
                    const float b = expf(sh[width + c] - fmaxf(sh[c], sh[width + c]));
                    v = (a * kh[c] + b * kh[width + c]) / (a + b);
                }
            }
        }
        values[j] = v; sum += v * v;
    }
    if (!complete) return;
    const float inv = rsqrtf(sum_block(sum) / width + epsilon);
    for (int j = 0; j < (width + threads - 1) / threads; ++j) {
        const int c = threadIdx.x + j * threads;
        if (c < width) pooled[c] = values[j] * inv * norm[c];
    }
}
__global__ void compress_kernel(const float* kv, const float* score, float* kh, float* sh, const float* norm,
                                float* pooled, int width, int ratio, int pos, float epsilon) {
    compress_stage(kv, score, kh, sh, norm, pooled, width, ratio, pos, epsilon);
}
__device__ void compress_pair_stage(const float* kv,const float* scores,const float* kh,const float* sh,const float* norm,
                                     float* pooled,int width,int position,int group,float epsilon) {
    const int first=group*2-position%2;
    const float *ka=first<0?kh:kv+size_t(first)*width,*kb=kv+size_t(first+1)*width;
    const float *sa=first<0?sh:scores+size_t(first)*width,*sb=scores+size_t(first+1)*width;
    float values[32],sum=0;
    for(int j=0;j<(width+threads-1)/threads;++j) {
        const int c=threadIdx.x+j*threads;float v=0;
        if(c<width) {
            const float a=expf(sa[c]-fmaxf(sa[c],sb[c])),b=expf(sb[c]-fmaxf(sa[c],sb[c]));
            v=(a*ka[c]+b*kb[c])/(a+b);
        }
        values[j]=v;sum+=v*v;
    }
    const float inv=rsqrtf(sum_block(sum)/width+epsilon);
    for(int j=0;j<(width+threads-1)/threads;++j) {
        const int c=threadIdx.x+j*threads;if(c<width)pooled[size_t(group)*width+c]=values[j]*inv*norm[c];
    }
}
__global__ void compress_rows_kernel(const float* kv,const float* scores,float* kh,float* sh,const float* norm,
                                      float* pooled,int width,int ratio,int position,float epsilon) {
    if(ratio==1)compress_stage(kv+size_t(blockIdx.x)*width,nullptr,nullptr,nullptr,norm,
                               pooled+size_t(blockIdx.x)*width,width,1,position+blockIdx.x,epsilon);
    else compress_pair_stage(kv,scores,kh,sh,norm,pooled,width,position,blockIdx.x,epsilon);
}
__device__ void save_compress_stage(const float* kv,const float* scores,float* kh,float* sh,int width,int position,int tokens,int slot) {
    const int last=tokens-1-(((position+tokens-1)&1)^slot);if(last<0)return;
    for(int c=threadIdx.x;c<width;c+=blockDim.x) {kh[slot*width+c]=kv[size_t(last)*width+c];sh[slot*width+c]=scores[size_t(last)*width+c];}
}
__global__ void save_compress_kernel(const float* kv,const float* scores,float* kh,float* sh,int width,int position,int tokens) {
    save_compress_stage(kv,scores,kh,sh,width,position,tokens,blockIdx.x);
}
__device__ void engram_stage(float* streams, const float* projected, const float* w, int dim, float epsilon, int token, int hc) {
    float* h = streams + (size_t(token) * 4 + hc) * dim;
    const float* k = projected + (size_t(token) * 5 + hc) * dim;
    const float* v = projected + (size_t(token) * 5 + 4) * dim;
    float hh = 0, kk = 0, dot = 0;
    for (int c = threadIdx.x; c < dim; c += threads) { hh += h[c] * h[c]; kk += k[c] * k[c]; dot += h[c] * w[hc * dim + c] * k[c]; }
    hh = sum_block(hh); kk = sum_block(kk); dot = sum_block(dot);
    dot *= rsqrtf(hh / dim + epsilon) * rsqrtf(kk / dim + epsilon) * rsqrtf(float(dim));
    const float gate = 1.f / (1.f + expf(-copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot)));
    for (int c = threadIdx.x; c < dim; c += threads) h[c] += gate * v[c];
}
__global__ void engram_kernel(float* h, const float* kv, const float* w, int dim, float eps) {
    engram_stage(h, kv, w, dim, eps, blockIdx.x, blockIdx.y);
}
template<bool decoded=false,bool probabilities_only=false,bool shared_rows=false,bool fixed_width=false>
__device__ void attention_stage(const float* queries, PackedRows window, PackedRows compressed, const int32_t* selected,
                                int selection_stride, AttentionRows a, const float* sinks, int heads, int width,
                                float scale, float* output, int token, int head,const float* unpacked=nullptr,int shared_count=0) {
    __shared__ float scores[1024];
    const int nw=shared_rows?shared_count:a.window_count[token];
    const int count=nw+(shared_rows?0:a.selected_count[token]);
    const size_t unpacked_base=shared_rows?0:size_t(token)*640*width;
    const float* q = queries + (size_t(token) * heads + head) * width;
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    float query[16];
    if constexpr(fixed_width) {
        static_assert(decoded && !probabilities_only && !shared_rows);
        // The same query is used for every key assigned to this warp.
#pragma unroll
        for(int j=0;j<16;++j)query[j]=q[lane+32*j];
    }
    for (int i = warp; i < count; i += 8) {
        int row=shared_rows?i:i<nw?a.window_start[token]+i:selected[token*selection_stride+i-nw];
        if (!shared_rows && i < nw && a.window_capacity > 0) row %= a.window_capacity;
        const auto matrix = i < nw ? window : compressed;
        float dot = 0;
        if (row >= 0) {
            if constexpr(fixed_width) {
                const float* key=unpacked+unpacked_base+size_t(i)*512+lane;
#pragma unroll
                for(int j=0;j<16;++j)dot+=query[j]*key[32*j];
            } else for (int c = lane; c < width; c += 32)
                dot += q[c] * (decoded?unpacked[unpacked_base+size_t(i)*width+c]:row_value(matrix, row, c, width));
        }
        for (int d = 16; d; d >>= 1) dot += __shfl_down_sync(0xffffffff, dot, d);
        if (!lane) scores[i] = row >= 0 ? dot * scale : -CUDART_INF_F;
    }
    __syncthreads();
    float maximum = sinks[head];
    for (int i = threadIdx.x; i < count; i += threads) maximum = fmaxf(maximum, scores[i]);
    maximum = max_block(maximum);
    float sum = 0;
    for (int i = threadIdx.x; i < count; i += threads) { scores[i] = expf(scores[i] - maximum); sum += scores[i]; }
    sum = sum_block(sum) + expf(sinks[head] - maximum);
    // Normalize each probability once; every output dimension uses the same rounded FP32 quotient.
    for(int i=threadIdx.x;i<count;i+=threads)scores[i]/=sum;
    __syncthreads();
    if constexpr(probabilities_only) {
        for(int i=threadIdx.x;i<count;i+=threads)output[(size_t(token)*heads+head)*640+i]=scores[i];
        return;
    }
    if constexpr(fixed_width) {
        // Compute both columns together, keeping each column's key order.
        float lo=0.f,hi=0.f;
        const float* values=unpacked+unpacked_base+threadIdx.x;
        for(int i=0;i<count;++i) {
            const float p=scores[i];
            lo+=p*values[size_t(i)*512];
            hi+=p*values[size_t(i)*512+256];
        }
        float* out=output+(size_t(token)*heads+head)*512+threadIdx.x;
        out[0]=lo;out[256]=hi;
    } else for (int c = threadIdx.x; c < width; c += threads) {
        float value = 0;
        if constexpr(decoded) {
            // Invalid selections have zero probability and a zero-filled unpacked row.
            for(int i=0;i<count;++i)value+=scores[i]*unpacked[unpacked_base+size_t(i)*width+c];
        } else for (int i = 0; i < count; ++i) {
                int row = i < nw ? a.window_start[token] + i : selected[token * selection_stride + i - nw];
                if (i < nw && a.window_capacity > 0) row %= a.window_capacity;
                if (row >= 0) value += scores[i] * row_value(i < nw ? window : compressed, row, c, width);
            }
        output[(size_t(token) * heads + head) * width + c] = value;
    }
}
template<bool overlay=false>
__device__ void selection_unpack_stage(PackedRows w,PackedRows c,const int32_t* ids,int stride,
                                       AttentionRows a,int width,float* out,int token,int i,const uint8_t* ring=nullptr,
                                       int ring_start=0) {
    const int nw=a.window_count[token],count=nw+a.selected_count[token];
    if(i>=count)return;
    int row=i<nw?a.window_start[token]+i:ids[token*stride+i-nw];
    if(i<nw && a.window_capacity>0)row%=a.window_capacity;
    PackedRows source=i<nw?w:c;
    if constexpr(overlay)if(i<nw && row<128) {
        source.data=ring;row=(ring_start+row)&127;
    }
    for(int col=threadIdx.x;col<width;col+=blockDim.x)
        out[(size_t(token)*640+i)*width+col]=row>=0?row_value(source,row,col,width):0.f;
}
__global__ void selection_unpack_kernel(PackedRows w,PackedRows c,const int32_t* ids,int stride,
                                        AttentionRows a,int width,float* out) {
    selection_unpack_stage(w,c,ids,stride,a,width,out,blockIdx.x,blockIdx.y);
}
__global__ void selection_unpack_overlay_kernel(PackedRows w,PackedRows c,const int32_t* ids,int stride,
                                                AttentionRows a,float* out,const uint8_t* ring,int ring_start) {
    selection_unpack_stage<true>(w,c,ids,stride,a,512,out,blockIdx.x,blockIdx.y,ring,ring_start);
}
template<bool fixed_width>
__global__ void decoded_attention_kernel(const float* q,PackedRows w,PackedRows c,const int32_t* ids,int stride,
                                         AttentionRows a,const float* sinks,int heads,int width,float scale,float* out,
                                         const float* unpacked) {
    attention_stage<true,false,false,fixed_width>(q,w,c,ids,stride,a,sinks,heads,width,scale,out,blockIdx.x,blockIdx.y,unpacked);
}
__global__ void draft_attention_shared_kernel(const float* q,const float* sinks,float scale,float* out,
                                               const float* unpacked,int count) {
    attention_stage<true,false,true>(q,{},{},nullptr,0,{},sinks,64,512,scale,out,blockIdx.x,blockIdx.y,unpacked,count);
}
__global__ void attention_probability_kernel(const float* q,PackedRows w,PackedRows c,const int32_t* ids,int stride,
                                             AttentionRows rows,const float* sinks,float scale,float* scores,const float* kv) {
    attention_stage<true,true>(q,w,c,ids,stride,rows,sinks,64,512,scale,scores,0,blockIdx.x,kv);
}
// Independent key tiles leave each dot and the subsequent softmax reduction order unchanged.
__device__ void attention_score_stage(const float* q,const float* kv,const int32_t* ids,
                                     AttentionRows rows,float scale,float* scores,int head,int tile) {
    const int lane=threadIdx.x&31,warp=threadIdx.x/32,nw=rows.window_count[0],count=nw+rows.selected_count[0];
    for(int i=tile*128+warp;i<min(count,(tile+1)*128);i+=8) {
        int row=i<nw?rows.window_start[0]+i:ids[i-nw];
        if(i<nw && rows.window_capacity>0)row%=rows.window_capacity;
        const bool valid=row>=0;float dot=0;
        if(valid)for(int c=lane;c<512;c+=32)dot+=q[head*512+c]*kv[i*512+c];
        for(int d=16;d;d>>=1)dot+=__shfl_down_sync(0xffffffff,dot,d);
        if(!lane)scores[head*640+i]=valid?dot*scale:-CUDART_INF_F;
    }
}
__global__ void attention_score_kernel(const float* q,const float* kv,const int32_t* ids,
                                       AttentionRows rows,float scale,float* scores) {
    attention_score_stage(q,kv,ids,rows,scale,scores,blockIdx.x,blockIdx.y);
}
__device__ void attention_normalize_stage(float* values,const float* sinks,int count,int head) {
    __shared__ float scores[1024];
    for(int i=threadIdx.x;i<count;i+=threads)scores[i]=values[head*640+i];
    __syncthreads();
    float maximum=sinks[head];
    for(int i=threadIdx.x;i<count;i+=threads)maximum=fmaxf(maximum,scores[i]);
    maximum=max_block(maximum);float sum=0;
    for(int i=threadIdx.x;i<count;i+=threads) {scores[i]=expf(scores[i]-maximum);sum+=scores[i];}
    sum=sum_block(sum)+expf(sinks[head]-maximum);
    for(int i=threadIdx.x;i<count;i+=threads)scores[i]/=sum;
    __syncthreads();
    for(int i=threadIdx.x;i<count;i+=threads)values[head*640+i]=scores[i];
}
__global__ void attention_normalize_kernel(float* scores,const float* sinks,int count) {
    attention_normalize_stage(scores,sinks,count,blockIdx.x);
}
__device__ void attention_value_stage(const float* scores,const float* kv,int count,float* out,int head,int tile) {
    const int col=tile*128+threadIdx.x;float value=0;
    for(int i=0;i<count;++i)value+=scores[head*640+i]*kv[i*512+col];
    out[head*512+col]=value;
}
__global__ void attention_value_kernel(const float* scores,const float* kv,int count,float* out) {
    attention_value_stage(scores,kv,count,out,blockIdx.x,blockIdx.y);
}
__global__ void attention_kernel(const float* q, PackedRows w, PackedRows c, const int32_t* ids, int stride,
                                 AttentionRows rows, const float* sinks, int heads, int width, float scale, float* out) {
    attention_stage(q, w, c, ids, stride, rows, sinks, heads, width, scale, out, blockIdx.x, blockIdx.y);
}
#include "deepseek41_tensor_attention.cuh"
__global__ void decode_attention_bf16_kernel(PackedRows src,int first,uint16_t* out) {
    const int row=first+blockIdx.x;
    for(int c=threadIdx.x;c<512;c+=blockDim.x)
        out[size_t(row)*512+c]=tensor_bf16(row_value(src,row,c,512));
}
__device__ void matrix_prepare_stage(PackedRows w,PackedRows c,const int32_t* ids,AttentionRows rows,int stride,
                                      float* kv,float* kt,int row) {
    const int nw=rows.window_count[0],count=nw+rows.selected_count[0];
    int selected=-1;
    if(row<count)selected=row<nw?rows.window_start[0]+row:ids[row-nw];
    if(row<nw && rows.window_capacity>0)selected%=rows.window_capacity;
    for(int col=threadIdx.x;col<512;col+=blockDim.x) {
        const float value=selected>=0?row_value(row<nw?w:c,selected,col,512):0.f;
        kv[size_t(row)*512+col]=value;kt[size_t(col)*stride+row]=value;
    }
}
__global__ void matrix_prepare_kernel(PackedRows w,PackedRows c,const int32_t* ids,AttentionRows rows,int stride,float* kv,float* kt) {
    matrix_prepare_stage(w,c,ids,rows,stride,kv,kt,blockIdx.x);
}
__device__ void matrix_softmax_stage(float* scores,const float* sinks,const int32_t* ids,AttentionRows rows,int stride,
                                      float scale,int head) {
    __shared__ float values[640];
    const int nw=rows.window_count[0],count=nw+rows.selected_count[0];float maximum=sinks[head];
    for(int i=threadIdx.x;i<stride;i+=blockDim.x) {
        const bool valid=i<count && (i<nw || ids[i-nw]>=0);
        values[i]=valid?scores[size_t(head)*stride+i]*scale:-CUDART_INF_F;maximum=fmaxf(maximum,values[i]);
    }
    maximum=max_block(maximum);float sum=0;
    for(int i=threadIdx.x;i<stride;i+=blockDim.x) {values[i]=expf(values[i]-maximum);sum+=values[i];}
    sum=sum_block(sum)+expf(sinks[head]-maximum);
    for(int i=threadIdx.x;i<stride;i+=blockDim.x) {
        scores[size_t(head)*stride+i]=values[i]/sum;
    }
}
__global__ void matrix_softmax_kernel(float* scores,const float* sinks,const int32_t* ids,AttentionRows rows,int stride,float scale) {
    matrix_softmax_stage(scores,sinks,ids,rows,stride,scale,blockIdx.x);
}
bool valid_format(CacheFormat f) { return int(f) >= 1 && int(f) <= 3; }
}

void sparse_attention_split(const float* q,PackedRows w,PackedRows c,const int32_t* ids,int selection_stride,
                              const AttentionRows& rows,const float* sinks,float scale,float* out,float* scratch,void* stream) {
    const int count=rows.window_count[0]+rows.selected_count[0];
    if(count<1 || count>640)throw std::invalid_argument("invalid split attention key count");
    const auto s=static_cast<cudaStream_t>(stream);float* scores=scratch+640*512;
    selection_unpack_kernel<<<dim3(1,640),256,0,s>>>(w,c,ids,selection_stride,rows,512,scratch);
    if(count>=256 && core::exp_flag("ds-attention-key-tiles",1)!=0) {
        attention_score_kernel<<<dim3(64,(count+127)/128),256,0,s>>>(q,scratch,ids,rows,scale,scores);
        attention_normalize_kernel<<<64,256,0,s>>>(scores,sinks,count);
    } else attention_probability_kernel<<<64,256,0,s>>>(q,w,c,ids,selection_stride,rows,sinks,scale,scores,scratch);
    attention_value_kernel<<<dim3(64,4),128,0,s>>>(scores,scratch,count,out);
}
void prepare_attention_matrix(PackedRows w,PackedRows c,const int32_t* ids,const AttentionRows& rows,int stride,
                               float* kv,float* kt,void* stream) {
    if(stride<1 || stride>640 || stride%32)throw std::invalid_argument("invalid attention matrix stride");
    const auto s=static_cast<cudaStream_t>(stream);
    matrix_prepare_kernel<<<stride,256,0,s>>>(w,c,ids,rows,stride,kv,kt);
}
void softmax_attention_matrix(float* scores,const float* sinks,const int32_t* ids,const AttentionRows& rows,
                               int stride,float scale,void* stream) {
    if(stride<1 || stride>640 || stride%32)throw std::invalid_argument("invalid attention matrix stride");
    matrix_softmax_kernel<<<64,256,0,static_cast<cudaStream_t>(stream)>>>(scores,sinks,ids,rows,stride,scale);
}
__global__ void rotary_pairs_kernel(float2* out,const float* frequencies,int first,int count,bool inverse) {
    const int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count*32)return;
    const float angle=float(first+i/32)*frequencies[i%32];
    float sn,cs;sincosf(inverse?-angle:angle,&sn,&cs);
    out[i]=make_float2(sn,cs);
}
void make_rotary_pairs(float* out,const float* frequencies,int first,int count,bool inverse,void* stream) {
    rotary_pairs_kernel<<<(count*32+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<float2*>(out),frequencies,first,count,inverse);
}
void finish_rows(float* rows, int vectors, int width, size_t stride, const RowFinish& f,
                 uint8_t* packed, size_t packed_stride, void* stream) {
    if (vectors < 1 || width < 1 || width > 8192 || stride < size_t(width) || f.vectors_per_token < 1 ||
        f.rotary < 0 || f.rotary > width || (f.rotary && (!f.frequencies || (width & 1) || (f.rotary & 1))) ||
        (f.rotary_pairs && (f.rotary!=64 || f.position_step!=1)) ||
        (f.quant != CacheFormat::none && (!valid_format(f.quant) || width % 32)) ||
        (packed && (f.quant == CacheFormat::none || packed_stride < cache_row_bytes(width, f.quant))))
        throw std::invalid_argument("deepseek41 finish: invalid geometry or cache format");
    finish_kernel<<<vectors, threads, 0, static_cast<cudaStream_t>(stream)>>>(rows, width, stride, f, packed, packed_stride);
}
void unpack_rows(const uint8_t* packed, int rows, int width, size_t stride, CacheFormat format, float* output, void* stream) {
    if (!valid_format(format) || rows < 1 || width < 32 || width % 32 || stride < cache_row_bytes(width, format))
        throw std::invalid_argument("deepseek41 unpack: invalid cache geometry");
    unpack_kernel<<<rows, threads, 0, static_cast<cudaStream_t>(stream)>>>({packed, stride, format}, width, output);
}
void update_window_ring(uint8_t* ring,const uint8_t* extended,int position,int tokens,size_t bytes,void* stream) {
    if(position<0 || tokens<0 || !bytes)throw std::invalid_argument("deepseek41 ring: invalid geometry");
    if(!tokens)return;
    const int n=std::min(tokens,128),first=tokens-n,start=(position+first)%128,a=std::min(n,128-start);
    const auto s=static_cast<cudaStream_t>(stream);
    auto status=cudaMemcpyAsync(ring+size_t(start)*bytes,extended+size_t(128+first)*bytes,size_t(a)*bytes,cudaMemcpyDeviceToDevice,s);
    if(status==cudaSuccess && a<n)status=cudaMemcpyAsync(ring,extended+size_t(128+first+a)*bytes,size_t(n-a)*bytes,cudaMemcpyDeviceToDevice,s);
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
__global__ void rollback_window_kernel(const __grid_constant__ WindowRollback window,int position,int proposed,int accepted) {
    const auto& l=window.layers[blockIdx.x];
    constexpr int columns=528/sizeof(uint4);
    auto* ring=reinterpret_cast<uint4*>(l.ring);
    const auto* old=reinterpret_cast<const uint4*>(l.extended);
    for(int i=threadIdx.x;i<(proposed-accepted)*columns;i+=blockDim.x) {
        const int row=accepted+i/columns,col=i%columns;
        ring[((position+row)&127)*columns+col]=old[row*columns+col];
    }
    if(l.kv_history)for(int slot=0;slot<2;++slot) {
        const int last=accepted-1-(((position+accepted+1)&1)^slot);
        const auto* kv=last<0?l.saved_kv+slot*512:l.raw_kv+last*512;
        const auto* score=last<0?l.saved_score+slot*512:l.raw_gate+last*512;
        for(int i=threadIdx.x;i<512;i+=blockDim.x) {
            l.kv_history[slot*512+i]=kv[i];
            l.score_history[slot*512+i]=score[i];
        }
    }
}
void rollback_window(const WindowRollback& window,int position,int proposed,int accepted,void* stream) {
    if(position<0 || accepted<0 || accepted>=proposed || proposed>128)
        throw std::invalid_argument("deepseek41 rollback: invalid window geometry");
    rollback_window_kernel<<<40,128,0,static_cast<cudaStream_t>(stream)>>>(window,position,proposed,accepted);
}
__global__ void append_window_snapshot_kernel(uint4* ring,uint4* extended,int position) {
    constexpr int columns=528/sizeof(uint4);
    if(threadIdx.x>=columns)return;
    const int row=blockIdx.x,col=threadIdx.x;
    auto& dst=ring[((position+row)&127)*columns+col];
    extended[row*columns+col]=dst;
    dst=extended[(128+row)*columns+col];
}
void append_window_snapshot(uint8_t* ring,uint8_t* extended,int position,int tokens,void* stream) {
    if(position<0 || tokens<1 || tokens>6)throw std::invalid_argument("deepseek41 ring snapshot: invalid geometry");
    append_window_snapshot_kernel<<<tokens,64,0,static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<uint4*>(ring),reinterpret_cast<uint4*>(extended),position);
}
bool compress_step(const float* kv, const float* score, float* kh, float* sh, const float* norm, float* pooled,
                   int width, int ratio, int pos, float epsilon, void* stream) {
    if ((ratio != 1 && ratio != 2) || pos < 0 || width < 1 || width > 8192 || !norm ||
        (ratio == 2 && (!score || !kh || !sh))) throw std::invalid_argument("deepseek41 compressor: invalid geometry");
    compress_kernel<<<1, threads, 0, static_cast<cudaStream_t>(stream)>>>(kv, score, kh, sh, norm, pooled, width, ratio, pos, epsilon);
    return (pos + 1) % ratio == 0;
}
int compress_rows(const float* kv,const float* scores,float* kh,float* sh,const float* norm,float* pooled,int width,
                    int ratio,int position,int tokens,float epsilon,void* stream) {
    if(tokens<1 || width<1 || width>8192 || position<0 || !norm || (ratio!=1 && ratio!=2) || (ratio==2 && (!scores || !kh || !sh)))
        throw std::invalid_argument("deepseek41 compressor rows: invalid geometry");
    if(tokens==1)return compress_step(kv,scores,kh,sh,norm,pooled,width,ratio,position,epsilon,stream)?1:0;
    const int n=ratio==1?tokens:(position%2+tokens)/2;const auto s=static_cast<cudaStream_t>(stream);
    if(n)compress_rows_kernel<<<n,threads,0,s>>>(kv,scores,kh,sh,norm,pooled,width,ratio,position,epsilon);
    if(ratio==2)save_compress_kernel<<<2,threads,0,s>>>(kv,scores,kh,sh,width,position,tokens);
    return n;
}
void engram_add(float* h, const float* kv, const float* w, int hidden, int tokens, float epsilon, void* stream) {
    if (hidden < 1 || tokens < 1) throw std::invalid_argument("deepseek41 Engram: invalid geometry");
    engram_kernel<<<dim3(tokens, 4), threads, 0, static_cast<cudaStream_t>(stream)>>>(h, kv, w, hidden, epsilon);
}
void sparse_attention(const float* q, PackedRows w, PackedRows c, const int32_t* ids, int stride,
                      const AttentionRows& a, const float* sinks, int heads, int width, int tokens,
                      float scale, float* out, void* stream) {
    if (tokens < 1 || tokens > kAttentionWindow || heads < 1 || width < 32 || width % 32 || !sinks)
        throw std::invalid_argument("deepseek41 attention: invalid geometry");
    for (int t = 0; t < tokens; ++t) {
        if (a.window_count[t] < 0 || a.selected_count[t] < 0 || a.window_start[t] < 0 ||
            a.window_count[t] + a.selected_count[t] > 1024 ||
            (a.window_count[t] && (!w.data || !valid_format(w.format) || w.stride < cache_row_bytes(width, w.format))) ||
            (a.selected_count[t] && (!c.data || !ids || stride < a.selected_count[t] || !valid_format(c.format) ||
                                    c.stride < cache_row_bytes(width, c.format))))
            throw std::invalid_argument("deepseek41 attention: invalid row selection");
    }
    attention_kernel<<<dim3(tokens, heads), threads, 0, static_cast<cudaStream_t>(stream)>>>(q, w, c, ids, stride, a, sinks, heads, width, scale, out);
}
void sparse_attention_predecoded(const float* q,PackedRows w,PackedRows c,const int32_t* ids,int stride,
                                 const AttentionRows& a,const float* sinks,int heads,int width,int tokens,
                                 float scale,float* out,float* scratch,void* stream,bool fixed_width,
                                 const uint8_t* ring,int ring_start) {
    if(tokens<1 || tokens>kAttentionWindow || heads<1 || width!=512 || !scratch || !sinks)
        throw std::invalid_argument("deepseek41 decoded attention: invalid geometry");
    if(ring && (w.format!=CacheFormat::window_fp8 || a.window_capacity || ring_start<0 || ring_start>=128 || tokens>6))
        throw std::invalid_argument("deepseek41 decoded attention: invalid overlay");
    for(int t=0;t<tokens;++t) {
        if(a.window_count[t]<0 || a.selected_count[t]<0 || a.window_start[t]<0 ||
           a.window_count[t]+a.selected_count[t]>640 ||
           (a.window_count[t] && (!w.data || !valid_format(w.format) || w.stride<cache_row_bytes(width,w.format))) ||
           (a.selected_count[t] && (!c.data || !ids || stride<a.selected_count[t] || !valid_format(c.format) ||
                                    c.stride<cache_row_bytes(width,c.format))))
            throw std::invalid_argument("deepseek41 decoded attention: invalid row selection");
    }
    auto s=static_cast<cudaStream_t>(stream);
    for(int first=0;first<tokens;first+=8) {
        const int nr=std::min(8,tokens-first);AttentionRows tile;tile.window_capacity=a.window_capacity;
        int maximum=0;
        for(int i=0;i<nr;++i) {
            tile.window_start[i]=a.window_start[first+i];tile.window_count[i]=a.window_count[first+i];
            tile.selected_count[i]=a.selected_count[first+i];maximum=std::max(maximum,tile.window_count[i]+tile.selected_count[i]);
        }
        const auto* selection=ids?ids+first*stride:nullptr;
        if(maximum) {
            if(ring)selection_unpack_overlay_kernel<<<dim3(nr,maximum),threads,0,s>>>(w,c,selection,stride,tile,scratch,ring,ring_start);
            else selection_unpack_kernel<<<dim3(nr,maximum),threads,0,s>>>(w,c,selection,stride,tile,width,scratch);
        }
        if(fixed_width)
            decoded_attention_kernel<true><<<dim3(nr,heads),threads,0,s>>>(q+size_t(first)*heads*width,w,c,selection,stride,
                tile,sinks,heads,width,scale,out+size_t(first)*heads*width,scratch);
        else
            decoded_attention_kernel<false><<<dim3(nr,heads),threads,0,s>>>(q+size_t(first)*heads*width,w,c,selection,stride,
                tile,sinks,heads,width,scale,out+size_t(first)*heads*width,scratch);
    }
}
void sparse_attention_draft_shared(const float* q,PackedRows extended,int first,int count,const float* sinks,
                                   float scale,float* out,float* scratch,void* stream) {
    if(!q || !extended.data || extended.format!=CacheFormat::window_fp8 ||
       extended.stride<cache_row_bytes(512,extended.format) || !sinks || !out || !scratch ||
       first<0 || count<5 || first>133-count)
        throw std::invalid_argument("deepseek41 shared draft attention: invalid geometry");
    unpack_rows(extended.data+size_t(first)*extended.stride,count,512,extended.stride,extended.format,scratch,stream);
    draft_attention_shared_kernel<<<dim3(5,64),threads,0,static_cast<cudaStream_t>(stream)>>>(q,sinks,scale,out,scratch,count);
}
void decode_attention_bf16(PackedRows packed,int first,int rows,uint16_t* output,void* stream) {
    if(!packed.data || (packed.format!=CacheFormat::compressed_fp4 && packed.format!=CacheFormat::window_fp8) ||
       packed.stride<cache_row_bytes(512,packed.format) || first<0 || rows<1 || !output)
        throw std::invalid_argument("deepseek41: invalid BF16 attention cache range");
    decode_attention_bf16_kernel<<<rows,256,0,static_cast<cudaStream_t>(stream)>>>(packed,first,output);
}

void sparse_attention_tensor(const float* q,PackedRows w,PackedRows c,const int32_t* ids,int stride,
                             const AttentionRows& a,const float* sinks,int heads,int width,int tokens,
                             float scale,float* out,void* stream,const uint16_t* decoded,const uint16_t* decoded_window) {
    if(tokens<1 || tokens>kAttentionWindow || heads<16 || heads%16 || width!=512 || !sinks)
        throw std::invalid_argument("deepseek41 tensor attention: invalid geometry");
    for(int t=0;t<tokens;++t) {
        if(a.window_count[t]<0 || a.selected_count[t]<0 || a.window_start[t]<0 ||
           a.window_count[t]+a.selected_count[t]>640 ||
           (a.window_count[t] && (!w.data || !valid_format(w.format) || w.stride<cache_row_bytes(width,w.format))) ||
           (a.selected_count[t] && (!c.data || !ids || stride<a.selected_count[t] || !valid_format(c.format) ||
                                    c.stride<cache_row_bytes(width,c.format))))
            throw std::invalid_argument("deepseek41 tensor attention: invalid row selection");
    }
    const auto s=static_cast<cudaStream_t>(stream);
    const dim3 grid(heads/16,tokens);
    const bool local=core::exp_flag("ds-attention-local-rows")!=0;
    if(decoded || decoded_window) {
        if(local)tensor_attention_kernel<true,true><<<grid,threads,0,s>>>(q,w,c,ids,stride,a,sinks,heads,scale,out,decoded,decoded_window);
        else if(core::exp_flag("ds-attention-packed-prob")!=0)
            tensor_attention_kernel<false,true,true><<<grid,threads,0,s>>>(q,w,c,ids,stride,a,sinks,heads,scale,out,decoded,decoded_window);
        else tensor_attention_kernel<false,true><<<grid,threads,0,s>>>(q,w,c,ids,stride,a,sinks,heads,scale,out,decoded,decoded_window);
    } else {
        if(local)tensor_attention_kernel<true,false><<<grid,threads,0,s>>>(q,w,c,ids,stride,a,sinks,heads,scale,out,nullptr,nullptr);
        else tensor_attention_kernel<false,false><<<grid,threads,0,s>>>(q,w,c,ids,stride,a,sinks,heads,scale,out,nullptr,nullptr);
    }
}
}  // namespace basalt::deepseek41
