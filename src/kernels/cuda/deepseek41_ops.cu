// Hyper-connection and routing equations adapted from Project Maya (MIT; see THIRD_PARTY_LICENSES).
// Copyright (c) 2026 Peasant Smith and the Project Maya contributors.
#include "basalt/deepseek41/ops.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/log.hpp"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace basalt::deepseek41 {
namespace {
constexpr int block_threads = 256;

__device__ __forceinline__ float warp_sum(float x) {
    for (int d = 16; d; d >>= 1) x += __shfl_down_sync(0xffffffffu, x, d);
    return x;
}
__device__ float block_sum(float x, float* scratch, int tid) {
    x = warp_sum(x);
    if ((tid & 31) == 0) scratch[tid / 32] = x;
    __syncthreads();
    x = tid < block_threads / 32 ? scratch[tid] : 0.f;
    if (tid < 32) {
        x = warp_sum(x);
        if (tid == 0) scratch[0] = x;
    }
    __syncthreads();
    return scratch[0];
}
__device__ void project_stage(const float* h, const float* fn, float* mixes, int hidden, float eps,
                             int token, int mix, int tid, float* scratch) {
    const int n = 4 * hidden;
    const float* x = h + (size_t) token * n;
    const float* w = fn + (size_t) mix * n;
    float dot = 0.f, squares = 0.f;
    for (int i = tid; i < n; i += block_threads) {
        dot = fmaf(x[i], w[i], dot);
        squares = fmaf(x[i], x[i], squares);
    }
    dot = block_sum(dot, scratch, tid);
    squares = block_sum(squares, scratch + 8, tid);
    if (tid == 0) mixes[(size_t) token * 24 + mix] = dot * rsqrtf(squares / n + eps);
}
__global__ void project_kernel(const float* h, const float* fn, float* mixes, int hidden, float eps) {
    __shared__ float scratch[16];
    project_stage(h, fn, mixes, hidden, eps, blockIdx.y, blockIdx.x, threadIdx.x, scratch);
}
// Reuse each activation across four mixes, with the same per-lane FMA and warp reduction order.
__global__ void project_grouped_kernel(const float* h, const float* fn, float* mixes, int hidden, float eps) {
    constexpr int group = 4, warps = block_threads / 32;
    __shared__ float partial[group + 1][warps];
    const int n = 4 * hidden, token = blockIdx.y, mix = blockIdx.x * group;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const float* x = h + size_t(token) * n;
    const float* w = fn + size_t(mix) * n;
    float dot[group]{}, squares = 0.f;
    for (int i = tid; i < n; i += block_threads) {
        const float v = x[i];
        #pragma unroll
        for (int j = 0; j < group; ++j) dot[j] = fmaf(v, w[size_t(j) * n + i], dot[j]);
        squares = fmaf(v, v, squares);
    }
    #pragma unroll
    for (int j = 0; j < group; ++j) {
        const float sum = warp_sum(dot[j]);
        if (lane == 0) partial[j][warp] = sum;
    }
    squares = warp_sum(squares);
    if (lane == 0) partial[group][warp] = squares;
    __syncthreads();
    if (warp < group) {
        const float sum = warp_sum(lane < warps ? partial[warp][lane] : 0.f);
        const float norm = warp_sum(lane < warps ? partial[group][lane] : 0.f);
        if (lane == 0) mixes[size_t(token) * 24 + mix + warp] = sum * rsqrtf(norm / n + eps);
    }
}
// Each thread reuses its four weights across a tile of tokens. Each dot product
// still visits exactly the same channels, FMAs and reduction as the grouped path.
template<int tile>
__global__ void project_token_tile_kernel(const float* h,const float* fn,float* mixes,int hidden,float eps) {
    constexpr int group=4,warps=block_threads/32;
    __shared__ float partial[tile][group+1][warps];
    const int n=4*hidden,mix=blockIdx.x*group,token=blockIdx.y*tile;
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32;
    const float* x=h+size_t(token)*n;
    const float* w=fn+size_t(mix)*n;
    float dots[tile][group]{},squares[tile]{};
    for(int i=tid;i<n;i+=block_threads) {
        float weights[group];
        #pragma unroll
        for(int j=0;j<group;++j)weights[j]=w[size_t(j)*n+i];
        #pragma unroll
        for(int t=0;t<tile;++t) {
            const float v=x[size_t(t)*n+i];
            #pragma unroll
            for(int j=0;j<group;++j)dots[t][j]=fmaf(v,weights[j],dots[t][j]);
            squares[t]=fmaf(v,v,squares[t]);
        }
    }
    #pragma unroll
    for(int t=0;t<tile;++t) {
        #pragma unroll
        for(int j=0;j<group;++j) {
            const float sum=warp_sum(dots[t][j]);
            if(lane==0)partial[t][j][warp]=sum;
        }
        const float norm=warp_sum(squares[t]);
        if(lane==0)partial[t][group][warp]=norm;
    }
    __syncthreads();
    for(int i=warp;i<tile*group;i+=warps) {
        const int t=i/group,j=i%group;
        const float sum=warp_sum(lane<warps?partial[t][j][lane]:0.f);
        const float norm=warp_sum(lane<warps?partial[t][group][lane]:0.f);
        if(lane==0)mixes[size_t(token+t)*24+mix+j]=sum*rsqrtf(norm/n+eps);
    }
}

__device__ __forceinline__ float row_sum(float x) {
    x += __shfl_xor_sync(0xffffffffu, x, 1);
    return x + __shfl_xor_sync(0xffffffffu, x, 2);
}
__device__ __forceinline__ float col_sum(float x) {
    x += __shfl_xor_sync(0xffffffffu, x, 4);
    return x + __shfl_xor_sync(0xffffffffu, x, 8);
}
__device__ void split_stage(const float* mixes, const float* scale, const float* base,
                           float* pre, float* post, float* residual, int iterations, float eps, int token, int lane) {
    const float* m = mixes + (size_t) token * 24;
    if (lane < 4) {
        pre[token * 4 + lane] = 1.f / (1.f + expf(-(m[lane] * scale[0] + base[lane]))) + eps;
        post[token * 4 + lane] = 2.f / (1.f + expf(-(m[4 + lane] * scale[1] + base[4 + lane])));
    }
    float c = lane < 16 ? m[8 + lane] * scale[2] + base[8 + lane] : -FLT_MAX;
    float max = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 1));
    max = fmaxf(max, __shfl_xor_sync(0xffffffffu, max, 2));
    c = lane < 16 ? expf(c - max) : 0.f;
    const float sum = row_sum(c);
    c = lane < 16 ? c / sum + eps : 0.f;
    c /= col_sum(c) + eps;
    for (int i = 1; i < iterations; ++i) {
        c /= row_sum(c) + eps;
        c /= col_sum(c) + eps;
    }
    if (lane < 16) residual[token * 16 + lane] = c;
}
__global__ void split_kernel(const float* m, const float* scale, const float* base, float* pre, float* post,
                             float* residual, int iterations, float eps) {
    split_stage(m, scale, base, pre, post, residual, iterations, eps, blockIdx.x, threadIdx.x);
}
__device__ void reduce_stage(const float* h, const float* pre, const float* norm, float* out, int hidden, float eps,
                            int token, int tid, float* scratch) {
    const float* x = h + (size_t) token * 4 * hidden;
    const float* p = pre + token * 4;
    float* y = out + (size_t) token * hidden;
    float squares = 0.f;
    for (int i = tid; i < hidden; i += block_threads) {
        float a = 0.f;
        for (int s = 0; s < 4; ++s) a = fmaf(p[s], x[s * hidden + i], a);
        y[i] = a;
        squares = fmaf(a, a, squares);
    }
    if (!norm) return;
    const float inv = rsqrtf(block_sum(squares, scratch, tid) / hidden + eps);
    for (int i = tid; i < hidden; i += block_threads) y[i] = y[i] * inv * norm[i];
}
__global__ void reduce_kernel(const float* h, const float* pre, const float* norm, float* out, int hidden, float eps) {
    __shared__ float scratch[8];
    reduce_stage(h, pre, norm, out, hidden, eps, blockIdx.x, threadIdx.x, scratch);
}
__device__ void expand_stage(const float* branch, const float* h, const float* post, const float* residual,
                            float* out, int hidden, int token, int i) {
    if (i >= hidden) return;
    const float* x = h + (size_t) token * 4 * hidden;
    const float* c = residual + token * 16;
    float r[4];
    for (int j = 0; j < 4; ++j) r[j] = x[j * hidden + i];
    const float y = branch[(size_t) token * hidden + i];
    for (int k = 0; k < 4; ++k) {
        float a = post[token * 4 + k] * y;
        for (int j = 0; j < 4; ++j) a = fmaf(c[j * 4 + k], r[j], a);
        out[((size_t) token * 4 + k) * hidden + i] = a;
    }
}
__global__ void expand_kernel(const float* y, const float* h, const float* post, const float* residual, float* out, int hidden) {
    expand_stage(y, h, post, residual, out, hidden, blockIdx.y, blockIdx.x * block_threads + threadIdx.x);
}
__device__ void route_stage(const float* logits, const float* bias, int32_t* ids, float* weights,
                           int n, int k, float scale, int token, int tid, float* scores, float* ranked) {
    for (int i = tid; i < n; i += block_threads) {
        const float l = logits[(size_t) token * n + i];
        scores[i] = sqrtf(l > 20.f ? l : log1pf(expf(l)));
        ranked[i] = scores[i] + (bias ? bias[i] : 0.f);
    }
    __syncthreads();
    if (tid >= 32) return;
    int picked[6];
    float sum = 0.f;
    for (int r = 0; r < k; ++r) {
        int best = n;
        float value = -CUDART_INF_F;
        for (int i = tid; i < n; i += 32) if (best == n || ranked[i] > value) {
            best = i; value = ranked[i];
        }
        for (int delta = 16; delta; delta >>= 1) {
            const float other_value = __shfl_down_sync(0xffffffffu, value, delta);
            const int other = __shfl_down_sync(0xffffffffu, best, delta);
            if (other_value > value || (other_value == value && other < best)) {
                best = other; value = other_value;
            }
        }
        if (tid == 0) {
            picked[r] = best;
            ranked[best] = -CUDART_INF_F;
            sum += scores[best];
        }
        // The next search sees the selected row masked; ties retain the lowest expert ID.
        __syncwarp();
    }
    if (tid == 0) for (int r = 0; r < k; ++r) {
        ids[token * k + r] = picked[r];
        weights[token * k + r] = scores[picked[r]] / (sum + 1e-20f) * scale;
    }
}
__global__ void route_kernel(const float* logits, const float* bias, int32_t* ids, float* weights, int n, int k, float scale) {
    __shared__ float scores[384], ranked[384];
    route_stage(logits, bias, ids, weights, n, k, scale, blockIdx.x, threadIdx.x, scores, ranked);
}
void check(const char* op) {
    const auto e = cudaGetLastError();
    if (e != cudaSuccess) { basalt::log::error("deepseek41", "%s: %s", op, cudaGetErrorString(e)); std::exit(1); }
}
void dimensions(int hidden, int tokens) {
    if (hidden <= 0 || tokens < 0) throw std::invalid_argument("deepseek41: invalid operation dimensions");
}
}  // namespace

void hc_project(const float* h, const float* fn, float* mixes, int hidden, int tokens, float eps, void* stream) {
    dimensions(hidden, tokens); if (!tokens) return;
    project_kernel<<<dim3(24, tokens), block_threads, 0, (cudaStream_t) stream>>>(h, fn, mixes, hidden, eps);
    check("hc_project");
}
void hc_project_grouped(const float* h, const float* fn, float* mixes, int hidden, int tokens, float eps, void* stream) {
    dimensions(hidden, tokens); if (!tokens) return;
    const auto s=static_cast<cudaStream_t>(stream);
    const double tile_option=core::exp_flag("ds-prefill-hc-token-tile",8);
    const int tile=tile_option>=8?8:4;
    const int tiled=tile_option!=0?tokens/tile*tile:0;
    if(tiled) {
        if(tile==8)project_token_tile_kernel<8><<<dim3(6,tiled/8),block_threads,0,s>>>(h,fn,mixes,hidden,eps);
        else project_token_tile_kernel<4><<<dim3(6,tiled/4),block_threads,0,s>>>(h,fn,mixes,hidden,eps);
    }
    if(tiled<tokens)
        project_grouped_kernel<<<dim3(6,tokens-tiled),block_threads,0,s>>>(h+size_t(tiled)*4*hidden,fn,mixes+size_t(tiled)*24,hidden,eps);
    check("hc_project_grouped");
}
void hc_split(const float* mixes, const float* scale, const float* base, float* pre, float* post, float* residual,
              int tokens, int iterations, float eps, void* stream) {
    if (tokens < 0 || iterations <= 0 || !(eps > 0.f)) throw std::invalid_argument("deepseek41: invalid Sinkhorn dimensions");
    if (!tokens) return;
    split_kernel<<<tokens, 32, 0, (cudaStream_t) stream>>>(mixes, scale, base, pre, post, residual, iterations, eps);
    check("hc_split");
}
void hc_reduce(const float* h, const float* pre, const float* norm, float* out, int hidden, int tokens, float eps, void* stream) {
    dimensions(hidden, tokens); if (!tokens) return;
    reduce_kernel<<<tokens, block_threads, 0, (cudaStream_t) stream>>>(h, pre, norm, out, hidden, eps);
    check("hc_reduce");
}
void hc_expand(const float* y, const float* h, const float* post, const float* residual, float* out, int hidden, int tokens, void* stream) {
    dimensions(hidden, tokens); if (!tokens) return;
    expand_kernel<<<dim3((hidden + block_threads - 1) / block_threads, tokens), block_threads, 0, (cudaStream_t) stream>>>(y, h, post, residual, out, hidden);
    check("hc_expand");
}
void route(const float* logits, const float* bias, int32_t* ids, float* weights,
           int experts, int k, int tokens, float scale, void* stream) {
    if ((experts != 384 && experts != 128) || k <= 0 || k > 6 || tokens < 0)
        throw std::invalid_argument("deepseek41: invalid router dimensions");
    if (!tokens) return;
    route_kernel<<<tokens, block_threads, 0, (cudaStream_t) stream>>>(logits, bias, ids, weights, experts, k, scale);
    check("route");
}
}  // namespace basalt::deepseek41
