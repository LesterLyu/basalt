#pragma once
// The vision FlashAttention tile with BF16 inputs and 64-wide DeepSeek heads.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace basalt::vision::deepseek_attention {
namespace {
constexpr int E = 1024, D = 64, DS = 72;
constexpr int BQ = 64, BK = 64;

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    const auto x = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&x);
}

// One block handles 64 queries of one head. All keys participate in the online softmax.
__global__ void __launch_bounds__(128) attention(const uint16_t* __restrict__ qkv,
                                                  uint16_t* __restrict__ O, int n, float scale_log2) {
    __shared__ __align__(16) uint16_t sQ[BQ * DS], sK[BK * DS], sV[BK * DS];
    const int h = blockIdx.y, q0 = blockIdx.x * BQ, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gq = lane >> 2, tg = lane & 3;
    const uint16_t* Qh = qkv + h * D;
    const uint16_t* Kh = Qh + E;
    const uint16_t* Vh = Kh + E;
    const uint4 zero4 = make_uint4(0, 0, 0, 0);
    for (int i = tid; i < BQ * 8; i += 128) {
        const int r = i / 8, c = i % 8;
        *reinterpret_cast<uint4*>(&sQ[r * DS + c * 8]) =
            q0 + r < n ? *reinterpret_cast<const uint4*>(Qh + (size_t) (q0 + r) * (3 * E) + c * 8) : zero4;
    }
    __syncthreads();
    uint32_t qa[4][4];
    const int r0 = warp * 16 + gq;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        qa[kk][0] = *reinterpret_cast<const uint32_t*>(&sQ[r0 * DS + kk * 16 + tg * 2]);
        qa[kk][1] = *reinterpret_cast<const uint32_t*>(&sQ[(r0 + 8) * DS + kk * 16 + tg * 2]);
        qa[kk][2] = *reinterpret_cast<const uint32_t*>(&sQ[r0 * DS + kk * 16 + tg * 2 + 8]);
        qa[kk][3] = *reinterpret_cast<const uint32_t*>(&sQ[(r0 + 8) * DS + kk * 16 + tg * 2 + 8]);
    }
    float o[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    for (int k0 = 0; k0 < n; k0 += BK) {
        __syncthreads();
        for (int i = tid; i < BK * 8; i += 128) {
            const int r = i / 8, c = i % 8;
            const bool ok = k0 + r < n;
            *reinterpret_cast<uint4*>(&sK[r * DS + c * 8]) = ok ? *reinterpret_cast<const uint4*>(Kh + (size_t) (k0 + r) * (3 * E) + c * 8) : zero4;
            *reinterpret_cast<uint4*>(&sV[r * DS + c * 8]) = ok ? *reinterpret_cast<const uint4*>(Vh + (size_t) (k0 + r) * (3 * E) + c * 8) : zero4;
        }
        __syncthreads();
        float s[8][4];
#pragma unroll
        for (int i = 0; i < 8; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sK[(nt * 8 + gq) * DS + kk * 16 + tg * 2]);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sK[(nt * 8 + gq) * DS + kk * 16 + tg * 2 + 8]);
                mma16816(s[nt], qa[kk], b0, b1);
            }
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int key = k0 + nt * 8 + tg * 2 + (e & 1);
                s[nt][e] = key < n ? s[nt][e] * scale_log2 : -INFINITY;
                if (e < 2) mx0 = fmaxf(mx0, s[nt][e]); else mx1 = fmaxf(mx1, s[nt][e]);
            }
#pragma unroll
        for (int o2 = 1; o2 <= 2; o2 <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o2));
        }
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        const float al0 = exp2f(m0 - mn0), al1 = exp2f(m1 - mn1);
        m0 = mn0;
        m1 = mn1;
        float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            s[nt][0] = exp2f(s[nt][0] - mn0);
            s[nt][1] = exp2f(s[nt][1] - mn0);
            s[nt][2] = exp2f(s[nt][2] - mn1);
            s[nt][3] = exp2f(s[nt][3] - mn1);
            sum0 += s[nt][0] + s[nt][1];
            sum1 += s[nt][2] + s[nt][3];
        }
        l0 = l0 * al0 + sum0;
        l1 = l1 * al1 + sum1;
#pragma unroll
        for (int dt = 0; dt < 8; ++dt) {
            o[dt][0] *= al0; o[dt][1] *= al0;
            o[dt][2] *= al1; o[dt][3] *= al1;
        }
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            uint32_t a[4];
            a[0] = pack_bf16(s[2 * kt][0], s[2 * kt][1]);
            a[1] = pack_bf16(s[2 * kt][2], s[2 * kt][3]);
            a[2] = pack_bf16(s[2 * kt + 1][0], s[2 * kt + 1][1]);
            a[3] = pack_bf16(s[2 * kt + 1][2], s[2 * kt + 1][3]);
#pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
                uint32_t b0, b1;
                const uint32_t addr = (uint32_t) __cvta_generic_to_shared(&sV[(kt * 16 + (lane & 15)) * DS + dt * 8]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(b0), "=r"(b1) : "r"(addr));
                mma16816(o[dt], a, b0, b1);
            }
        }
    }
#pragma unroll
    for (int o2 = 1; o2 <= 2; o2 <<= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, o2);
        l1 += __shfl_xor_sync(0xffffffffu, l1, o2);
    }
    const float i0 = 1.0f / l0, i1 = 1.0f / l1;
    const int ra = q0 + warp * 16 + gq, rb = ra + 8;
#pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
        const int col = h * D + dt * 8 + tg * 2;
        if (ra < n) *reinterpret_cast<uint32_t*>(&O[(size_t) ra * E + col]) = pack_bf16(o[dt][0] * i0, o[dt][1] * i0);
        if (rb < n) *reinterpret_cast<uint32_t*>(&O[(size_t) rb * E + col]) = pack_bf16(o[dt][2] * i1, o[dt][3] * i1);
    }
}

}  // namespace
}  // namespace basalt::vision::deepseek_attention
