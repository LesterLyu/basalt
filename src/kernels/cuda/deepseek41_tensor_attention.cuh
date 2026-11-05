/*
Adapted from Project Maya sparse attention; retained MIT notice:
MIT License

Copyright (c) 2026 Peasant Smith and the Project Maya contributors
Copyright (c) 2026 Niko1221 and the Strata contributors
Copyright (c) 2026 Jesús Gulfo and the Basalt contributors (Windows host registration budget policy)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/
__device__ __forceinline__ uint16_t tensor_bf16(float value) {
    uint32_t bits=__float_as_uint(value);bits+=0x7fffu+((bits>>16)&1u);return uint16_t(bits>>16);
}
__device__ __forceinline__ float tensor_float(uint16_t value){return __uint_as_float(uint32_t(value)<<16);}
__device__ __forceinline__ void mma_ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}
__device__ __forceinline__ void mma_ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}

__device__ __forceinline__ void mma_bf16(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void split_bf16x2(float2 v, uint32_t& hi, uint32_t& lo) {
    const uint16_t h0 = tensor_bf16(v.x), h1 = tensor_bf16(v.y);
    hi = (uint32_t) h0 | ((uint32_t) h1 << 16);
    lo = (uint32_t) tensor_bf16(v.x - tensor_float(h0)) | ((uint32_t) tensor_bf16(v.y - tensor_float(h1)) << 16);
}
template<bool local_rows,bool decoded_rows,bool packed_prob>
__device__ void tensor_attention_stage(const float* q,PackedRows window,PackedRows compressed,const int32_t* sel,
        int sel_stride,AttentionRows ar,const float* sink,int heads,float scale,float* out,int group,int token,const uint16_t* decoded,const uint16_t* decoded_window) {

    constexpr int d = 512, R = 16, dp = d + 8;
    const int hb=group*R,t=token;
    const int n_win=ar.window_count[t],n_sel=ar.selected_count[t],w0=ar.window_start[t];
    const int* sl=sel?sel+size_t(t)*sel_stride:nullptr;
    __shared__ __align__(16) uint16_t tile[R * dp];
    __shared__ float sp[threads / 32][R * R];
    union ProbTile {
        float fp32[R*R];
        struct {uint32_t hi[R*R/2],lo[R*R/2];} packed;
    };
    __shared__ __align__(16) ProbTile pr;
    __shared__ int rows[R];
    __shared__ bool valid[R];
    __shared__ float m_run[R], den[R], tsc[R];
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5, g = lane >> 2, c2 = (lane & 3) * 2;
    uint32_t qa[4][4], qb[4][4];
    {
        const float* q0 = q + ((size_t) t * heads + hb + g) * d + wp * 64 + c2;
        const float* q1 = q0 + (size_t) 8 * d;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            split_bf16x2(*(const float2*) (q0 + kk * 16), qa[kk][0], qb[kk][0]);
            split_bf16x2(*(const float2*) (q1 + kk * 16), qa[kk][1], qb[kk][1]);
            split_bf16x2(*(const float2*) (q0 + kk * 16 + 8), qa[kk][2], qb[kk][2]);
            split_bf16x2(*(const float2*) (q1 + kk * 16 + 8), qa[kk][3], qb[kk][3]);
        }
    }
    if (threadIdx.x < R) {
        m_run[threadIdx.x] = -1e30f;
        den[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    float acc[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
    const int sr = threadIdx.x >> 4, part = threadIdx.x & 15;

    const uint16_t* kb = tile + ((lane & 7) + ((lane >> 4) << 3)) * dp + wp * 64 + ((lane >> 3) & 1) * 8;
    const uint16_t* vb = tile + (lane & 15) * dp + wp * 64 + (lane >> 4) * 8;
    const int total = n_win + n_sel;
    for (int t0 = 0; t0 < total; t0 += R) {
        const int n = min(R, total - t0);
        int row;
        bool live;
        if constexpr(local_rows) {
            // Each 16-thread unpack group owns one row. Publish validity for
            // softmax at the existing tile barrier, rather than before unpack.
            const int j=t0+sr;
            row=j<n_win?w0+j:j<total?sl[j-n_win]:-1;
            if(j<n_win && ar.window_capacity>0)row%=ar.window_capacity;
            live=j<total && row>=0;
            if(part==0)valid[sr]=live;
        } else {
        if(threadIdx.x<R) {
            const int j=t0+threadIdx.x;
            int row=j<n_win?w0+j:j<total?sl[j-n_win]:-1;
            if(j<n_win && ar.window_capacity>0)row%=ar.window_capacity;
            rows[threadIdx.x]=row;valid[threadIdx.x]=j<total && row>=0;
        }
        __syncthreads();
        row=rows[sr];live=valid[sr];
        }
        auto* dst=tile+sr*dp+part*32;
        const auto matrix=t0+sr<n_win?window:compressed;
        if(!live) {
            for(int j=0;j<4;++j)reinterpret_cast<uint4*>(dst)[j]=make_uint4(0,0,0,0);
        } else if(decoded_rows && (t0+sr<n_win?decoded_window:decoded)) {
            const auto* p=(t0+sr<n_win?decoded_window:decoded)+size_t(row)*d+part*32;
            #pragma unroll
            for(int j=0;j<4;++j)reinterpret_cast<uint4*>(dst)[j]=reinterpret_cast<const uint4*>(p)[j];
        } else if(matrix.format!=CacheFormat::window_fp8) {
            const uint8_t* p=matrix.data+size_t(row)*matrix.stride;
            const auto raw=*reinterpret_cast<const uint4*>(p+part*16);
            const uint32_t words[4]={raw.x,raw.y,raw.z,raw.w};
            const bool e4=matrix.format==CacheFormat::compressed_fp4;
            const float s0=e4?e4_value(p[256+part*2]):__uint_as_float(uint32_t(p[256+part])<<23);
            const float s1=e4?e4_value(p[256+part*2+1]):s0;
#pragma unroll
            for(int word=0;word<4;++word) {
                uint32_t packed[4];const float scale=word<2?s0:s1;
#pragma unroll
                for(int pair=0;pair<4;++pair) {
                    const uint32_t bits=words[word]>>(pair*8);
                    const uint16_t lo=tensor_bf16(e2_value(uint8_t(bits))*scale),hi=tensor_bf16(e2_value(uint8_t(bits>>4))*scale);
                    packed[pair]=uint32_t(lo)|(uint32_t(hi)<<16);
                }
                reinterpret_cast<uint4*>(dst)[word]=make_uint4(packed[0],packed[1],packed[2],packed[3]);
            }
        } else {
#pragma unroll
            for(int c=0;c<32;++c)dst[c]=tensor_bf16(row_value(matrix,row,part*32+c,d));
        }
        __syncthreads();
        float s[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            uint32_t b0, b1, b2, b3;
            mma_ldsm_x4(b0, b1, b2, b3, kb + kk * 16);
            mma_bf16(s[0], qa[kk], b0, b1);
            mma_bf16(s[0], qb[kk], b0, b1);
            mma_bf16(s[1], qa[kk], b2, b3);
            mma_bf16(s[1], qb[kk], b2, b3);
        }
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int r0 = nt * 8 + c2;
            sp[wp][g * R + r0] = s[nt][0];
            sp[wp][g * R + r0 + 1] = s[nt][1];
            sp[wp][(g + 8) * R + r0] = s[nt][2];
            sp[wp][(g + 8) * R + r0 + 1] = s[nt][3];
        }
        __syncthreads();
        {
            const int e = threadIdx.x, h = e >> 4, r = e & 15;
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < threads / 32; ++w) v += sp[w][e];
            v = r < n && valid[r] ? v * scale : -INFINITY;
            float mx = v;
#pragma unroll
            for (int k = 8; k >= 1; k >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, k));
            const float m_old = m_run[h], m_new = fmaxf(m_old, mx);
            const float p = r < n ? expf(v - m_new) : 0.0f;
            float sum = p;
#pragma unroll
            for (int k = 8; k >= 1; k >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, k);
            if constexpr(packed_prob) {
                // Every output warp consumes the same probability tile. Round each
                // adjacent pair once, using the original split, before publishing it.
                const float next=__shfl_xor_sync(0xffffffffu,p,1);
                if(!(r&1))split_bf16x2(make_float2(p,next),pr.packed.hi[e/2],pr.packed.lo[e/2]);
            } else pr.fp32[e]=p;
            if (r == 0) {
                const float ts = expf(m_old - m_new);
                tsc[h] = ts;
                den[h] = den[h] * ts + sum;
                m_run[h] = m_new;
            }
        }
        __syncthreads();
        uint32_t pa[4], pb[4];
        const int pbase=g*R+c2;
        constexpr int poff[4]={0,8*R,8,8*R+8};
#pragma unroll
        for(int j=0;j<4;++j) {
            const int i=pbase+poff[j];
            if constexpr(packed_prob) {pa[j]=pr.packed.hi[i/2];pb[j]=pr.packed.lo[i/2];}
            else split_bf16x2(*(const float2*)(pr.fp32+i),pa[j],pb[j]);
        }
        const float ts0 = tsc[g], ts1 = tsc[g + 8];
#pragma unroll
        for (int np = 0; np < 4; ++np) {
            uint32_t b0, b1, b2, b3;
            mma_ldsm_x4_t(b0, b1, b2, b3, vb + np * 16);
            float* a0 = acc[2 * np];
            float* a1 = acc[2 * np + 1];
            a0[0] *= ts0, a0[1] *= ts0, a0[2] *= ts1, a0[3] *= ts1;
            a1[0] *= ts0, a1[1] *= ts0, a1[2] *= ts1, a1[3] *= ts1;
            mma_bf16(a0, pa, b0, b1);
            mma_bf16(a0, pb, b0, b1);
            mma_bf16(a1, pa, b2, b3);
            mma_bf16(a1, pb, b2, b3);
        }
        __syncthreads();
    }
    const float dn0 = den[g] + expf(sink[hb + g] - m_run[g]);
    const float dn1 = den[g + 8] + expf(sink[hb + g + 8] - m_run[g + 8]);
    float* o0 = out + ((size_t) t * heads + hb + g) * d + wp * 64 + c2;
    float* o1 = o0 + (size_t) 8 * d;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        *(float2*) (o0 + nt * 8) = make_float2(acc[nt][0] / dn0, acc[nt][1] / dn0);
        *(float2*) (o1 + nt * 8) = make_float2(acc[nt][2] / dn1, acc[nt][3] / dn1);
    }
}

template<bool local_rows,bool decoded_rows,bool packed_prob=false>
__global__ void tensor_attention_kernel(const float* q,PackedRows w,PackedRows c,const int32_t* ids,int stride,
        AttentionRows a,const float* sinks,int heads,float scale,float* out,const uint16_t* decoded,const uint16_t* decoded_window) {
    tensor_attention_stage<local_rows,decoded_rows,packed_prob>(q,w,c,ids,stride,a,sinks,heads,scale,out,blockIdx.x,blockIdx.y,decoded,decoded_window);
}
