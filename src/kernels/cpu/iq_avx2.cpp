// src/kernels/cpu/iq_avx2.cpp - the i-quant expert rows in 256-bit lanes, several tokens at once (iq_avx2_rows.inc),
// the activation quantizer, and the choice between this file's rows and the AVX-VNNI build of the same code.

#include "basalt/kernels/cpu/iq_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "basalt/ggml/ggml_common.h"
#include "basalt/log.hpp"
#include "basalt/platform/compiler.hpp"

#include <immintrin.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "basalt/kernels/cpu/expert_layout.hpp"

#define BASALT_IQ_VNNI 0
#define BASALT_IQ_BMI2 0
#define BASALT_IQ_EXPORT(name) name##_base
#include "iq_avx2_rows.inc"

namespace basalt::kernels::cpu {

namespace {
int iq1s_sum(__m256i v) {
    const __m128i s=_mm_add_epi32(_mm256_castsi256_si128(v),_mm256_extracti128_si256(v,1));
    const __m128i h=_mm_add_epi32(s,_mm_unpackhi_epi64(s,s));
    return _mm_cvtsi128_si32(_mm_add_epi32(h,_mm_shuffle_epi32(h,_MM_SHUFFLE(2,3,0,1))));
}
template<int NT>
void iq1s_dot(const uint8_t* raw,int n,const void* const* act,float* output) {
    const auto* x=reinterpret_cast<const block_iq1_s*>(raw);
    float sums[NT]{};
    // Pack the high index bits with AVX2 shuffles, as in Maya's IQ1_S row decoder.
    const __m256i shuffle=_mm256_setr_epi16(0x0100,0x0100,0x0100,0x0100,0x0302,0x0302,0x0302,0x0302,
                                           0x0504,0x0504,0x0504,0x0504,0x0706,0x0706,0x0706,0x0706);
    const __m256i shifts=_mm256_setr_epi16(512,64,8,1,512,64,8,1,512,64,8,1,512,64,8,1);
    const auto indices=[&](__m256i header,const uint8_t* q,__m256i lanes) {
        const auto hi=_mm256_mullo_epi16(_mm256_shuffle_epi8(header,lanes),shifts);
        return _mm256_or_si256(_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q))),
                              _mm256_srli_epi16(_mm256_and_si256(hi,_mm256_set1_epi16(0xe00)),1));
    };
    for(int b=0;b<n/256;++b) {
        const auto header=_mm_loadu_si128(reinterpret_cast<const __m128i*>(x[b].qh));
        const auto repeated=_mm256_broadcastsi128_si256(header);
        alignas(32) uint16_t index[32];
        _mm256_store_si256(reinterpret_cast<__m256i*>(index),indices(repeated,x[b].qs,shuffle));
        _mm256_store_si256(reinterpret_cast<__m256i*>(index+16),
                           indices(repeated,x[b].qs+16,_mm256_add_epi8(shuffle,_mm256_set1_epi8(8))));
        const auto scale16=_mm_or_si128(_mm_and_si128(_mm_srli_epi16(header,11),_mm_set1_epi16(14)),_mm_set1_epi16(1));
        const auto signed_scale=_mm_sign_epi16(scale16,_mm_or_si128(header,_mm_set1_epi16(1)));
        __m256i products[NT];
        for(int t=0;t<NT;++t)products[t]=_mm256_setzero_si256();
        for(int ib=0;ib<8;++ib) {
            const uint16_t h=x[b].qh[ib];const uint16_t* q=index+4*ib;
            const __m256i grid=_mm256_set_epi64x(iq1s_grid[q[3]],iq1s_grid[q[2]],iq1s_grid[q[1]],iq1s_grid[q[0]]);
            const __m256i magnitude=_mm256_sign_epi8(grid,grid);
            const int scale=2*((h>>12)&7)+1;
            const __m256i scales=_mm256_set1_epi16((short)scale);
            for(int t=0;t<NT;++t) {
                const auto& y=reinterpret_cast<const block_q8_K*>(act[t])[b];
                if(y.d==0)continue;
                const __m256i values=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(y.qs+32*ib));
                const __m256i pairs=_mm256_maddubs_epi16(magnitude,_mm256_sign_epi8(values,grid));
                products[t]=_mm256_add_epi32(products[t],_mm256_madd_epi16(pairs,scales));
            }
        }
        for(int t=0;t<NT;++t) {
            const auto& y=reinterpret_cast<const block_q8_K*>(act[t])[b];
            if(y.d!=0) {
                const auto* bs=reinterpret_cast<const __m128i*>(y.bsums);
                const auto delta=_mm_madd_epi16(_mm_hadd_epi16(_mm_loadu_si128(bs),_mm_loadu_si128(bs+1)),signed_scale);
                const auto pair=_mm_add_epi32(delta,_mm_unpackhi_epi64(delta,delta));
                const int offset=_mm_cvtsi128_si32(_mm_add_epi32(pair,_mm_shuffle_epi32(pair,_MM_SHUFFLE(2,3,0,1))));
                const int sum=8*iq1s_sum(products[t])+offset;
                sums[t]+=(h2f(x[b].d)*y.d*IQ1S_DELTA)*sum;
            }
        }
    }
    for(int t=0;t<NT;++t)output[t]=sums[t];
}
void iq1s_window(const uint8_t* row,int n,const void* const* act,int nt,float* output) {
    switch(nt) {
#define IQ1S_NT(N) case N:iq1s_dot<N>(row,n,act,output);break;
        IQ1S_NT(1) IQ1S_NT(2) IQ1S_NT(3) IQ1S_NT(4) IQ1S_NT(5) IQ1S_NT(6)
        IQ1S_NT(7) IQ1S_NT(8) IQ1S_NT(9) IQ1S_NT(10) IQ1S_NT(11)
#undef IQ1S_NT
    }
}
}
void iq1s256_gu_rows(const uint8_t* blob,size_t stride,size_t up,int n,const void* const* act,int nt,
                      float* const* ff,int r0,int r1,float limit) {
    for(int first=0;first<nt;first+=11) {
        const int count=std::min(11,nt-first);
        for(int row=r0;row<r1;++row) {
            float gates[11],ups[11];
            iq1s_window(blob+size_t(row)*stride,n,act+first,count,gates);
            iq1s_window(blob+up+size_t(row)*stride,n,act+first,count,ups);
            for(int t=0;t<count;++t) {
                float g=gates[t],u=ups[t];
                if(limit>0) {g=std::fmin(g,limit);u=std::fmin(std::fmax(u,-limit),limit);}
                ff[first+t][row]=(g/(1.f+std::exp(-g)))*u;
            }
        }
    }
}
void iq1s256_rows(const uint8_t* weights,size_t stride,int n,const void* const* act,int nt,float* const* out,int r0,int r1) {
    for(int first=0;first<nt;first+=11) {
        const int count=std::min(11,nt-first);
        for(int row=r0;row<r1;++row) {
            float values[11];iq1s_window(weights+size_t(row)*stride,n,act+first,count,values);
            for(int t=0;t<count;++t)out[first+t][row]=values[t];
        }
    }
}

// ggml's quantize_row_q8_K, bit for bit: the signed value of the first element of largest magnitude fixes the
// scale, codes are rounded to nearest-even (ggml's magic-number rounding) and capped at 127, and each 16 codes
// are summed.  An all-zero block writes d = 0 and zero codes and sums.
void quantize_q8_K_avx2(const float* x, void* dst, int64_t k) {
    block_q8_K* y = (block_q8_K*) dst;
    const int64_t nb = k / QK_K;
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256i top = _mm256_set1_epi32(127), ones = _mm256_set1_epi16(1);
    for (int64_t i = 0; i < nb; ++i, x += QK_K) {
        __m256 v[QK_K / 8];
        __m256 am = _mm256_setzero_ps();
        for (int j = 0; j < QK_K / 8; ++j) {
            v[j] = _mm256_loadu_ps(x + 8 * j);
            am = _mm256_max_ps(am, _mm256_and_ps(v[j], absmask));
        }
        __m128 h = _mm_max_ps(_mm256_castps256_ps128(am), _mm256_extractf128_ps(am, 1));
        h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        const float amax = _mm_cvtss_f32(_mm_max_ss(h, _mm_movehdup_ps(h)));
        if (amax == 0.f) {
            y[i].d = 0;
            std::memset(y[i].qs, 0, QK_K);
            std::memset(y[i].bsums, 0, sizeof(y[i].bsums));
            continue;
        }
        const __m256 amv = _mm256_set1_ps(amax);
        float mx = 0.f;
        for (int j = 0; j < QK_K / 8; ++j) {
            const int m = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_and_ps(v[j], absmask), amv, _CMP_EQ_OQ));
            if (m != 0) { mx = x[8 * j + lowest_bit((unsigned) m)]; break; }
        }
        const float iscale = -127.f / mx;
        const __m256 is = _mm256_set1_ps(iscale);
        for (int j = 0; j < QK_K / 16; ++j) {
            const __m256i q0 = _mm256_min_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(is, v[2 * j])), top);
            const __m256i q1 = _mm256_min_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(is, v[2 * j + 1])), top);
            // 16 codes in order as int16, then as bytes
            const __m256i p = _mm256_permute4x64_epi64(_mm256_packs_epi32(q0, q1), 0xD8);
            _mm_storeu_si128((__m128i*) (y[i].qs + 16 * j),
                             _mm_packs_epi16(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1)));
            const __m256i s = _mm256_madd_epi16(p, ones);
            __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
            s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
            s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
            y[i].bsums[j] = (int16_t) _mm_cvtsi128_si32(s4);
        }
        y[i].d = 1 / iscale;
    }
}

bool iq256_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22 || type == 23;
}

void iq256_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    static const bool vnni = cpu_avxvnni_ok();
    if (vnni) iq256_gu_rows_vnni(type, blob, gu_row, up_off, n, act, nt, ff, r0, r1);
    else iq256_gu_rows_base(type, blob, gu_row, up_off, n, act, nt, ff, r0, r1);
}

void iq256_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    static const bool vnni = cpu_avxvnni_ok();
    if (vnni) iq256_rows_vnni(type, w, row_bytes, n, act, nt, out, r0, r1);
    else iq256_rows_base(type, w, row_bytes, n, act, nt, out, r0, r1);
}

bool iq4nl256_fits(int n) noexcept { return n > 0 && n % QK4_NL == 0 && n / QK4_NL <= iq4nl_max_blocks; }

void iq4nl256_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out,
                        int r0, int r1) {
    static const bool vnni = cpu_avxvnni_ok();
    if (vnni) iq4nl256_down_rows_vnni(w, row_bytes, n, hq, nt, out, r0, r1);
    else iq4nl256_down_rows_base(w, row_bytes, n, hq, nt, out, r0, r1);
}

}  // namespace basalt::kernels::cpu
