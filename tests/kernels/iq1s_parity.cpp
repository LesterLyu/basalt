// IQ1_S on the CPU and GPU against dequantized weights accumulated in double.
// The CPU check runs without a GPU; --bench adds a device row-dot timing.
#include "basalt/ggml/ggml.h"
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "basalt/ggml/ggml_common.h"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/iq_avx2.hpp"
#include "basalt/kernels/f16_bits.hpp"
#include "basalt/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = basalt::kernels;
static int failures = 0;

static void ck(cudaError_t e) {
    if (e == cudaSuccess) return;
    std::printf("CUDA error: %s\n", cudaGetErrorString(e));
    std::exit(2);
}
template<class T> static T* device(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)));
    return p;
}

static std::vector<block_iq1_s> weights(int n, std::mt19937& rng) {
    std::vector<block_iq1_s> w(n);
    for (auto& b : w) {
        b.d = k::f16_from_f32(std::ldexp((rng() & 1 ? -1.f : 1.f) * (1.f + (rng() % 16) / 16.f), -8));
        for (auto& v : b.qs) v = uint8_t(rng());
        for (auto& v : b.qh) v = uint16_t(rng());
    }
    return w;
}

static void cpu_rows_check(std::mt19937& rng, bool bench) {
    constexpr int rows=9, tokens=12;
    for(int n:{2304,5120}) {
        auto w=weights(rows*n/256,rng);
        std::vector<block_q8_K> q(tokens*n/256);
        std::vector<const void*> inputs(tokens);
        std::vector<std::vector<float>> out(tokens,std::vector<float>(rows));
        std::vector<float*> outputs(tokens);
        std::vector<float> x(n),wf(rows*n);
        ggml_get_type_traits(GGML_TYPE_IQ1_S)->to_float(w.data(),wf.data(),rows*n);
        std::normal_distribution<float> nd;
        const auto* dot=ggml_get_type_traits_cpu(GGML_TYPE_IQ1_S);
        const auto* quant=ggml_get_type_traits_cpu(dot->vec_dot_type);
        for(int t=0;t<tokens;++t) {
            for(auto& value:x)value=t==tokens-1?0.f:nd(rng);
            inputs[t]=q.data()+t*n/256;outputs[t]=out[t].data();
            quant->from_float(x.data(),q.data()+t*n/256,n);
        }
        const size_t stride=n/256*sizeof(block_iq1_s);
        double max_abs=0,max_scaled=0;
        for(int width:{1,2,3,5,6,11,12}) {
            k::cpu::iq1s256_rows(reinterpret_cast<const uint8_t*>(w.data()),stride,n,inputs.data(),width,outputs.data(),0,rows);
            for(int t=0;t<width;++t)for(int row=0;row<rows;++row) {
                float old=0;
                dot->vec_dot(n,&old,0,w.data()+row*n/256,0,inputs[t],0,1);
                failures+=std::memcmp(&old,&out[t][row],sizeof(float))!=0;
                double ref=0,magnitude=0;
                for(int i=0;i<n;++i) {
                    const auto& b=q[t*n/256+i/256];
                    const double term=double(wf[row*n+i])*b.d*b.qs[i%256];
                    ref+=term;magnitude+=std::abs(term);
                }
                max_abs=std::max(max_abs,std::abs(out[t][row]-ref));
                max_scaled=std::max(max_scaled,std::abs(out[t][row]-ref)/(magnitude+1e-30));
            }
        }
        failures+=max_scaled>2e-6;
        std::printf("IQ1_S CPU rows n=%d: windows 1..12 ggml bit-identical, double max abs %.3g scaled %.3g\n",n,max_abs,max_scaled);
        if(bench) {
          for(int width:{1,2,6}) {
            for(int type=0;type<2;++type) {
                const auto run=[&] {
                    if(type)k::cpu::iq1s256_rows(reinterpret_cast<const uint8_t*>(w.data()),stride,n,inputs.data(),width,outputs.data(),0,rows);
                    else for(int t=0;t<width;++t)for(int r=0;r<rows;++r)dot->vec_dot(n,&out[t][r],0,w.data()+r*n/256,0,inputs[t],0,1);
                };
                for(int repeat=0;repeat<32;++repeat)run();
                const auto start=std::chrono::steady_clock::now();
                for(int repeat=0;repeat<1024;++repeat)run();
                const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/1024;
                std::printf("IQ1_S CPU %d x %d T%d %s: %.3f us, %.3f weight GB/s\n",rows,n,width,type?"block-sum":"ggml",us,rows*stride/us/1000.);
            }
          }
        }
    }
}
static void cpu_check(std::mt19937& rng) {
    const auto* dq = ggml_get_type_traits(GGML_TYPE_IQ1_S);
    if (!dq || !dq->to_float || ggml_row_size(GGML_TYPE_IQ1_S, 5120) != 1000) {
        std::printf("IQ1_S: missing CPU format\n"); ++failures; return;
    }
    for (int n : {256, 2304, 5120}) {
        auto w = weights(n / 256, rng);
        std::vector<float> wf(n), x(n);
        dq->to_float(w.data(), wf.data(), n);
        std::normal_distribution<float> nd;
        for (auto& v : x) v = nd(rng);
        double max_abs = 0, max_rel = 0;
        float first = 0;
        bool have_first = false;
        for (int isa : {2, 3}) {
            if (isa == 3 && !k::cpu::cpu_avx512_ok()) continue;
            const auto* t = isa == 2 ? ggml_get_type_traits_cpu_avx2(GGML_TYPE_IQ1_S)
                                     : ggml_get_type_traits_cpu_avx512(GGML_TYPE_IQ1_S);
            const auto* a = ggml_get_type_traits_cpu(t->vec_dot_type);
            std::vector<block_q8_K> q(n / 256);
            a->from_float(x.data(), q.data(), n);
            double ref = 0, magnitude = 0;
            for (int i = 0; i < n; ++i) {
                const auto& b = q[i / 256];
                const double term = double(wf[i]) * b.d * b.qs[i % 256];
                ref += term; magnitude += std::abs(term);
            }
            float got = 0;
            t->vec_dot(n, &got, 0, w.data(), 0, q.data(), 0, 1);
            const double abs = std::abs(got - ref), rel = abs / (std::abs(ref) + 1e-30);
            max_abs = std::max(max_abs, abs); max_rel = std::max(max_rel, rel);
            failures += !std::isfinite(got) || abs > 2e-6 * magnitude;
            if (have_first) failures += std::memcmp(&first, &got, sizeof got) != 0;
            first = got; have_first = true;
            for (auto& b : q) b.d = 0.f;
            t->vec_dot(n, &got, 0, w.data(), 0, q.data(), 0, 1);
            failures += got != 0.f;
        }
        std::printf("IQ1_S CPU n=%d: max abs %.3g rel %.3g%s\n", n, max_abs, max_rel,
                    k::cpu::cpu_avx512_ok() ? ", AVX2/AVX512 identical" : ", AVX2");
    }
}

static void gpu_check(std::mt19937& rng, bool bench) {
    cudaStream_t s;
    ck(cudaStreamCreate(&s));
    // Every codebook entry, both offsets and every sub-block scale.
    auto w = weights(2048, rng);
    for (int i = 0; i < 2048; ++i) {
        for (int ib = 0; ib < 8; ++ib) {
            w[i].qh[ib] = uint16_t(((ib & 1) << 15) | ((ib & 7) << 12));
            for (int j = 0; j < 4; ++j) {
                w[i].qs[4 * ib + j] = uint8_t(i);
                w[i].qh[ib] |= uint16_t((i >> 8) << (3 * j));
            }
        }
    }
    auto* dw = device<block_iq1_s>(w.size());
    auto* df = device<float>(w.size() * 256);
    ck(cudaMemcpy(dw, w.data(), w.size() * sizeof(w[0]), cudaMemcpyHostToDevice));
    std::vector<float> ref(w.size() * 256), got(ref.size());
    ggml_get_type_traits(GGML_TYPE_IQ1_S)->to_float(w.data(), ref.data(), ref.size());
    k::iq_dequant_f32(19, dw, ref.size(), df, s);
    ck(cudaStreamSynchronize(s));
    ck(cudaMemcpy(got.data(), df, got.size() * 4, cudaMemcpyDeviceToHost));
    failures += std::memcmp(got.data(), ref.data(), got.size() * 4) != 0;
    std::printf("IQ1_S dequant: all 2048 grids, offsets and scales, CPU/GPU byte comparison\n");
    cudaFree(dw); cudaFree(df);

    for (int n : {2304, 5120}) {
        constexpr int rows = 67, cols = 11;
        auto m = weights(rows * n / 256, rng);
        std::vector<float> mf(rows * n), x(cols * n);
        ggml_get_type_traits(GGML_TYPE_IQ1_S)->to_float(m.data(), mf.data(), mf.size());
        std::normal_distribution<float> nd;
        for (auto& v : x) v = nd(rng);
        auto* dm = device<block_iq1_s>(m.size());
        auto* dx = device<float>(x.size());
        auto* q = device<uint8_t>(cols * (n / 32) * 36);
        auto* dy = device<float>(rows * cols);
        ck(cudaMemcpy(dm, m.data(), m.size() * sizeof(m[0]), cudaMemcpyHostToDevice));
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        k::quantize_q8_1_rows(dx, cols, n, q, s);
        ck(cudaStreamSynchronize(s));
        std::vector<uint8_t> hq(cols * (n / 32) * 36);
        ck(cudaMemcpy(hq.data(), q, hq.size(), cudaMemcpyDeviceToHost));
        double max_abs = 0, max_rel = 0, max_scaled = 0;
        std::vector<float> out(cols * rows), single(cols * rows);
        for (int c = 0; c < cols; ++c)
            k::iq_mmvq(19, dm, q + c * (n / 32) * 36, dy + c * rows, n, rows, 1, s);
        ck(cudaStreamSynchronize(s));
        ck(cudaMemcpy(single.data(), dy, single.size() * 4, cudaMemcpyDeviceToHost));
        for (int nc : {1, 2, 3, 5, 8, 11}) {
            for (bool old : {true, false}) {
                k::iq_set_old_kernels(old);
                k::iq_mmvq(19, dm, q, dy, n, rows, nc, s);
                ck(cudaStreamSynchronize(s));
                ck(cudaMemcpy(out.data(), dy, nc * rows * 4, cudaMemcpyDeviceToHost));
                failures += std::memcmp(out.data(), single.data(), nc * rows * 4) != 0;
                for (int c = 0; c < nc; ++c) for (int r = 0; r < rows; ++r) {
                    double expected = 0, magnitude = 0;
                    for (int i = 0; i < n; ++i) {
                        const uint8_t* b = hq.data() + (c * (n / 32) + i / 32) * 36;
                        uint16_t d; std::memcpy(&d, b, 2);
                        const double term = double(mf[r * n + i]) * k::f32_from_f16(d) * int8_t(b[4 + i % 32]);
                        expected += term; magnitude += std::abs(term);
                    }
                    const double abs = std::abs(out[c * rows + r] - expected);
                    failures += !std::isfinite(out[c * rows + r]);
                    max_abs = std::max(max_abs, abs);
                    max_rel = std::max(max_rel, abs / (std::abs(expected) + 1e-30));
                    max_scaled = std::max(max_scaled, abs / (magnitude + 1e-30));
                }
            }
        }
        failures += max_scaled > 2e-6;
        std::printf("IQ1_S GPU n=%d: max abs %.3g rel %.3g, windows/positions identical\n", n, max_abs, max_rel);
        k::iq_set_old_kernels(false);
        if (bench) {
            cudaEvent_t a, b;
            ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b));
            for (int i = 0; i < 10; ++i) k::iq_mmvq(19, dm, q, dy, n, rows, 3, s);
            ck(cudaEventRecord(a, s));
            for (int i = 0; i < 100; ++i) k::iq_mmvq(19, dm, q, dy, n, rows, 3, s);
            ck(cudaEventRecord(b, s)); ck(cudaEventSynchronize(b));
            float ms = 0; ck(cudaEventElapsedTime(&ms, a, b));
            std::printf("IQ1_S %dx%d nc=3: %.2f us/call, %.2f weight GB/s\n", rows, n, ms * 10, m.size() * sizeof(m[0]) / (ms * 1e4));
            cudaEventDestroy(a); cudaEventDestroy(b);
        }
        cudaFree(dm); cudaFree(dx); cudaFree(q); cudaFree(dy);
    }
    cudaStreamDestroy(s);
}

static int q8k_probe(std::mt19937& rng) {
    const auto* qtraits=ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    cudaStream_t stream{};ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    bool compatible=true;
    for(int n:{5120,2304}) {
        const int rows=n==5120?4608:5120;
        auto matrix=weights(rows*n/256,rng);
        const size_t bytes=matrix.size()*sizeof(block_iq1_s);
        std::vector<float> input(n),reference(rows),got(rows);
        std::vector<block_q8_K> activation(n/256);
        for(auto& value:input)value=float(int(rng()>>16)-32768)*0x1p-14f;
        qtraits->from_float(input.data(),activation.data(),n);
        const void* aq[]={activation.data()};float* output[]={reference.data()};
        k::cpu::iq1s256_rows(reinterpret_cast<const uint8_t*>(matrix.data()),size_t(n/256)*sizeof(block_iq1_s),
                            n,aq,1,output,0,rows);
        auto* w=device<block_iq1_s>(matrix.size());
        auto* x=device<block_q8_K>(activation.size());
        auto* y=device<float>(rows);
        void* mapped_host=nullptr,*mapped_device=nullptr;
        ck(cudaHostAlloc(&mapped_host,bytes,cudaHostAllocMapped));
        ck(cudaHostGetDevicePointer(&mapped_device,mapped_host,0));
        std::memcpy(mapped_host,matrix.data(),bytes);
        ck(cudaMemcpy(w,matrix.data(),bytes,cudaMemcpyHostToDevice));
        ck(cudaMemcpy(x,activation.data(),activation.size()*sizeof(block_q8_K),cudaMemcpyHostToDevice));
        int exact_mode=-1;
        for(int fused:{0,1}) {
            k::iq1s_q8k_rows(w,x,y,n,rows,fused!=0,stream);ck(cudaStreamSynchronize(stream));
            ck(cudaMemcpy(got.data(),y,got.size()*sizeof(float),cudaMemcpyDeviceToHost));
            int exact=0;double max_abs=0;
            for(int r=0;r<rows;++r) {
                exact+=std::memcmp(&reference[r],&got[r],sizeof(float))==0;
                max_abs=std::max(max_abs,std::abs(double(reference[r])-got[r]));
            }
            std::printf("IQ1_S/Q8_K %dx%d fused=%d exact=%d/%d max_abs=%.8g\n",rows,n,fused,exact,rows,max_abs);
            if(exact==rows)exact_mode=fused;
        }
        if(exact_mode>=0) {
            for(int mapped:{0,1}) {
                const auto launch=[&]{k::iq1s_q8k_rows(mapped?mapped_device:w,x,y,n,rows,exact_mode!=0,stream);};
                launch();ck(cudaStreamSynchronize(stream));
                ck(cudaMemcpy(got.data(),y,got.size()*sizeof(float),cudaMemcpyDeviceToHost));
                if(std::memcmp(got.data(),reference.data(),rows*sizeof(float))!=0)compatible=false;
                cudaEvent_t a{},b{};ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));ck(cudaEventRecord(a,stream));
                for(int i=0;i<40;++i)launch();
                ck(cudaEventRecord(b,stream));ck(cudaEventSynchronize(b));float ms=0;
                ck(cudaEventElapsedTime(&ms,a,b));
                std::printf("IQ1_S/Q8_K %dx%d mapped=%d %.3f us per matrix\n",rows,n,mapped,ms*1000.f/40);
                ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));
            }
        } else compatible=false;
        ck(cudaFreeHost(mapped_host));ck(cudaFree(w));ck(cudaFree(x));ck(cudaFree(y));
    }
    ck(cudaStreamDestroy(stream));
    return compatible?0:1;
}
int main(int argc, char** argv) {
    bool cpu_only = false, bench = false,q8k=false;
    int device_id = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--cpu") cpu_only = true;
        else if (arg == "--bench") bench = true;
        else if (arg == "--q8k") q8k = true;
        else if (arg == "--device" && i + 1 < argc) device_id = std::stoi(argv[++i]);
        else { std::printf("usage: iq1s_parity [--cpu] [--bench] [--device ID]\n"); return 2; }
    }
    std::mt19937 rng(82);
    if(q8k) {ck(cudaSetDevice(device_id));return q8k_probe(rng);}
    cpu_check(rng);
    auto rows_rng=rng;
    cpu_rows_check(rows_rng,bench);
    if (!cpu_only) {
        ck(cudaSetDevice(device_id));
        cudaDeviceProp prop{};
        ck(cudaGetDeviceProperties(&prop, device_id));
        std::printf("device %d: %s\n", device_id, prop.name);
        gpu_check(rng, bench);
    }
    std::printf("iq1s_parity: %d failures\n", failures);
    return failures ? 1 : 0;
}
