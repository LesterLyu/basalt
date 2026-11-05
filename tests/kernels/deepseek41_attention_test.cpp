#include "basalt/deepseek41/attention.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/deepseek41/execution_ops.hpp"
#include "basalt/deepseek41/packed_weights.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "../../src/deepseek41/prefill_index.hpp"
#include "../../src/deepseek41/decode_attention.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ds = basalt::deepseek41;
static int failures = 0;
static void ck(cudaError_t error) { if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); }
static void check(bool ok, const char* message) { if (!ok) { ++failures; std::printf("FAIL: %s\n", message); } }
template<class T> class Device {
public:
    explicit Device(size_t count) : n(count) { ck(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
    explicit Device(const std::vector<T>& data) : Device(data.size()) { put(data); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    void put(const std::vector<T>& data) { check(data.size() <= n, "upload size"); ck(cudaMemcpy(p, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get() const { std::vector<T> data(n); ck(cudaMemcpy(data.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return data; }
    T* p = nullptr;
    size_t n;
};
static double e4(unsigned c) { const double a = (c & 0x78) ? std::ldexp(1. + (c & 7) / 8., int((c >> 3) & 15) - 7) : (c & 7) / 512.; return c & 128 ? -a : a; }
static double e2(unsigned c) { const double v[] = {0, .5, 1, 1.5, 2, 3, 4, 6}; return c & 8 ? -v[c & 7] : v[c & 7]; }
static uint8_t nearest(float value, bool fp8) {
    unsigned best = 0;
    double distance = std::numeric_limits<double>::infinity();
    for (unsigned c = 0; c < (fp8 ? 127u : 8u); ++c) {
        const double d = std::abs(std::abs(double(value)) - (fp8 ? e4(c) : e2(c)));
        if (d < distance || (d == distance && !(c & 1))) { distance = d; best = c; }
    }
    return uint8_t(best | (std::signbit(value) ? (fp8 ? 128 : 8) : 0));
}
static std::vector<uint8_t> pack(const std::vector<float>& x, int width, ds::CacheFormat format) {
    const bool fp8 = format == ds::CacheFormat::window_fp8, sf8 = format == ds::CacheFormat::compressed_fp4;
    const int group = sf8 ? 16 : 32, bytes = int(ds::cache_row_bytes(width, format));
    std::vector<uint8_t> out(x.size() / width * bytes);
    for (size_t r = 0; r < x.size() / width; ++r) for (int g = 0; g < width / group; ++g) {
        float maximum = 0;
        for (int j = 0; j < group; ++j) maximum = std::max(maximum, std::abs(x[r * width + g * group + j]));
        const float base = std::max(maximum, fp8 ? 1e-4f : sf8 ? 6.f / 512 : 6.f * std::numeric_limits<float>::min()) / (fp8 ? 448.f : 6.f);
        const float scale = sf8 ? float(e4(nearest(base, true))) : std::exp2(std::ceil(std::log2(double(base))));
        uint32_t bits; std::memcpy(&bits, &scale, 4);
        out[r * bytes + (fp8 ? width : width / 2) + g] = sf8 ? nearest(scale, true) : uint8_t(bits >> 23);
        for (int j = 0; j < group; ++j) {
            const int c = g * group + j;
            const uint8_t code = nearest(x[r * width + c] / scale, fp8);
            if (fp8) out[r * bytes + c] = code;
            else out[r * bytes + c / 2] |= uint8_t(code << (4 * (c & 1)));
        }
    }
    return out;
}
static double value(const std::vector<uint8_t>& x, int row, int c, int width, ds::CacheFormat format) {
    const auto* p = x.data() + size_t(row) * ds::cache_row_bytes(width, format);
    const bool fp8 = format == ds::CacheFormat::window_fp8, sf8 = format == ds::CacheFormat::compressed_fp4;
    const unsigned s = p[(fp8 ? width : width / 2) + c / (sf8 ? 16 : 32)];
    return (fp8 ? e4(p[c]) : e2((p[c / 2] >> (4 * (c & 1))) & 15)) * (sf8 ? e4(s) : std::exp2(int(s) - 127));
}
static std::vector<float> random(size_t n, std::mt19937& rng) {
    std::normal_distribution<float> normal;
    std::vector<float> out(n); for (auto& v : out) v = normal(rng); return out;
}
static void compare(const std::vector<float>& actual, const std::vector<double>& wanted, double tolerance, const char* name) {
    double absolute = 0, largest = 0, error = 0, magnitude = 0;
    for (size_t i = 0; i < wanted.size(); ++i) {
        check(std::isfinite(actual[i]), "finite output");
        absolute = std::max(absolute, std::abs(double(actual[i]) - wanted[i]));
        largest = std::max(largest, std::abs(wanted[i]));
        error += std::abs(actual[i] - wanted[i]); magnitude += std::abs(wanted[i]);
    }
    std::printf("%s: max abs %.3g L1 rel %.3g\n", name, absolute, error / std::max(magnitude, 1e-30));
    check(absolute <= tolerance * std::max(1., largest), name);
}
static void bench(const char* name, size_t bytes, const std::function<void()>& fn) {
    cudaEvent_t first, last; ck(cudaEventCreate(&first)); ck(cudaEventCreate(&last));
    for (int i = 0; i < 5; ++i) fn();
    ck(cudaEventRecord(first)); for (int i = 0; i < 100; ++i) fn(); ck(cudaEventRecord(last)); ck(cudaEventSynchronize(last));
    float ms; ck(cudaEventElapsedTime(&ms, first, last));
    std::printf("micro %s: %.3f us, %.2f logical GB/s\n", name, ms * 10, double(bytes) / (ms * 10 * 1e3));
    cudaEventDestroy(first); cudaEventDestroy(last);
}
static void draft_picking(std::mt19937& rng) {
    for(int n:{1,17,255,256,257,1023,1024,1025,129280})for(int trial=0;trial<7;++trial) {
        auto values=random(n,rng);
        if(trial==1)std::fill(values.begin(),values.end(),-0.f);
        if(trial==2) {values[0]=7;values[n-1]=7;}
        if(trial==3)values[n/2]=INFINITY;
        if(trial==4)values[n/3]=std::numeric_limits<float>::quiet_NaN();
        if(trial==5)std::fill(values.begin(),values.end(),-INFINITY);
        if(trial==6)values[n-1]=-std::numeric_limits<float>::max();
        int wanted=-1,bad=0;
        for(int i=0;i<n;++i)if(!std::isfinite(values[i]))bad=1;
            else if(wanted<0 || values[i]>values[wanted])wanted=i;
        Device<float> input(values);Device<int32_t> result(std::vector<int32_t>{77,99,0,88}),old(1);
        Device<uint8_t> work(ds::finite_argmax_workspace_bytes(n));
        ds::finite_argmax(input.p,n,result.p+1,result.p+2,work.p,work.n,nullptr);
        const auto got=result.get();
        check(got==std::vector<int32_t>({77,wanted,bad,88}),"finite argmax CPU reference, ties, nonfinite flag and canaries");
        ds::finite_argmax(input.p,n,result.p+1,result.p+2,work.p,work.n,nullptr);
        check(result.get()==got,"finite argmax retains a previous error flag");
        if(!bad) {
            auto sticky=got;sticky[2]=1;result.put(sticky);
            ds::finite_argmax(input.p,n,result.p+1,result.p+2,work.p,work.n,nullptr);
            check(result.get()==sticky,"finite rows cannot clear an earlier nonfinite flag");
            basalt::kernels::SamplerParams params;params.greedy=true;
            basalt::kernels::sample_tokens(input.p,1,n,nullptr,0,params,old.p,nullptr);
            check(old.get()[0]==wanted,"finite argmax matches existing GPU greedy pick");
        }
        if(n==129280 && trial==0)bench("finite draft argmax",size_t(n)*4,[&]{
            ds::finite_argmax(input.p,n,result.p+1,result.p+2,work.p,work.n,nullptr);
        });
    }
    for(int width:{1,31,256,257}) {
        const int rows=129280;std::vector<uint16_t> table(size_t(rows)*width);
        for(auto& x:table)x=uint16_t((rng()&0x807f)|((120+rng()%15)<<7));
        Device<uint16_t> packed(table);Device<int32_t> index(1);Device<float> output(width+2),legacy(width);
        for(int row:{-1,0,rows-1,rows}) {
            index.put({row});output.put(std::vector<float>(width+2,-77));
            ds::gather_bf16_row(packed.p,index.p,rows,width,output.p+1,nullptr);const auto got=output.get();
            check(got.front()==-77 && got.back()==-77,"BF16 gather canaries");
            for(int c=0;c<width;++c) {
                uint32_t bits=row>=0 && row<rows?uint32_t(table[size_t(row)*width+c])<<16:0;
                float expected;std::memcpy(&expected,&bits,4);
                check(std::memcmp(&expected,&got[c+1],4)==0,"BF16 gather exact CPU bits and invalid row bounds");
            }
            if(row>=0 && row<rows && width%32==0) {
                basalt::kernels::iq_dequant_f32(30,packed.p+size_t(row)*width,width,legacy.p,nullptr);
                const auto old=legacy.get();
                check(std::memcmp(old.data(),got.data()+1,width*4)==0,"BF16 gather matches current GPU decoder");
            }
        }
        if(width==256) {index.put({rows-1});bench("draft BF16 gather",size_t(width)*6,[&]{
            ds::gather_bf16_row(packed.p,index.p,rows,width,output.p+1,nullptr);
        });}
    }
    std::printf("Draft pick/gather: current GPU and CPU reference exact, nonfinite propagation and bounds passed\n");
}
static void finish(std::mt19937& rng) {
    for (int width : {32, 128, 512, 1280, 5120}) {
        constexpr int rows = 11;
        auto x = random(size_t(rows) * width, rng), norm = random(width, rng), freq = random(32, rng);
        for (auto& f : freq) f *= .001f;
        Device<float> dx(x), dw(norm), df(freq);
        ds::RowFinish finish; finish.norm = dw.p; finish.epsilon = 1e-20f;
        finish.rotary = std::min(width, 64); finish.frequencies = df.p; finish.first_position = 262143;
        finish.vectors_per_token = 1; finish.position_step = 2;
        ds::finish_rows(dx.p, rows, width, width, finish, nullptr, 0, nullptr);
        const auto normal = dx.get();
        std::vector<double> ref(x.size());
        for (int r = 0; r < rows; ++r) {
            double sq = 0; for (int c = 0; c < width; ++c) sq += double(x[r * width + c]) * x[r * width + c];
            const double inv = 1 / std::sqrt(sq / width + finish.epsilon);
            for (int c = 0; c < width; ++c) ref[r * width + c] = x[r * width + c] * inv * norm[c];
            for (int c = width - finish.rotary; c < width; c += 2) {
                const float angle = float(finish.first_position + r * 2) * freq[(c - width + finish.rotary) / 2];
                const double a = ref[r * width + c], b = ref[r * width + c + 1];
                ref[r * width + c] = a * std::cos(double(angle)) - b * std::sin(double(angle));
                ref[r * width + c + 1] = a * std::sin(double(angle)) + b * std::cos(double(angle));
            }
        }
        compare(normal, ref, 5e-6, "RMS/RoPE finish");
        for (auto format : {ds::CacheFormat::window_fp8, ds::CacheFormat::compressed_fp4, ds::CacheFormat::index_fp4}) {
            const size_t stride = ds::cache_row_bytes(width, format);
            Device<uint8_t> bytes(rows * stride); Device<float> back(x.size());
            dx.put(x); finish.quant = format;
            ds::finish_rows(dx.p, rows, width, width, finish, bytes.p, stride, nullptr);
            check(bytes.get() == pack(normal, width, format), "composed finish packed bytes equal independent CPU quantizer");
            check(dx.get() == x, "packing preserves source");
            ds::unpack_rows(bytes.p, rows, width, stride, format, back.p, nullptr);
            const auto packed = bytes.get(); std::vector<double> unpacked(x.size());
            for (int r = 0; r < rows; ++r) for (int c = 0; c < width; ++c) unpacked[r * width + c] = value(packed, r, c, width, format);
            compare(back.get(), unpacked, 0, "packed cache decode");
            dx.put(x); ds::finish_rows(dx.p, rows, width, width, finish, nullptr, 0, nullptr);
            check(back.get() == dx.get(), "packed and in-place roundings match");
            if (width == 512 && format == ds::CacheFormat::window_fp8) {
                bench("finish", x.size() * 4 + rows * stride, [&] { ds::finish_rows(dx.p, rows, width, width, finish, bytes.p, stride, nullptr); });
                bench("unpack", x.size() * 4 + rows * stride, [&] { ds::unpack_rows(bytes.p, rows, width, stride, format, back.p, nullptr); });
            }
        }
    }
}
static void compressor(std::mt19937& rng) {
    constexpr int width = 512;
    const auto norm = random(width, rng); Device<float> dw(norm), dx(width), dg(width), kh(width * 2), sh(width * 2), out(width);
    for (int ratio : {1, 2}) for (int pos = 0; pos < 7; ++pos) {
        const auto kv = random(width, rng), gate = random(width, rng);
        static std::vector<float> previous, previous_gate;
        dx.put(kv); dg.put(gate);
        const bool complete = ds::compress_step(dx.p, dg.p, kh.p, sh.p, dw.p, out.p, width, ratio, pos, 1e-20f, nullptr);
        check(complete == ((pos + 1) % ratio == 0), "compressor completes exactly at group boundaries");
        if (complete) {
            std::vector<double> pooled(width), wanted(width); double ss = 0;
            for (int c = 0; c < width; ++c) {
                const double a = ratio == 1 ? 0 : std::exp(double(previous_gate[c]) - std::max(previous_gate[c], gate[c]));
                const double b = ratio == 1 ? 1 : std::exp(double(gate[c]) - std::max(previous_gate[c], gate[c]));
                pooled[c] = (a * (ratio == 1 ? 0 : previous[c]) + b * kv[c]) / (a + b); ss += pooled[c] * pooled[c];
            }
            for (int c = 0; c < width; ++c) wanted[c] = pooled[c] / std::sqrt(ss / width + 1e-20) * norm[c];
            compare(out.get(), wanted, 5e-6, "compressed latent");
        }
        previous = kv; previous_gate = gate;
    }
    bench("compressor", 8 * width * 4, [&] { ds::compress_step(dx.p, dg.p, kh.p, sh.p, dw.p, out.p, width, 2, 1, 1e-20f, nullptr); });
}
static void engram(std::mt19937& rng, const std::vector<float>* real_weights = nullptr) {
    constexpr int dim = 5120, tokens = 5;
    auto h = random(tokens * 4 * dim, rng), kv = random(tokens * 5 * dim, rng);
    auto w = real_weights ? *real_weights : random(4 * dim, rng);
    Device<float> dh(h), dk(kv), dw(w);
    ds::engram_add(dh.p, dk.p, dw.p, dim, tokens, 1e-20f, nullptr);
    std::vector<double> expected(h.begin(), h.end());
    for (int t = 0; t < tokens; ++t) for (int s = 0; s < 4; ++s) {
        double hh = 0, kk = 0, dot = 0;
        for (int c = 0; c < dim; ++c) { const double a = h[(t * 4 + s) * dim + c], b = kv[(t * 5 + s) * dim + c]; hh += a*a; kk += b*b; dot += a*w[s*dim+c]*b; }
        dot *= 1 / std::sqrt(hh / dim + 1e-20) / std::sqrt(kk / dim + 1e-20) / std::sqrt(double(dim));
        const double gate = 1 / (1 + std::exp(-std::copysign(std::sqrt(std::max(std::abs(dot), 1e-6)), dot)));
        for (int c = 0; c < dim; ++c) expected[(t * 4 + s) * dim + c] += gate * kv[(t * 5 + 4) * dim + c];
    }
    const auto batch = dh.get(); compare(batch, expected, 5e-6, "Engram gate");
    dh.put(h);
    for (int t = 0; t < tokens; ++t) ds::engram_add(dh.p + t*4*dim, dk.p + t*5*dim, dw.p, dim, 1, 1e-20f, nullptr);
    check(dh.get() == batch, "Engram token positions match individual calls");
    bench("engram", size_t(tokens) * 20 * dim * 4, [&] { ds::engram_add(dh.p, dk.p, dw.p, dim, tokens, 1e-20f, nullptr); });
}
static void compressor_rows(std::mt19937& rng) {
    constexpr int width=512;
    const auto norm=random(width,rng),history=random(width*2,rng),old_scores=random(width*2,rng);
    Device<float> dn(norm),kh(history),sh(old_scores),actual_k(history),actual_s(old_scores);
    for(int ratio:{1,2})for(int position:{0,1,123})for(int tokens:{2,3,11,128,2048}) {
        const auto input=random(size_t(tokens)*width,rng),scores=random(size_t(tokens)*width,rng);
        Device<float> kv(input),gate(scores),reference(size_t(tokens)*width),out(size_t(tokens)*width);
        kh.put(history);sh.put(old_scores);actual_k.put(history);actual_s.put(old_scores);
        int count=0;
        for(int t=0;t<tokens;++t)if(ds::compress_step(kv.p+size_t(t)*width,gate.p+size_t(t)*width,kh.p,sh.p,dn.p,
                                                   reference.p+size_t(count)*width,width,ratio,position+t,1e-20f,nullptr))++count;
        const auto expected=reference.get();
        const int got=ds::compress_rows(kv.p,gate.p,actual_k.p,actual_s.p,dn.p,out.p,width,ratio,position,tokens,1e-20f,nullptr);
        const auto result=out.get();
        check(got==count,"batch compressor output count");
        check(std::equal(result.begin(),result.begin()+size_t(count)*width,expected.begin()),"batch compressor pooled bits match sequential steps");
        check(actual_k.get()==kh.get() && actual_s.get()==sh.get(),"batch compressor history matches sequential steps");
        if(ratio==2 && position==1 && tokens==2048)bench("compress 2048 rows",size_t(tokens)*width*8,[&]{
            ds::compress_rows(kv.p,gate.p,actual_k.p,actual_s.p,dn.p,out.p,width,ratio,position,tokens,1e-20f,nullptr);
        });
    }
}
static void attention(std::mt19937& rng,bool packed_prob_only=false,bool verify_fixed_only=false) {
    constexpr int dim = 512, heads = 64, nwin = 128, nc = 513, select = 512;
    const auto win = pack(random(nwin * dim, rng), dim, ds::CacheFormat::window_fp8);
    const auto comp = pack(random(nc * dim, rng), dim, ds::CacheFormat::compressed_fp4);
    const auto sinks = random(heads, rng); Device<float> sink(sinks); Device<uint8_t> dw(win), dc(comp);
    std::vector<double> window_values(nwin*dim),compressed_values(nc*dim);
    for(int r=0;r<nwin;++r)for(int col=0;col<dim;++col)window_values[r*dim+col]=value(win,r,col,dim,ds::CacheFormat::window_fp8);
    for(int r=0;r<nc;++r)for(int col=0;col<dim;++col)compressed_values[r*dim+col]=value(comp,r,col,dim,ds::CacheFormat::compressed_fp4);
    const auto widths=verify_fixed_only?std::vector<int>{2,3,6}:std::vector<int>{1,5,11,32,128};
    for (int tokens : widths) {
        const auto queries = random(size_t(tokens) * heads * dim, rng); Device<float> q(queries), output(queries.size());
        std::vector<int32_t> ids(tokens * select); ds::AttentionRows rows;
        rows.window_capacity = nwin;
        for (int t = 0; t < tokens; ++t) {
            rows.window_start[t] = (t * 17) % nwin; rows.window_count[t] = t ? nwin : 1;
            rows.selected_count[t] = t ? select : 0;
            for (int j = 0; j < select; ++j) ids[t * select + j] = (j + t) % 17 == 0 ? -1 : (j + t) % nc;
        }
        Device<int32_t> di(ids);
        ds::PackedRows w{dw.p, ds::cache_row_bytes(dim, ds::CacheFormat::window_fp8), ds::CacheFormat::window_fp8};
        ds::PackedRows c{dc.p, ds::cache_row_bytes(dim, ds::CacheFormat::compressed_fp4), ds::CacheFormat::compressed_fp4};
        if(verify_fixed_only) {
            Device<float> scratch(ds::kAttentionScratch);
            const auto run=[&](bool fixed) {
                ds::sparse_attention_predecoded(q.p,w,c,di.p,select,rows,sink.p,heads,dim,tokens,
                                               1/std::sqrt(float(dim)),output.p,scratch.p,nullptr,fixed);
            };
            for(int n:{0,128,512}) {
                for(int t=1;t<tokens;++t)rows.selected_count[t]=n;
                run(false);const auto ref=output.get();
                run(true);const auto actual=output.get();
                check(std::memcmp(ref.data(),actual.data(),ref.size()*sizeof(float))==0,
                      "fixed-width verification preserves attention bits across wrapped, invalid and mixed key counts");
                for(bool fixed:{false,true}) {
                    const auto name=std::string(fixed?"verify fixed ":"verify generic ")+std::to_string(tokens)+" rows, "+
                                    std::to_string(n+128)+" keys";
                    bench(name.c_str(),size_t(tokens)*heads*dim*2*sizeof(float),[&]{run(fixed);});
                }
            }
            continue;
        }
        if(packed_prob_only) {
            Device<uint16_t> window_bf16(nwin*dim),compressed_bf16(nc*dim);
            ds::decode_attention_bf16(w,0,nwin,window_bf16.p,nullptr);
            ds::decode_attention_bf16(c,0,nc,compressed_bf16.p,nullptr);
            const auto run=[&]{
                ds::sparse_attention_tensor(q.p,w,c,di.p,select,rows,sink.p,heads,dim,tokens,1/std::sqrt(float(dim)),
                                            output.p,nullptr,compressed_bf16.p,window_bf16.p);
            };
            auto& flag=basalt::core::diagnostics().exp["ds-attention-packed-prob"];
            flag=0;run();const auto reference=output.get();
            flag=1;run();const auto candidate=output.get();
            check(std::memcmp(reference.data(),candidate.data(),reference.size()*sizeof(float))==0,
                  "prepacked probability preserves attention bits");
            if(tokens==128)for(int enabled:{0,1,1,0}) {
                flag=enabled;bench(enabled?"packed probability":"FP32 probability",0,run);
            }
            flag=0;
            continue;
        }
        ds::sparse_attention(q.p, w, c, di.p, select, rows, sink.p, heads, dim, tokens, 1/std::sqrt(float(dim)), output.p, nullptr);
        const auto batch = output.get(); std::vector<double> expected(queries.size());
        Device<float> unpacked(ds::kAttentionScratch);
        Device<float> tensor_output(queries.size());
        ds::sparse_attention_tensor(q.p,w,c,di.p,select,rows,sink.p,heads,dim,tokens,1/std::sqrt(float(dim)),tensor_output.p,nullptr);
        const auto tensor_batch=tensor_output.get();
        ds::sparse_attention_predecoded(q.p,w,c,di.p,select,rows,sink.p,heads,dim,tokens,
                                       1/std::sqrt(float(dim)),output.p,unpacked.p,nullptr);
        check(output.get()==batch,"shared KV decode preserves all attention bytes");
        for (int t = 0; t < tokens; ++t) for (int h = 0; h < heads; ++h) {
            const int count = rows.window_count[t] + rows.selected_count[t];
            std::vector<double> logits(count); double maximum = sinks[h];
            auto row_value = [&](int i, int col) {
                if (i < rows.window_count[t]) return window_values[((rows.window_start[t]+i)%nwin)*dim+col];
                const int id = ids[t*select+i-rows.window_count[t]];
                return id < 0 ? 0. : compressed_values[id*dim+col];
            };
            for (int i = 0; i < count; ++i) {
                double dot = 0; for (int col = 0; col < dim; ++col) dot += queries[(size_t(t)*heads+h)*dim+col] * row_value(i,col);
                logits[i] = i >= rows.window_count[t] && ids[t*select+i-rows.window_count[t]] < 0 ? -INFINITY : dot / std::sqrt(float(dim));
                maximum = std::max(maximum, logits[i]);
            }
            double sum = std::exp(sinks[h] - maximum); for (auto& l : logits) { l = std::exp(l-maximum); sum += l; }
            for (int col = 0; col < dim; ++col) for (int i = 0; i < count; ++i)
                expected[(size_t(t)*heads+h)*dim+col] += logits[i] / sum * row_value(i,col);
        }
        compare(batch, expected, 5e-6, "latent sparse attention");
        compare(tensor_batch,expected,3e-5,"tensor latent sparse attention");
        for (int t = 0; t < tokens; ++t) {
            auto one = rows; one.window_start[0] = rows.window_start[t]; one.window_count[0] = rows.window_count[t]; one.selected_count[0] = rows.selected_count[t];
            ds::sparse_attention(q.p+size_t(t)*heads*dim, w, c, di.p+t*select, select, one, sink.p, heads, dim, 1,
                                 1/std::sqrt(float(dim)), output.p+size_t(t)*heads*dim, nullptr);
            ds::sparse_attention_tensor(q.p+size_t(t)*heads*dim,w,c,di.p+t*select,select,one,sink.p,heads,dim,1,
                                       1/std::sqrt(float(dim)),tensor_output.p+size_t(t)*heads*dim,nullptr);
        }
        check(output.get() == batch, "attention windows and token positions are byte-identical");
        check(tensor_output.get()==tensor_batch,"tensor attention token positions are byte-identical");
        if(tokens==5) {
            Device<uint8_t> workspace(32ull<<20);basalt::prefill::Gemm gemm;std::string error;
            check(gemm.init_external(nullptr,nullptr,0,workspace.p,workspace.n,error),"matrix decode GEMM setup");
            for(int t:{0,1,4}) {
                ds::AttentionRows one;one.window_start[0]=rows.window_start[t];one.window_capacity=nwin;
                one.window_count[0]=rows.window_count[t];one.selected_count[0]=rows.selected_count[t];
                ds::attention_matrix(gemm,q.p+size_t(t)*heads*dim,w,c,di.p+t*select,one,sink.p,
                                      1/std::sqrt(float(dim)),tensor_output.p,unpacked.p,nullptr);
                const auto got=tensor_output.get();
                const std::vector<float> value(got.begin(),got.begin()+heads*dim);
                const std::vector<double> ref(expected.begin()+size_t(t)*heads*dim,expected.begin()+size_t(t+1)*heads*dim);
                compare(value,ref,3e-5,"matrix decode attention CPU-double");
                basalt::core::diagnostics().exp["ds-attention-key-tiles"]=0;
                ds::sparse_attention_split(q.p+size_t(t)*heads*dim,w,c,di.p+t*select,select,one,sink.p,
                                           1/std::sqrt(float(dim)),output.p,unpacked.p,nullptr);
                const auto split=output.get();
                check(std::memcmp(split.data(),batch.data()+size_t(t)*heads*dim,heads*dim*sizeof(float))==0,
                      "split FP32 attention preserves output bits");
                basalt::core::diagnostics().exp["ds-attention-key-tiles"]=1;
                ds::sparse_attention_split(q.p+size_t(t)*heads*dim,w,c,di.p+t*select,select,one,sink.p,
                                           1/std::sqrt(float(dim)),output.p,unpacked.p,nullptr);
                const auto tiled=output.get();
                check(std::memcmp(split.data(),tiled.data(),heads*dim*sizeof(float))==0,"key-tiled attention preserves old GPU bits");
                compare(std::vector<float>(tiled.begin(),tiled.begin()+heads*dim),ref,3e-5,"key-tiled attention CPU-double");
                basalt::core::diagnostics().exp["ds-attention-key-tiles"]=0;
                if(t==1) {
                    for(int cap:{1,127,128,129,255,256,257,511,512,513,639,640}) {
                        auto bounded=one;bounded.window_count[0]=std::min(128,cap);bounded.selected_count[0]=cap-bounded.window_count[0];
                        const std::string name="decode-predecoded-"+std::to_string(cap),split_name="decode-split-"+std::to_string(cap);
                        bench(name.c_str(),size_t(heads)*dim*8+nwin*w.stride+select*c.stride,[&]{
                            ds::sparse_attention_predecoded(q.p+size_t(t)*heads*dim,w,c,di.p+t*select,select,bounded,sink.p,
                                                             heads,dim,1,1/std::sqrt(float(dim)),output.p,unpacked.p,nullptr);
                        });
                        const auto exact=output.get();
                        bench(split_name.c_str(),size_t(heads)*dim*8+nwin*w.stride+select*c.stride,[&]{
                            ds::sparse_attention_split(q.p+size_t(t)*heads*dim,w,c,di.p+t*select,select,bounded,sink.p,
                                                       1/std::sqrt(float(dim)),tensor_output.p,unpacked.p,nullptr);
                        });
                        const auto candidate=tensor_output.get();
                        check(std::memcmp(exact.data(),candidate.data(),heads*dim*4)==0,"bounded split preserves bytes");
                        basalt::core::diagnostics().exp["ds-attention-key-tiles"]=1;
                        const std::string tiled_name="decode-key-tiles-"+std::to_string(cap);
                        bench(tiled_name.c_str(),size_t(heads)*dim*8+nwin*w.stride+select*c.stride,[&]{
                            ds::sparse_attention_split(q.p+size_t(t)*heads*dim,w,c,di.p+t*select,select,bounded,sink.p,
                                                       1/std::sqrt(float(dim)),tensor_output.p,unpacked.p,nullptr);
                        });
                        const auto key_tiled=tensor_output.get();
                        check(std::memcmp(exact.data(),key_tiled.data(),heads*dim*4)==0,"bounded key-tiled attention preserves bytes");
                        basalt::core::diagnostics().exp["ds-attention-key-tiles"]=0;
                    }
                    bench("decode-predecoded-640",size_t(heads)*dim*8+nwin*w.stride+select*c.stride,[&]{
                        ds::sparse_attention_predecoded(q.p+size_t(t)*heads*dim,w,c,di.p+t*select,select,one,sink.p,
                                                         heads,dim,1,1/std::sqrt(float(dim)),output.p,unpacked.p,nullptr);
                    });
                    bench("decode-matrix-640",size_t(heads)*dim*8+nwin*w.stride+select*c.stride,[&]{
                        ds::attention_matrix(gemm,q.p+size_t(t)*heads*dim,w,c,di.p+t*select,one,sink.p,
                                               1/std::sqrt(float(dim)),tensor_output.p,unpacked.p,nullptr);
                    });
                    bench("decode-split-640",size_t(heads)*dim*8+nwin*w.stride+select*c.stride,[&]{
                        ds::sparse_attention_split(q.p+size_t(t)*heads*dim,w,c,di.p+t*select,select,one,sink.p,
                                                   1/std::sqrt(float(dim)),output.p,unpacked.p,nullptr);
                    });
                }
            }
        }
        if (tokens == 5 || tokens==128) {
        bench(tokens==5?"attention-5":"attention-128", size_t(tokens)*(heads*dim*8+nwin*w.stride+select*c.stride), [&] {
            ds::sparse_attention(q.p,w,c,di.p,select,rows,sink.p,heads,dim,tokens,1/std::sqrt(float(dim)),output.p,nullptr);
        });
        bench(tokens==5?"attention-predecoded-5":"attention-predecoded-128",size_t(tokens)*(heads*dim*8+nwin*w.stride+select*c.stride),[&]{
            ds::sparse_attention_predecoded(q.p,w,c,di.p,select,rows,sink.p,heads,dim,tokens,
                                          1/std::sqrt(float(dim)),output.p,unpacked.p,nullptr);
        });
        bench(tokens==5?"attention-tensor-5":"attention-tensor-128",size_t(tokens)*(heads*dim*8+nwin*w.stride+select*c.stride),[&]{
            ds::sparse_attention_tensor(q.p,w,c,di.p,select,rows,sink.p,heads,dim,tokens,1/std::sqrt(float(dim)),tensor_output.p,nullptr);
        });
        }
    }
}
static void draft_shared_attention(std::mt19937& rng) {
    constexpr int dim=512,heads=64,tokens=5,rows=133;
    const auto kv=pack(random(rows*dim,rng),dim,ds::CacheFormat::window_fp8);
    const auto q=random(tokens*heads*dim,rng);
    Device<uint8_t> device_kv(kv);
    Device<float> queries(q),sinks(random(heads,rng)),out(q.size()),scratch(ds::kDraftAttentionScratch);
    std::vector<int32_t> ids(tokens*tokens);
    for(int t=0;t<tokens;++t)for(int i=0;i<tokens;++i)ids[t*tokens+i]=i;
    Device<int32_t> selected(ids);
    const auto bytes=ds::cache_row_bytes(dim,ds::CacheFormat::window_fp8);
    const ds::PackedRows win{device_kv.p,bytes,ds::CacheFormat::window_fp8};
    const ds::PackedRows provisional{device_kv.p+128*bytes,bytes,ds::CacheFormat::window_fp8};
    const float scale=1/std::sqrt(float(dim));
    for(int pos:{0,1,5,123,127,128,384}) {
        const int first=std::max(0,128-pos),visible=128-first;
        ds::AttentionRows a;
        for(int t=0;t<tokens;++t) {a.window_start[t]=first;a.window_count[t]=visible;a.selected_count[t]=tokens;}
        const auto run=[&](bool shared) {
            if(shared)ds::sparse_attention_draft_shared(queries.p,win,first,visible+tokens,sinks.p,scale,out.p,scratch.p,nullptr);
            else ds::sparse_attention(queries.p,win,provisional,selected.p,tokens,a,sinks.p,heads,dim,tokens,scale,out.p,nullptr);
        };
        run(false);const auto ref=out.get();
        run(true);const auto got=out.get();
        check(std::memcmp(ref.data(),got.data(),q.size()*sizeof(float))==0,"draft shared KV attention bits");
        if(pos==128)for(int enabled:{0,1,1,0})
            bench(enabled?"draft one KV unpack":"draft repeated KV unpack",0,[&]{run(enabled!=0);});
    }
}
static void cache_codes() {
    constexpr int dim=512;
    std::vector<uint8_t> bytes(ds::cache_row_bytes(dim,ds::CacheFormat::window_fp8),127);
    for(int i=0;i<dim;++i)bytes[i]=uint8_t(i);
    Device<uint8_t> input(bytes);Device<float> output(dim);
    ds::unpack_rows(input.p,1,dim,bytes.size(),ds::CacheFormat::window_fp8,output.p,nullptr);
    const auto got=output.get();
    for(int i=0;i<dim;++i) {
        const float wanted=float(e4(uint8_t(i)));
        check(std::memcmp(&got[i],&wanted,4)==0,"all signed E4M3 code values match CPU bits");
    }
}
static void rollback_copies(std::mt19937& rng) {
    constexpr int layers=40,rows=128,stride=528,dim=512;
    constexpr size_t ring_bytes=rows*stride,extended_bytes=2*ring_bytes,history=2*dim,raw=rows*dim;
    std::vector<uint8_t> snapshots(layers*extended_bytes);
    for(auto& byte:snapshots)byte=uint8_t(rng());
    const auto saved=random(layers*history,rng), values=random(layers*raw,rng);
    const auto scores=random(layers*raw,rng), old_scores=random(layers*history,rng);
    Device<uint8_t> extended(snapshots),ring(layers*ring_bytes);
    Device<float> kv(layers*history),gate(layers*history),saved_kv(saved),saved_gate(old_scores);
    Device<float> raw_kv(values),raw_gate(scores),norm(std::vector<float>(dim,1.f)),pooled(dim);
    ds::WindowRollback window{};
    for(int i=0;i<layers;++i) {
        auto& l=window.layers[i];
        l.ring=ring.p+i*ring_bytes;l.extended=extended.p+i*extended_bytes;
        if(i%13==1) {
            l.kv_history=kv.p+i*history;l.score_history=gate.p+i*history;
            l.saved_kv=saved_kv.p+i*history;l.saved_score=saved_gate.p+i*history;
            l.raw_kv=raw_kv.p+i*raw;l.raw_gate=raw_gate.p+i*raw;
        }
    }
    for(int position:{0,1,123,127,128,255})for(int proposed:{1,6,11,128})for(int accepted:{0,proposed/2,proposed-1}) {
        std::vector<uint8_t> wanted(layers*ring_bytes),initial(wanted.size());
        auto expected_kv=saved,expected_gate=old_scores;
        kv.put(saved);gate.put(old_scores);
        for(int i=0;i<layers;++i) {
            for(int j=0;j<rows;++j) {
                const int dst=(position+j)%rows;
                const auto from=snapshots.begin()+i*extended_bytes+j*stride;
                std::copy_n(from,stride,initial.begin()+i*ring_bytes+dst*stride);
                std::copy_n(from+(j<accepted?ring_bytes:0),stride,wanted.begin()+i*ring_bytes+dst*stride);
            }
            if(i%13==1)for(int t=0;t<accepted;++t) {
                const size_t dst=i*history+((position+t)&1)*dim;
                std::copy_n(values.begin()+i*raw+t*dim,dim,expected_kv.begin()+dst);
                std::copy_n(scores.begin()+i*raw+t*dim,dim,expected_gate.begin()+dst);
            }
        }
        ring.put(initial);
        for(const auto& l:window.layers)ds::update_window_ring(l.ring,l.extended,position,proposed,stride,nullptr);
        ds::rollback_window(window,position,proposed,accepted,nullptr);
        check(ring.get()==wanted && kv.get()==expected_kv && gate.get()==expected_gate,
              "batched rollback matches CPU cache and compression history, including wrap and zero accept");
    }
    constexpr int position=127,accepted=2,proposed=5;
    bench("legacy rollback (40 layers)",layers*ring_bytes,[&] {
        for(const auto& l:window.layers) {
            const int first=position%rows;
            ck(cudaMemcpyAsync(l.ring+first*stride,l.extended,(rows-first)*stride,cudaMemcpyDeviceToDevice));
            ck(cudaMemcpyAsync(l.ring,l.extended+(rows-first)*stride,first*stride,cudaMemcpyDeviceToDevice));
            ds::update_window_ring(l.ring,l.extended,position,accepted,stride,nullptr);
            if(l.kv_history) {
                ck(cudaMemcpyAsync(l.kv_history,l.saved_kv,history*sizeof(float),cudaMemcpyDeviceToDevice));
                ck(cudaMemcpyAsync(l.score_history,l.saved_score,history*sizeof(float),cudaMemcpyDeviceToDevice));
                for(int t=0;t<accepted;++t)ds::compress_step(l.raw_kv+t*dim,l.raw_gate+t*dim,l.kv_history,l.score_history,
                    norm.p,pooled.p,dim,2,position+t,1e-20f,nullptr);
            }
        }
    });
    bench("batched rollback (40 layers)",layers*(proposed-accepted)*stride,[&] {
        ds::rollback_window(window,position,proposed,accepted,nullptr);
    });
}
static void ring_overlay(std::mt19937& rng) {
    constexpr int width=512,heads=64,nwin=128,stride=528,ns=512;
    const auto ring_bytes=pack(random(nwin*width,rng),width,ds::CacheFormat::window_fp8);
    const auto appended=pack(random(6*width,rng),width,ds::CacheFormat::window_fp8);
    const auto compressed=pack(random(ns*width,rng),width,ds::CacheFormat::compressed_fp4);
    Device<uint8_t> ring(ring_bytes),actual((nwin+6)*stride),rotated((nwin+6)*stride),keys(compressed);
    Device<float> q(random(6*heads*width,rng)),sink(random(heads,rng)),out(6*heads*width),scratch(ds::kAttentionScratch);
    std::vector<int32_t> selected(6*ns);
    for(int t=0;t<6;++t)for(int i=0;i<ns;++i)selected[t*ns+i]=i%17==0?-1:(i+t)%ns;
    Device<int32_t> ids(selected);
    const ds::PackedRows c{keys.p,288,ds::CacheFormat::compressed_fp4};
    for(int pos:{0,1,123,127,128,259})for(int tokens:{2,3,6}) {
        std::vector<uint8_t> linear((nwin+6)*stride),overlay(linear.size(),0xcd),wanted=ring_bytes;
        const int start=pos%nwin;
        for(int i=0;i<nwin;++i)std::copy_n(ring_bytes.begin()+((start+i)%nwin)*stride,stride,linear.begin()+i*stride);
        std::copy(appended.begin(),appended.end(),linear.begin()+nwin*stride);
        std::copy(appended.begin(),appended.end(),overlay.begin()+nwin*stride);
        rotated.put(linear);actual.put(overlay);ring.put(ring_bytes);
        ds::AttentionRows rows;
        for(int t=0;t<tokens;++t) {
            rows.window_start[t]=std::max(nwin-pos,t+1);
            rows.window_count[t]=nwin+t+1-rows.window_start[t];
            rows.selected_count[t]=std::min(ns,pos+t+1);
        }
        const auto run=[&](bool use_overlay) {
            const ds::PackedRows w{use_overlay?actual.p:rotated.p,stride,ds::CacheFormat::window_fp8};
            ds::sparse_attention_predecoded(q.p,w,c,ids.p,ns,rows,sink.p,heads,width,tokens,1/std::sqrt(float(width)),
                out.p,scratch.p,nullptr,false,use_overlay?ring.p:nullptr,start);
        };
        run(false);const auto expected=out.get();run(true);const auto found=out.get();
        check(std::memcmp(expected.data(),found.data(),size_t(tokens)*heads*width*sizeof(float))==0,"ring overlay attention is bit identical");
        for(int t=0;t<tokens;++t) {
            std::copy_n(appended.begin()+t*stride,stride,wanted.begin()+((start+t)%nwin)*stride);
            std::copy_n(linear.begin()+t*stride,stride,overlay.begin()+t*stride);
        }
        ds::append_window_snapshot(ring.p,actual.p,pos,tokens,nullptr);
        check(ring.get()==wanted && actual.get()==overlay,"overlay snapshots only overwritten rows and appends in order");
        if(pos==127 && tokens==6) {
            bench("legacy window rotation and append",size_t(nwin+tokens)*stride,[&] {
                ck(cudaMemcpyAsync(rotated.p,ring.p+start*stride,(nwin-start)*stride,cudaMemcpyDeviceToDevice));
                ck(cudaMemcpyAsync(rotated.p+(nwin-start)*stride,ring.p,start*stride,cudaMemcpyDeviceToDevice));
                ds::update_window_ring(ring.p,rotated.p,pos,tokens,stride,nullptr);
            });
            bench("ring snapshot and append",size_t(tokens)*2*stride,[&]{ds::append_window_snapshot(ring.p,actual.p,pos,tokens,nullptr);});
        }
    }
}
static void ring_copies(std::mt19937& rng) {
    constexpr int bytes=528,capacity=2048;
    std::vector<uint8_t> initial(128*bytes),rows((128+capacity)*bytes);
    for(auto& v:initial)v=uint8_t(rng());for(auto& v:rows)v=uint8_t(rng());
    Device<uint8_t> ring(initial),extended(rows);
    for(int position:{0,1,17,123,128,129,261999})for(int count:{0,1,6,127,128,129,2048}) {
        ring.put(initial);auto expected=initial;
        for(int t=0;t<count;++t)std::memcpy(expected.data()+((position+t)%128)*bytes,rows.data()+(128+t)*bytes,bytes);
        ds::update_window_ring(ring.p,extended.p,position,count,bytes,nullptr);
        check(ring.get()==expected,"coalesced ring copies preserve sequential append bytes");
    }
}
static void frozen_attention() {
    constexpr int h=64,d=512,tokens=11,nwin=128,ncomp=513;
    std::mt19937 rng(9141);std::vector<float> q(size_t(tokens)*h*d),sinks(h);
    for(auto& v:q)v=float(int(rng()%4096)-2048)*0x1p-10f;
    for(auto& v:sinks)v=float(int(rng()%4096)-2048)*0x1p-10f;
    std::vector<uint8_t> win(nwin*528),comp(ncomp*288);
    for(int r=0;r<nwin;++r) {
        for(int c=0;c<512;++c)win[r*528+c]=uint8_t((rng()%127)|((rng()&1)<<7));
        for(int g=0;g<16;++g)win[r*528+512+g]=uint8_t(117+rng()%11);
    }
    for(int r=0;r<ncomp;++r) {
        for(int c=0;c<256;++c)comp[r*288+c]=uint8_t(rng());
        for(int g=0;g<32;++g)comp[r*288+256+g]=uint8_t(32+rng()%49);
    }
    std::vector<int32_t> ids(tokens*512);ds::AttentionRows rows;rows.window_capacity=128;
    for(int t=0;t<tokens;++t) {
        rows.window_start[t]=(t*17)%128;rows.window_count[t]=128;rows.selected_count[t]=512;
        for(int j=0;j<512;++j)ids[t*512+j]=j%17==0?-1:(t+j)%ncomp;
    }
    Device<float> query(q),sink(sinks),out(q.size()),scratch(ds::kAttentionScratch);
    Device<uint8_t> w(win),c(comp);Device<int32_t> selection(ids);
    const ds::PackedRows wr{w.p,528,ds::CacheFormat::window_fp8},cr{c.p,288,ds::CacheFormat::compressed_fp4};
    ds::sparse_attention(query.p,wr,cr,selection.p,512,rows,sink.p,h,d,tokens,1/std::sqrt(float(d)),out.p,nullptr);
    const auto expected=out.get();uint64_t hash=14695981039346656037ull;
    for(float v:expected) {uint32_t bits;std::memcpy(&bits,&v,4);for(int byte=0;byte<4;++byte){hash^=(bits>>(byte*8))&255;hash*=1099511628211ull;}}
    std::printf("frozen attention hash: %016llx\n",(unsigned long long)hash);
    check(hash==0x69b730441d3b9c25ull,"frozen attention output hash");
    ds::sparse_attention_predecoded(query.p,wr,cr,selection.p,512,rows,sink.p,h,d,tokens,1/std::sqrt(float(d)),out.p,scratch.p,nullptr);
    check(out.get()==expected,"frozen packed/shared KV output bits");
}
static void dense(std::mt19937& rng) {
    for (int type : {0,30}) {
        const int in=type==0?5120:1280, out=type==0?384:4096, tokens=5;
        const size_t xs=in+32,ys=out+17;
        const auto original=random(size_t(in)*out,rng),x=random(tokens*xs,rng);
        std::vector<float> weights=original;std::vector<uint16_t> bf16(weights.size());
        if(type==30)for(size_t i=0;i<weights.size();++i) {
            uint32_t bits;std::memcpy(&bits,&weights[i],4);bits+=0x7fff+((bits>>16)&1);bf16[i]=uint16_t(bits>>16);bits=uint32_t(bf16[i])<<16;std::memcpy(&weights[i],&bits,4);
        }
        Device<float> dx(x),dw(weights),dy(std::vector<float>(tokens*ys,123));Device<uint16_t> db(bf16);
        const void* matrix=type==30?static_cast<void*>(db.p):dw.p;
        ds::dense_linear(type,matrix,dx.p,dy.p,in,out,tokens,xs,ys,nullptr);
        const auto batch=dy.get();std::vector<double> wanted(batch.size(),123);
        for(int t=0;t<tokens;++t)for(int r=0;r<out;++r) {
            double dot=0;for(int c=0;c<in;++c)dot+=double(weights[size_t(r)*in+c])*x[t*xs+c];wanted[t*ys+r]=dot;
        }
        compare(batch,wanted,5e-6,type==0?"F32 router projection":"BF16 index projection");
        for(int t=0;t<tokens;++t)ds::dense_linear(type,matrix,dx.p+t*xs,dy.p+t*ys,in,out,1,xs,ys,nullptr);
        check(dy.get()==batch,"dense token positions and output padding match");
        bench(type==0?"F32 dense":"BF16 dense",weights.size()*(type==0?4:2)+x.size()*4+batch.size()*4,[&]{ds::dense_linear(type,matrix,dx.p,dy.p,in,out,tokens,xs,ys,nullptr);});
    }
}
static void residual_and_combine(std::mt19937& rng) {
    constexpr int dim=5120,tokens=5,k=6;
    const auto x=random(tokens*dim,rng),streams=random(tokens*4*dim,rng);
    Device<float> dx(x),dh(streams),pre(tokens*4);
    ds::expand_embedding(dx.p,dh.p,pre.p,dim,tokens,nullptr);
    const auto expanded=dh.get(),mix=pre.get();
    for(int t=0;t<tokens;++t)for(int s=0;s<4;++s) {
        check(mix[t*4+s]==(s?0.f:1.f),"identity pre mix");
        check(std::equal(x.begin()+t*dim,x.begin()+(t+1)*dim,expanded.begin()+(t*4+s)*dim),"embedding four-stream expansion");
    }
    dh.put(streams);constexpr int stride=dim*3;
    Device<float> mean(std::vector<float>(tokens*stride,-19));ds::stream_mean(dh.p,mean.p,dim,tokens,stride,dim,nullptr);
    std::vector<double> ref(tokens*stride,-19);
    for(int t=0;t<tokens;++t)for(int c=0;c<dim;++c) {double v=0;for(int s=0;s<4;++s)v+=streams[(t*4+s)*dim+c];ref[t*stride+dim+c]=v*.25;}
    compare(mean.get(),ref,5e-6,"DSpark anchor stream means");
    const auto gpu=random(tokens*k*dim,rng),cpu=random(tokens*k*dim,rng),weights=random(tokens*k,rng),shared=random(tokens*dim,rng);
    Device<float> g(gpu),c(cpu),w(weights),sh(shared),out(tokens*dim);
    ds::combine_experts(g.p,c.p,w.p,sh.p,out.p,dim,k,tokens,nullptr);
    ref.assign(tokens*dim,0);
    for(int t=0;t<tokens;++t)for(int col=0;col<dim;++col) {
        double sum=shared[t*dim+col];for(int e=0;e<k;++e)sum+=double(weights[t*k+e])*(double(gpu[(t*k+e)*dim+col])+cpu[(t*k+e)*dim+col]);ref[t*dim+col]=sum;
    }
    const auto batch=out.get();compare(batch,ref,5e-6,"CPU/GPU/shared expert combination");
    for(int t=0;t<tokens;++t)ds::combine_experts(g.p+t*k*dim,c.p+t*k*dim,w.p+t*k,sh.p+t*dim,out.p+t*dim,dim,k,1,nullptr);
    check(out.get()==batch,"expert combination token positions match");
    bench("expert combine",size_t(tokens)*(k*dim*2+dim*2+k)*4,[&]{ds::combine_experts(g.p,c.p,w.p,sh.p,out.p,dim,k,tokens,nullptr);});
    ds::add_rows(out.p,sh.p,out.n,nullptr);const auto added=out.get();
    for(size_t i=0;i<added.size();++i)check(added[i]==batch[i]+shared[i],"Markov logit addition");
}
static void expert_accumulator(std::mt19937& rng) {
    constexpr int tokens=128,top=6,h=5120,rows=tokens*top;
    const auto parts=random(size_t(rows)*h,rng),weights=random(rows,rng),shared=random(size_t(tokens)*h,rng);
    Device<float> p(parts),w(weights),sh(shared),out(shared.size()),old(shared.size());Device<double> accum(shared.size());
    std::vector<int32_t> entries(rows);std::iota(entries.begin(),entries.end(),0);std::shuffle(entries.begin(),entries.end(),rng);
    std::vector<float> shuffled(parts.size());
    for(int r=0;r<rows;++r)std::memcpy(shuffled.data()+size_t(r)*h,parts.data()+size_t(entries[r])*h,h*4);
    Device<float> grouped(shuffled);Device<int32_t> ids(entries);
    std::vector<double> reference(shared.begin(),shared.end());
    for(int t=0;t<tokens;++t)for(int k=0;k<top;++k)for(int c=0;c<h;++c)
        reference[size_t(t)*h+c]+=double(weights[t*top+k])*parts[(size_t(t)*top+k)*h+c];
    ds::combine_experts(p.p,nullptr,w.p,sh.p,old.p,h,top,tokens,nullptr);
    compare(old.get(),reference,5e-6,"original expert FP32 reduction");
    std::vector<float> first;
    for(int batch:{1,13,rows}) {
        ds::seed_experts_f64(sh.p,accum.p,shared.size(),nullptr);
        for(int r=0;r<rows;r+=batch)ds::accumulate_experts_f64(grouped.p+size_t(r)*h,ids.p+r,w.p,accum.p,std::min(batch,rows-r),h,top,nullptr);
        ds::finish_experts_f64(accum.p,out.p,shared.size(),nullptr);const auto got=out.get();
        compare(got,reference,1e-7,"bounded expert FP64 accumulator");
        if(first.empty())first=got;else check(first==got,"expert batch width preserves output bits");
    }
    bench("expert FP64 accumulator",parts.size()*4+shared.size()*20,[&]{
        ds::seed_experts_f64(sh.p,accum.p,shared.size(),nullptr);
        ds::accumulate_experts_f64(grouped.p,ids.p,w.p,accum.p,rows,h,top,nullptr);
        ds::finish_experts_f64(accum.p,out.p,shared.size(),nullptr);
    });
}
static void indexer(std::mt19937& rng) {
    constexpr int heads=32,dim=128,tokens=5,count=79,bs=8,blocks=(count+bs-1)/bs;
    const auto q=random(tokens*heads*dim,rng),weights=random(tokens*heads,rng);
    const auto raw=pack(random(count*dim,rng),dim,ds::CacheFormat::index_fp4);
    Device<float> dq(q),dw(weights),out(tokens*count);Device<uint8_t> dk(raw);
    std::vector<uint8_t> mask(tokens*blocks);for(size_t i=0;i<mask.size();++i)mask[i]=(i%3)!=0;Device<uint8_t> dm(mask);
    const ds::PackedRows keys{dk.p,ds::cache_row_bytes(dim,ds::CacheFormat::index_fp4),ds::CacheFormat::index_fp4};
    for(bool masked:{false,true})for(float scale:{1.f,1.f/64}) {
        ds::IndexRows rows;for(int t=0;t<tokens;++t)rows.visible[t]=t*19;
        ds::index_scores(dq.p,dw.p,keys,masked?dm.p:nullptr,blocks,bs,out.p,count,heads,dim,tokens,count,rows,nullptr,scale);
        const auto batch=out.get();std::vector<double> actual,wanted;
        for(int t=0;t<tokens;++t)for(int key=0;key<count;++key) {
            if(key>=rows.visible[t] || (masked && !mask[t*blocks+key/bs])) {check(batch[t*count+key]==-INFINITY,"future/candidate-excluded index positions unreachable");continue;}
            double sum=0;
            for(int h=0;h<heads;++h) {double dot=0;for(int c=0;c<dim;++c)dot+=double(q[(t*heads+h)*dim+c])*value(raw,key,c,dim,ds::CacheFormat::index_fp4);sum+=std::max(0.,dot)*weights[t*heads+h];}
            actual.push_back(batch[t*count+key]);wanted.push_back(sum*scale);
        }
        compare(std::vector<float>(actual.begin(),actual.end()),wanted,5e-6,"compressed index scores");
        for(int t=0;t<tokens;++t) {
            auto one=rows;one.visible[0]=rows.visible[t];
            ds::index_scores(dq.p+t*heads*dim,dw.p+t*heads,keys,masked?dm.p+t*blocks:nullptr,blocks,bs,out.p+t*count,count,heads,dim,1,count,one,nullptr,scale);
        }
        check(out.get()==batch,"index score windows and positions match");
        if(!masked)bench("index scores",q.size()*4+weights.size()*4+raw.size()+tokens*count*4,[&]{ds::index_scores(dq.p,dw.p,keys,nullptr,blocks,bs,out.p,count,heads,dim,tokens,count,rows,nullptr);});
    }
}
static void batched_selection(std::mt19937& rng) {
    constexpr int tokens=8,stride=32768,mask_stride=(stride+7)/8;
    auto values=random(size_t(tokens)*stride,rng);ds::IndexRows rows;
    const int counts[]={0,1,17,512,513,2048,16385,stride};
    for(int t=0;t<tokens;++t) {rows.visible[t]=counts[t];for(int j=0;j<stride;++j)
        values[size_t(t)*stride+j]=j%17==0?-INFINITY:float(int(values[size_t(t)*stride+j]*3));}
    Device<float> x(values);Device<int32_t> out(tokens*512);Device<uint8_t> mask(tokens*mask_stride);
    ds::select_topk_rows(x.p,stride,rows,tokens,out.p,512,nullptr);
    ds::candidate_topk_rows(x.p,stride,rows,tokens,mask.p,mask_stride,nullptr);
    const auto got=out.get();const auto masks=mask.get();
    for(int t=0;t<tokens;++t) {
        const int count=counts[t];std::vector<int32_t> order(count);std::iota(order.begin(),order.end(),0);
        const auto* scores=values.data()+size_t(t)*stride;
        std::stable_sort(order.begin(),order.end(),[&](int a,int b){return scores[a]>scores[b];});
        order.resize(std::min(count,512));for(auto& id:order)if(scores[id]==-INFINITY)id=INT32_MAX;
        std::sort(order.begin(),order.end());order.resize(512,INT32_MAX);
        for(int j=0;j<512;++j)check(got[t*512+j]==(order[j]==INT32_MAX?-1:order[j]),"batched top-512 stable ties, visibility and masks");
        const int blocks=(count+7)/8;std::vector<float> maxes(blocks,-INFINITY);std::vector<int32_t> ids(blocks);
        for(int b=0;b<blocks;++b){ids[b]=b;for(int j=b*8;j<std::min(count,b*8+8);++j)maxes[b]=std::max(maxes[b],scores[j]);}
        if(blocks)maxes.back()=INFINITY;
        std::stable_sort(ids.begin(),ids.end(),[&](int a,int b){return maxes[a]>maxes[b];});ids.resize(std::min(blocks,2048));
        std::vector<uint8_t> expected(blocks,0);for(int id:ids)if(maxes[id]!=-INFINITY)expected[id]=1;
        for(int b=0;b<blocks;++b)check(masks[t*mask_stride+b]==expected[b],"batched candidate blocks match CPU selection");
    }
    const size_t read_bytes=size_t(std::accumulate(std::begin(counts),std::end(counts),0))*4;
    bench("batched top-512",read_bytes,[&]{ds::select_topk_rows(x.p,stride,rows,tokens,out.p,512,nullptr);});
    bench("batched candidate top-2048",read_bytes,[&]{ds::candidate_topk_rows(x.p,stride,rows,tokens,mask.p,mask_stride,nullptr);});
}
static void radix_selection(std::mt19937& rng) {
    constexpr int tokens=8,stride=32768,out_stride=520;
    const int lengths[tokens]={0,513,1280,32768,511,12457,32767,32768};
    auto& flag=basalt::core::diagnostics().exp["ds-index-topk-radix"];
    auto values=random(size_t(tokens)*stride,rng);
    Device<float> x(values);Device<int32_t> out(tokens*out_stride);
    ds::IndexRows rows;std::copy_n(lengths,tokens,rows.visible);
    const auto run=[&]{ds::select_topk_rows(x.p,stride,rows,tokens,out.p,out_stride,nullptr);};
    const std::vector<int32_t> canary(tokens*out_stride,-7);
    for(int pattern=0;pattern<4;++pattern) {
        values=random(size_t(tokens)*stride,rng);
        for(int t=0;t<tokens;++t)for(int j=0;j<stride;++j) {
            auto& v=values[size_t(t)*stride+j];
            if(pattern==1)v=j%13==0?-INFINITY:j%2?-0.f:+0.f;
            if(pattern==2)v=j%97==0?INFINITY:j%17==0?float(j%29-15):-INFINITY;
            if(pattern==3 && (t==4 || t==6) && j%197==0)v=NAN;
        }
        x.put(values);out.put(canary);
        flag=0;run();const auto reference=out.get();
        flag=1;run();const auto candidate=out.get();
        check(candidate==reference,"radix top-512 matches selection order, ties, masks and stride");
        for(int t=0;t<tokens;++t)for(int j=512;j<out_stride;++j)
            check(candidate[t*out_stride+j]==-7,"radix top-512 output stride canary");
        if(pattern==0 || pattern==1)for(int enabled:{0,1,1,0}) {
            flag=enabled;
            bench(enabled?"radix top-512":"bitonic top-512",0,run);
        }
    }
    flag=0;
}
static void tensor_indexer(std::mt19937& rng) {
    constexpr int tokens=128,heads=32,width=128,count=4097,ms=(count+7)/8;
    const auto packed_q=pack(random(size_t(tokens)*heads*width,rng),width,ds::CacheFormat::index_fp4);
    const auto packed_k=pack(random(size_t(count)*width,rng),width,ds::CacheFormat::index_fp4);
    std::vector<float> q(size_t(tokens)*heads*width),keys(size_t(count)*width);
    for(int r=0;r<tokens*heads;++r)for(int c=0;c<width;++c)q[size_t(r)*width+c]=float(value(packed_q,r,c,width,ds::CacheFormat::index_fp4));
    for(int r=0;r<count;++r)for(int c=0;c<width;++c)keys[size_t(r)*width+c]=float(value(packed_k,r,c,width,ds::CacheFormat::index_fp4));
    const auto weights=random(tokens*heads,rng);Device<float> dq(q),dw(weights),old(tokens*count),out(tokens*count);
    Device<uint8_t> dk(packed_k);std::vector<uint8_t> masks(tokens*ms);
    for(size_t i=0;i<masks.size();++i)masks[i]=i%3!=0;Device<uint8_t> mask(masks);
    ds::IndexRows rows;for(int t=0;t<tokens;++t)rows.visible[t]=t*count/(tokens-1);
    const ds::PackedRows packed{dk.p,ds::cache_row_bytes(width,ds::CacheFormat::index_fp4),ds::CacheFormat::index_fp4};
    std::vector<void*> buffers;
    {
        ds::PrefillIndex index(nullptr,[&](size_t bytes){uint8_t* p=nullptr;ck(cudaMalloc(&p,bytes));buffers.push_back(p);return p;});
        for(bool masked:{false,true}) {
            ds::index_scores(dq.p,dw.p,packed,masked?mask.p:nullptr,ms,8,old.p,count,heads,width,tokens,count,rows,nullptr,1.f/64);
            index.run(dq.p,dw.p,packed,masked?mask.p:nullptr,ms,out.p,count,tokens,count,rows);
            const auto got=out.get(),previous=old.get();double maximum=0,error=0,total=0;
            for(size_t i=0;i<got.size();++i) {
                if(previous[i]==-INFINITY)check(got[i]==previous[i],"tensor index visibility and candidates");
                else {maximum=std::max(maximum,std::abs(double(got[i])-previous[i]));error+=std::abs(double(got[i])-previous[i]);total+=std::abs(previous[i]);}
            }
            std::printf("tensor index versus current masked=%d: max_abs %.7g L1_rel %.7g\n",masked,maximum,error/std::max(total,1e-30));
            check(maximum<3e-5,"tensor index current-kernel bound");
            double cpu_error=0,cpu_l1=0,cpu_magnitude=0;
            for(int t:{1,63,127})for(int key:{0,18,2048,4096}) {
                if(key>=rows.visible[t] || (masked && !masks[t*ms+key/8]))continue;
                double reference=0;
                for(int h=0;h<heads;++h) {
                    double dot=0;for(int c=0;c<width;++c)dot+=double(q[(size_t(t)*heads+h)*width+c])*keys[size_t(key)*width+c];
                    reference+=std::max(0.,dot)*double(weights[t*heads+h])/64;
                }
                const double difference=std::abs(got[t*count+key]-reference);
                cpu_error=std::max(cpu_error,difference);cpu_l1+=difference;cpu_magnitude+=std::abs(reference);
                check(difference<3e-5,"tensor index CPU-double bound");
            }
            std::printf("tensor index CPU-double masked=%d: max_abs %.7g L1_rel %.7g\n",masked,cpu_error,cpu_l1/std::max(cpu_magnitude,1e-30));
            auto one=rows;one.visible[0]=rows.visible[127];
            index.run(dq.p+size_t(127)*heads*width,dw.p+127*heads,packed,masked?mask.p+127*ms:nullptr,ms,out.p,count,1,count,one);
            const auto single=out.get();
            for(int key=0;key<count;++key)check(single[key]==got[127*count+key],"tensor index query position bits");
        }
        bench("tensor index 128x4097",size_t(tokens)*heads*width*4+size_t(count)*packed.stride+size_t(tokens)*count*4,[&]{
            index.run(dq.p,dw.p,packed,nullptr,ms,out.p,count,tokens,count,rows);
        });
    }
    for(void* p:buffers)cudaFree(p);
}
static void selection(std::mt19937& rng) {
    for(int count:{1,17,513,65536}) {
        auto scores=random(count,rng);for(int i=0;i<count;++i) {if(i%11==0)scores[i]=.5f;if(i%19==0)scores[i]=-INFINITY;}
        scores.back()=INFINITY;
        const int k=std::min(count,512);
        Device<float> dscores(scores),sorted(count);Device<int32_t> ids(count),sorted_ids(count),result(k);
        const size_t bytes=ds::topk_workspace_bytes(count);Device<uint8_t> workspace(bytes);
        ds::select_topk(dscores.p,count,k,ids.p,sorted.p,sorted_ids.p,result.p,workspace.p,bytes,nullptr);
        std::vector<int32_t> ref(count);std::iota(ref.begin(),ref.end(),0);
        std::stable_sort(ref.begin(),ref.end(),[&](int a,int b){return scores[a]>scores[b];});ref.resize(k);
        for(auto& id:ref)if(scores[id]==-INFINITY)id=INT32_MAX;std::sort(ref.begin(),ref.end());for(auto& id:ref)if(id==INT32_MAX)id=-1;
        check(result.get()==ref,"top-k stable ties, unreachable rows and ascending position order");
        if(count==65536)bench("top-k",size_t(count)*16,[&]{ds::select_topk(dscores.p,count,k,ids.p,sorted.p,sorted_ids.p,result.p,workspace.p,bytes,nullptr);});
        if(count==513) {
            const int bs=8,n=(count+bs-1)/bs,chosen=3;
            Device<float> block(n),blocksorted(n);Device<int32_t> bid(n),bsorted(n),picks(chosen);Device<uint8_t> mask(n);
            ds::candidate_block_scores(dscores.p,count,bs,block.p,nullptr);
            const auto got=block.get();for(int i=0;i<n;++i) {
                float maximum=-INFINITY;for(int j=i*bs;j<std::min(count,(i+1)*bs);++j)maximum=std::max(maximum,scores[j]);
                check(got[i]==(i==n-1?INFINITY:maximum),"candidate blocks preserve maxima and pin latest block");
            }
            ds::select_topk(block.p,n,chosen,bid.p,blocksorted.p,bsorted.p,picks.p,workspace.p,bytes,nullptr);
            ds::candidate_mask(picks.p,chosen,mask.p,n,nullptr);const auto picked=picks.get();const auto m=mask.get();
            for(int i=0;i<n;++i)check(m[i]==(std::find(picked.begin(),picked.end(),i)!=picked.end()),"candidate bitmap matches selected blocks");
            check(m.back()==1,"newest candidate block selected");
        }
    }
}
static void all_selection(std::mt19937& rng) {
    for(int count:{1,2,17,127,256,511,512,513,1024,2048})for(int trial=0;trial<5;++trial) {
        auto values=random(count,rng);
        for(int i=0;i<count;++i) {
            if(trial==1 && i%7==0)values[i]=-INFINITY;
            if(trial==2)values[i]=-INFINITY;
            if(trial==3)values[i]=INFINITY;
            if(trial==4 && i%3==0)values[i]=std::numeric_limits<float>::quiet_NaN();
        }
        Device<float> scores(values),sorted(count);Device<int32_t> ids(count),sorted_ids(count),out(count+2);
        const size_t bytes=ds::topk_workspace_bytes(count);Device<uint8_t> workspace(bytes);
        std::vector<int32_t> wanted(count+2,-77);int next=1;
        for(int i=0;i<count;++i)if(values[i]!=-INFINITY)wanted[next++]=i;
        while(next<=count)wanted[next++]=-1;
        for(int enabled:{0,1}) {
            out.put(std::vector<int32_t>(count+2,-77));basalt::core::diagnostics().exp["ds-all-select"]=enabled;
            ds::select_topk(scores.p,count,count,ids.p,sorted.p,sorted_ids.p,out.p+1,workspace.p,bytes,nullptr);
            check(out.get()==wanted,"select-all compaction matches CUB, CPU, NaNs/infinities and canaries");
        }
        if(trial==0 && (count==17 || count==512 || count==2048)) {
            cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));float times[2]{};
            for(int enabled=0;enabled<2;++enabled) {
                basalt::core::diagnostics().exp["ds-all-select"]=enabled;cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
                ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
                for(int i=0;i<32;++i)ds::select_topk(scores.p,count,count,ids.p,sorted.p,sorted_ids.p,out.p+1,workspace.p,bytes,stream);
                ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,0));
                for(int i=0;i<5;++i)ck(cudaGraphLaunch(exec,stream));
                cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));ck(cudaEventRecord(a,stream));
                for(int i=0;i<10;++i)ck(cudaGraphLaunch(exec,stream));
                ck(cudaEventRecord(b,stream));ck(cudaEventSynchronize(b));ck(cudaEventElapsedTime(&times[enabled],a,b));times[enabled]*=1000.f/320;
                cudaEventDestroy(a);cudaEventDestroy(b);cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);
            }
            std::printf("Select-all n=%d CUB_us=%.3f compact_us=%.3f minimum_GB_s=%.3f bits=identical\n",count,times[0],times[1],count*8./(times[1]*1000));
            ck(cudaStreamDestroy(stream));
        }
    }
    basalt::core::diagnostics().exp["ds-all-select"]=0;
}
int main(int argc, char** argv) {
    try {
        int device = 0; std::string path;bool picks_only=false,packed_prob_only=false,radix_only=false,draft_shared_only=false,verify_fixed_only=false,rollback_only=false,overlay_only=false;
        for (int i = 1; i < argc; ++i) { const std::string arg = argv[i];
            if (arg == "--device" && i+1 < argc) device = std::stoi(argv[++i]);
            else if(arg=="--draft-picks")picks_only=true;
            else if(arg=="--packed-prob")packed_prob_only=true;
            else if(arg=="--index-radix")radix_only=true;
            else if(arg=="--draft-shared-attention")draft_shared_only=true;
            else if(arg=="--verify-fixed-attention")verify_fixed_only=true;
            else if(arg=="--rollback-window")rollback_only=true;
            else if(arg=="--ring-overlay")overlay_only=true;
            else if (arg == "--pack" && i+1 < argc) path = argv[++i]; else return 2;
        }
        ck(cudaSetDevice(device)); std::mt19937 rng(417);
        if(picks_only) {
            draft_picking(rng);std::printf("draft picking: %d failures\n",failures);return failures?1:0;
        }
        if(packed_prob_only) {
            attention(rng,true);std::printf("packed probability: %d failures\n",failures);return failures?1:0;
        }
        if(radix_only) {
            radix_selection(rng);std::printf("radix selection: %d failures\n",failures);return failures?1:0;
        }
        if(draft_shared_only) {
            draft_shared_attention(rng);std::printf("shared draft attention: %d failures\n",failures);return failures?1:0;
        }
        if(verify_fixed_only) {
            attention(rng,false,true);std::printf("fixed-width verification attention: %d failures\n",failures);return failures?1:0;
        }
        if(rollback_only) {
            rollback_copies(rng);std::printf("batched rollback: %d failures\n",failures);return failures?1:0;
        }
        if(overlay_only) {
            ring_overlay(rng);std::printf("ring overlay: %d failures\n",failures);return failures?1:0;
        }
        cache_codes();ring_copies(rng);frozen_attention();draft_picking(rng);finish(rng); compressor(rng); engram(rng); attention(rng);dense(rng);residual_and_combine(rng);expert_accumulator(rng);indexer(rng);selection(rng);all_selection(rng);batched_selection(rng);tensor_indexer(rng);compressor_rows(rng);
        if (!path.empty()) {
            ds::PackedWeights weights(path);
            for (int layer : {1, 14}) {
                const auto tensor = weights.target_tensor("blk." + std::to_string(layer) + ".engram_gate.weight");
                check(tensor.tensor->type == 0 && tensor.bytes == 4 * 5120 * 4, "real Engram gate geometry");
                std::vector<float> w(4 * 5120); std::memcpy(w.data(), tensor.data(), size_t(tensor.bytes)); engram(rng, &w);
            }
        }
        std::printf("deepseek41_attention_test: %d failures\n", failures); return failures ? 1 : 0;
    } catch (const std::exception& e) { std::printf("deepseek41_attention_test: %s\n", e.what()); return 1; }
}
