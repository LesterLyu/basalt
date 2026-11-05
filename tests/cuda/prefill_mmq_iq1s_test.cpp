#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/ggml/ggml.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace mmq=basalt::prefill::mmq;
namespace k=basalt::kernels;
static void ck(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void check(bool ok,const char* why) {if(!ok)throw std::runtime_error(why);}
struct Dev {
    void* p=nullptr;
    explicit Dev(size_t n){ck(cudaMalloc(&p,n));}
    ~Dev(){cudaFree(p);}
    template<class T>T* as(){return static_cast<T*>(p);}
};
struct Q8 {float scale[4];int8_t values[128];};
static_assert(sizeof(Q8)==144);
static void run(int n,int cols,int tokens,std::mt19937& rng) {
    const size_t stride=k::iq_row_bytes(19,cols);
    std::vector<uint8_t> weights(size_t(n)*stride);
    for(auto& b:weights)b=uint8_t(rng());
    for(size_t b=0;b<weights.size();b+=50) {
        const uint16_t scale=ggml_fp32_to_fp16((rng()&1?-1.f:1.f)/256.f);
        std::memcpy(weights.data()+b,&scale,2);
    }
    std::vector<float> x(size_t(tokens)*cols);std::normal_distribution<float> normal;
    for(auto& v:x)v=normal(rng);
    if(tokens>1)std::fill_n(x.begin(),cols,0.f);
    if(tokens>1)x[cols+cols/3]=20.f;
    Dev dw(weights.size()),dx(x.size()*4),dq(mmq::q8_bytes(tokens,cols)),dy(size_t(tokens)*n*4);
    Dev native_q(size_t(tokens)*cols/32*36),native_y(size_t(tokens)*n*4),bounds(8),ids(size_t(tokens)*4);
    ck(cudaMemcpy(dw.p,weights.data(),weights.size(),cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dx.p,x.data(),x.size()*4,cudaMemcpyHostToDevice));
    const int32_t host_bounds[]={0,tokens};ck(cudaMemcpy(bounds.p,host_bounds,8,cudaMemcpyHostToDevice));
    mmq::iota(ids.as<int32_t>(),tokens,nullptr);
    mmq::quantize(dx.as<float>(),nullptr,dq.p,19,cols,cols,tokens,nullptr);
    mmq::Product p;
    p.w=dw.p;p.type=19;p.w_rows=n;p.w_cols=cols;p.expert_bytes=weights.size();p.n=1;
    p.xq=dq.p;p.bounds=bounds.as<int32_t>();p.ids=ids.as<int32_t>();p.total_rows=tokens;p.max_rows=tokens;
    p.dst=dy.as<float>();p.ld_dst=n;
    mmq::Context context;context.run(p,nullptr);
    k::quantize_q8_1_rows(dx.as<float>(),tokens,cols,native_q.p,nullptr);
    for(int first=0;first<tokens;first+=8)
        k::native_mmvq(19,dw.p,native_q.as<uint8_t>()+size_t(first)*cols/32*36,
                       native_y.as<float>()+size_t(first)*n,cols,n,std::min(8,tokens-first),nullptr);
    std::vector<float> got(size_t(tokens)*n),native(got.size());
    ck(cudaMemcpy(got.data(),dy.p,got.size()*4,cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(native.data(),native_y.p,native.size()*4,cudaMemcpyDeviceToHost));
    std::vector<Q8> q(mmq::q8_bytes(tokens,cols)/sizeof(Q8));
    ck(cudaMemcpy(q.data(),dq.p,q.size()*sizeof(Q8),cudaMemcpyDeviceToHost));
    double native_error=0,native_norm=0,max_abs=0,max_l1=0;
    for(size_t j=0;j<got.size();++j) {
        check(std::isfinite(got[j]),"nonfinite MMQ result");
        native_error+=std::pow(double(got[j])-native[j],2);native_norm+=double(native[j])*native[j];
    }
    const double relative_rms=std::sqrt(native_error/std::max(native_norm,1e-30));
    check(relative_rms<.02,"MMQ differs excessively from the existing IQ1_S kernel");
    const auto* traits=ggml_get_type_traits(GGML_TYPE_IQ1_S);
    std::vector<float> row(cols);
    for(int r:{0,1,2,3,17,n/2,n-1}) {
        if(r>=n)continue;
        traits->to_float(weights.data()+size_t(r)*stride,row.data(),cols);
        for(int t=0;t<tokens;++t) {
            double reference=0,l1=0;
            for(int c=0;c<cols;++c) {
                const auto& block=q[size_t(c/128)*tokens+t];
                const double v=double(row[c])*block.values[c%128]*block.scale[(c%128)/32];
                reference+=v;l1+=std::abs(v);
            }
            const double error=std::abs(got[size_t(t)*n+r]-reference);
            max_abs=std::max(max_abs,error);max_l1=std::max(max_l1,error/std::max(l1,1e-12));
            check(error<=2e-5+2e-6*l1,"MMQ violates CPU-double arithmetic bound");
        }
    }
    if(tokens>1)for(int r=0;r<n;++r)check(got[r]==0,"zero activation row is not zero");
    std::printf("IQ1_S MMQ N=%d K=%d T=%d: double max_abs=%.7g L1_rel=%.7g native_rel_rms=%.7g\n",
                n,cols,tokens,max_abs,max_l1,relative_rms);std::fflush(stdout);
    if(tokens==128) {
        cudaEvent_t start,stop;ck(cudaEventCreate(&start));ck(cudaEventCreate(&stop));
        for(int i=0;i<3;++i)context.run(p,nullptr);
        ck(cudaEventRecord(start));for(int i=0;i<5;++i)context.run(p,nullptr);ck(cudaEventRecord(stop));ck(cudaEventSynchronize(stop));
        float ms=0;ck(cudaEventElapsedTime(&ms,start,stop));
        std::printf("micro IQ1_S MMQ N=%d K=%d T=%d: %.3f us %.3f TFLOP/s\n",n,cols,tokens,ms*1000/5,2.*n*cols*tokens/(ms/5000)/1e12);
        cudaEventDestroy(start);cudaEventDestroy(stop);
    }
}
static void swiglu(std::mt19937& rng) {
    constexpr int rows=33,ff=2304;std::normal_distribution<float> normal;
    std::vector<float> gu(size_t(rows)*2*ff);for(auto& v:gu)v=20*normal(rng);
    Dev input(gu.size()*4),output(size_t(rows)*ff*4);ck(cudaMemcpy(input.p,gu.data(),gu.size()*4,cudaMemcpyHostToDevice));
    for(float limit:{0.f,10.f}) {
        mmq::swiglu(input.as<float>(),output.as<float>(),rows,ff,false,nullptr,nullptr,nullptr,0,limit);
        std::vector<float> got(size_t(rows)*ff);ck(cudaMemcpy(got.data(),output.p,got.size()*4,cudaMemcpyDeviceToHost));
        double maximum=0;
        for(int r=0;r<rows;++r)for(int j=0;j<ff;++j) {
            double g=gu[size_t(r)*2*ff+j],u=gu[size_t(r)*2*ff+ff+j];
            if(limit>0){g=std::min(g,double(limit));u=std::clamp(u,-double(limit),double(limit));}
            const double ref=g/(1+std::exp(-g))*u,error=std::abs(got[size_t(r)*ff+j]-ref);
            maximum=std::max(maximum,error);check(error<3e-5*std::max(1.,std::abs(ref)),"MMQ SwiGLU reference mismatch");
        }
        std::printf("MMQ SwiGLU limit=%g max_abs=%.7g\n",limit,maximum);
    }
}
// Screen portability of the exact MMQ math before scheduling any groups on the helper.
static std::vector<float> device_product(int device,int type,int out,int in,const std::vector<int32_t>& bounds,
                                         const std::vector<uint8_t>& weights,const std::vector<float>& x,int stream_k_sms=0) {
    ck(cudaSetDevice(device));
    const int tokens=bounds.back(),experts=int(bounds.size())-1;
    int max_rows=0;
    for(int e=0;e<experts;++e)max_rows=std::max(max_rows,bounds[e+1]-bounds[e]);
    Dev w(weights.size()),dx(x.size()*4),q(mmq::q8_bytes(tokens,in)),y(size_t(tokens)*out*4),
        bb(bounds.size()*sizeof(int32_t)),ids(tokens*sizeof(int32_t));
    ck(cudaMemcpy(w.p,weights.data(),weights.size(),cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dx.p,x.data(),x.size()*4,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(bb.p,bounds.data(),bounds.size()*sizeof(int32_t),cudaMemcpyHostToDevice));
    mmq::iota(ids.as<int32_t>(),tokens,nullptr);
    mmq::quantize(dx.as<float>(),nullptr,q.p,type,in,in,tokens,nullptr);
    mmq::Product p;
    p.w=w.p;p.type=type;p.w_rows=out;p.w_cols=in;p.expert_bytes=weights.size()/experts;p.n=experts;
    p.xq=q.p;p.bounds=bb.as<int32_t>();p.ids=ids.as<int32_t>();p.total_rows=tokens;p.max_rows=max_rows;
    p.dst=y.as<float>();p.ld_dst=out;p.stream_k_sms=stream_k_sms;
    mmq::Context ctx;
    for(int i=0;i<3;++i)ctx.run(p,nullptr);
    cudaEvent_t start,stop;ck(cudaEventCreate(&start));ck(cudaEventCreate(&stop));ck(cudaEventRecord(start));
    for(int i=0;i<8;++i)ctx.run(p,nullptr);
    ck(cudaEventRecord(stop));ck(cudaEventSynchronize(stop));
    float ms=0;ck(cudaEventElapsedTime(&ms,start,stop));ck(cudaEventDestroy(start));ck(cudaEventDestroy(stop));
    std::printf("GPU %d type %d N=%d K=%d experts=%d rows=%d max=%d logical_sms=%d: %.3f us/product\n",
                device,type,out,in,experts,tokens,max_rows,stream_k_sms,ms*125);
    std::vector<float> result(size_t(tokens)*out);ck(cudaMemcpy(result.data(),y.p,result.size()*4,cudaMemcpyDeviceToHost));
    return result;
}
static void cross_device(bool match_partition) {
    int devices=0;ck(cudaGetDeviceCount(&devices));check(devices>=2,"cross-device test requires two GPUs");
    cudaDeviceProp reference{};ck(cudaGetDeviceProperties(&reference,0));
    bool all_exact=true;
    std::mt19937 rng(731);
    const int geometries[][3]={{19,4608,5120},{19,5120,2304},{16,5120,2304}};
    const std::vector<std::vector<int32_t>> groups={{0,31,63,96,160,226,321,449,594},
                                                     {0,7,20,39,70,101,160,243,499}};
    for(const auto& shape:geometries) {
        const int type=shape[0],out=shape[1],in=shape[2];
        std::vector<uint8_t> weights(8*mmq::matrix_bytes(type,out,in));
        for(auto& b:weights)b=uint8_t(rng());
        const size_t block=ggml_type_size(static_cast<ggml_type>(type));
        for(size_t i=0;i<weights.size();i+=block) {
            const uint16_t scale=ggml_fp32_to_fp16((rng()&1?-1.f:1.f)/256.f);
            std::memcpy(weights.data()+i,&scale,2);
        }
        for(const auto& bounds:groups) {
            std::vector<float> x(size_t(bounds.back())*in);
            for(auto& v:x)v=float(int(rng()>>16)-32768)*0x1p-14f;
            const auto a=device_product(0,type,out,in,bounds,weights,x);
            const auto b=device_product(1,type,out,in,bounds,weights,x,match_partition?reference.multiProcessorCount:0);
            size_t exact=0;double peak=0,delta=0,norm=0;
            for(size_t i=0;i<a.size();++i) {
                check(std::isfinite(a[i]) && std::isfinite(b[i]),"nonfinite cross-device output");
                exact+=std::memcmp(&a[i],&b[i],sizeof(float))==0;
                const double d=double(a[i])-b[i];
                peak=std::max(peak,std::abs(d));delta+=d*d;norm+=double(a[i])*a[i];
            }
            std::printf("cross-device exact=%zu/%zu max_abs=%.9g relative_rms=%.9g\n",
                        exact,a.size(),peak,std::sqrt(delta/std::max(norm,1e-30)));
            all_exact&=exact==a.size();
        }
    }
    if(match_partition)check(all_exact,"logical stream-K partitions changed output bits");
}

static void fused_swiglu() {
    constexpr int width=2304;
    std::mt19937 rng(417);
    for(int rows:{1,17,2048}) {
        std::vector<float> x(size_t(rows)*width*2);
        for(auto& v:x)v=float(int(rng()>>16)-32768)*0x1p-10f;
        std::fill_n(x.begin(),width*2,0.f);
        Dev gu(x.size()*sizeof(float)),hidden(size_t(rows)*width*sizeof(float));
        const size_t bytes=mmq::q8_bytes(rows,width);
        Dev a(bytes),b(bytes);
        ck(cudaMemcpy(gu.p,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice));
        for(int type:{19,16}) {
            const auto launch=[&](int fused) {
                if(fused)mmq::quantize_swiglu(gu.as<float>(),b.p,type,width,rows,10,nullptr);
                else {
                    mmq::swiglu(gu.as<float>(),hidden.as<float>(),rows,width,false,nullptr,nullptr,nullptr,0,10);
                    mmq::quantize(hidden.as<float>(),nullptr,a.p,type,width,width,rows,nullptr);
                }
            };
            ck(cudaMemset(a.p,0x7b,bytes));ck(cudaMemset(b.p,0x7b,bytes));launch(0);launch(1);
            std::vector<uint8_t> ref(bytes),got(bytes);
            ck(cudaMemcpy(ref.data(),a.p,bytes,cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(got.data(),b.p,bytes,cudaMemcpyDeviceToHost));
            check(ref==got,"fused SwiGLU quantizer changed Q8 bytes");
            if(rows!=2048)continue;
            cudaEvent_t start{},end{};ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));
            for(int fused:{0,1,1,0}) {
                launch(fused);ck(cudaEventRecord(start));
                for(int i=0;i<80;++i)launch(fused);
                ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));float ms=0;
                ck(cudaEventElapsedTime(&ms,start,end));
                std::printf("SwiGLU %d fused=%d %.3f us per 2048 rows, exact bytes\n",type,fused,ms*1000.f/80);
            }
            ck(cudaEventDestroy(start));ck(cudaEventDestroy(end));
        }
    }
}
int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--cross-device") {cross_device(false);return 0;}
        if(argc==2 && std::string(argv[1])=="--cross-device-fixed") {cross_device(true);return 0;}
        if(argc==2 && std::string(argv[1])=="--fused-swiglu") {fused_swiglu();return 0;}
        int device=argc==2?std::stoi(argv[1]):0;ck(cudaSetDevice(device));
        check(mmq::supported(19) && mmq::fits(19,4608),"IQ1_S MMQ is unavailable");
        std::mt19937 rng(417);
        for(int tokens:{1,8,33,128}) {run(4608,5120,tokens,rng);run(5120,2304,tokens,rng);}
        run(129,512,33,rng);
        swiglu(rng);
        std::printf("prefill_mmq_iq1s_test: 0 failures\n");return 0;
    }catch(const std::exception& e){std::printf("prefill_mmq_iq1s_test: %s\n",e.what());return 1;}
}
