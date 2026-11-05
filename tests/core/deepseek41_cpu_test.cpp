#include "basalt/core/diag.hpp"
#include "basalt/deepseek41/cpu_experts.hpp"
#include "basalt/deepseek41/draft_window.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>

namespace cpu = basalt::kernels::cpu;
static std::vector<uint8_t> blob(const cpu::NativeFmt& f, std::mt19937& rng) {
    std::vector<uint8_t> result(f.bytes); for (auto& b : result) b = uint8_t(rng());
    for (int role = 0; role < 3; ++role) {
        const int type = role == 2 ? f.d_type : f.gu_type;
        const size_t begin = role == 0 ? 0 : role == 1 ? f.up_off : f.down_off;
        const size_t end = role == 0 ? f.up_off : role == 1 ? f.down_off : f.bytes;
        const size_t stride = type == 19 ? 50 : type == 16 ? 66 : 17;
        for (size_t at = begin; at < end; at += stride) {
            if (type == 39) result[at] = 120;
            else { const uint16_t scale = 0x1400; std::memcpy(result.data() + at, &scale, 2); }
        }
    }
    return result;
}
static std::vector<float> serial(const cpu::NativeFmt& f, const std::vector<float>& x, const std::vector<const uint8_t*>& blobs) {
    std::vector<uint8_t> aq(f.act_bytes), hq(f.h_bytes);std::vector<float> h(size_t(f.n_ff)), out(blobs.size()*size_t(f.n_embd));
    cpu::native_quant_act(f,x.data(),aq.data());
    for(size_t e=0;e<blobs.size();++e) {
        const void* a[]={aq.data()};float* ff[]={h.data()};cpu::native_gu_rows(f,blobs[e],a,1,ff,0,int(f.n_ff));
        cpu::native_quant_h(f,h.data(),hq.data());const void* q[]={hq.data()};float* y[]={out.data()+e*size_t(f.n_embd)};
        cpu::native_down_rows(f,blobs[e],q,1,y,0,int(f.n_embd));
    }
    return out;
}
static int cold_bench() {
    std::mt19937 rng(417);std::vector<float> x(5120);
    for(auto& value:x)value=float(int(rng()>>16)-32768)*0x1p-15f;
    for(auto pair:{std::pair<int,int>{19,19},{19,16}}) {
        cpu::NativeFmt f;std::string error;
        if(!cpu::native_fmt(pair.first,pair.second,5120,2304,f,error,{8192,4096}))throw std::runtime_error(error);
        f.swiglu_limit=10;
        std::vector<std::vector<uint8_t>> weights;
        for(int i=0;i<16;++i)weights.push_back(blob(f,rng));
        const auto expected=serial(f,x,{weights[0].data()});
        std::printf("cold CPU %d/%d working set %.2f MiB\n",pair.first,pair.second,16.*f.bytes/(1<<20));
        for(int workers:{1,4,7,8,11,15}) {
            basalt::deepseek41::CpuExperts pool(workers);
            for(const auto& row:weights)pool.run(f,x.data(),{row.data()});
            const auto start=std::chrono::steady_clock::now();
            for(int repeat=0;repeat<128;++repeat)pool.run(f,x.data(),{weights[size_t(repeat)%weights.size()].data()});
            const auto us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/128;
            if(pool.run(f,x.data(),{weights[0].data()})!=expected)throw std::runtime_error("cold CPU benchmark changed rows");
            std::printf("cold CPU %d/%d %d workers: %.3f us, %.3f logical GB/s\n",pair.first,pair.second,workers,us,f.bytes/us/1000.);
        }
    }
    return 0;
}
static int iq2_exact_bench() {
    if(!cpu::cpu_avx512_ok()) {std::puts("AVX-512 VNNI unavailable");return 0;}
    std::mt19937 rng(924);
    cpu::NativeFmt f;std::string error;
    if(!cpu::native_fmt(19,16,5120,2304,f,error,{8192,4096}))throw std::runtime_error(error);
    const auto weights=blob(f,rng);
    constexpr int max_tokens=6;
    std::vector<float> input(f.n_ff);
    std::vector<uint8_t> q(f.h_bytes*max_tokens);
    std::vector<float> reference(size_t(f.n_embd)*max_tokens),candidate(reference.size());
    const void* rows[max_tokens];
    float* output[max_tokens];
    for(int t=0;t<max_tokens;++t) {
        for(auto& x:input)x=t==5?0.f:float(int(rng()>>16)-32768)*0x1p-14f;
        auto* dst=q.data()+size_t(t)*f.h_bytes;cpu::native_quant_h(f,input.data(),dst);rows[t]=dst;
    }
    auto& flag=basalt::core::diagnostics().exp["ds-cpu-iq2-down-exact"];
    const auto run=[&](int nt,bool enabled) {
        auto* values=(enabled?candidate:reference).data();
        for(int t=0;t<nt;++t)output[t]=values+size_t(t)*f.n_embd;
        cpu::native_down_rows(f,weights.data(),rows,nt,output,0,int(f.n_embd));
    };
    for(int nt:{1,2,3,6}) {
        flag=0;run(nt,false);flag=1;run(nt,true);
        if(std::memcmp(reference.data(),candidate.data(),size_t(nt)*f.n_embd*sizeof(float))!=0)
            throw std::runtime_error("IQ2_XXS AVX-512 changed ggml output bits");
        for(int enabled:{0,1,1,0}) {
            flag=enabled;run(nt,enabled!=0);const auto start=std::chrono::steady_clock::now();
            for(int i=0;i<16;++i)run(nt,enabled!=0);
            const auto us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/16;
            std::printf("IQ2_XXS down width=%d AVX512=%d %.3f us per expert, exact\n",nt,enabled,us);
        }
    }
    flag=0;
    return 0;
}
static int draft_window_test() {
    using basalt::deepseek41::DraftWindow;
    DraftWindow policy;
    const std::array<float,5> high={.99f,.99f,.99f,.99f,.99f};
    const std::array<float,5> low={.1f,.1f,.1f,.1f,.1f},medium={.8f,.8f,.8f,.8f,.8f};
    for(int limit=0;limit<=5;++limit)if(policy.choose(high,limit)!=limit)throw std::runtime_error("high confidence limit");
    if(policy.choose(low,5)!=1 || policy.choose(medium,5)!=4)throw std::runtime_error("cumulative draft bound");
    const std::array<float,5> rejected={.99f,.1f,.1f,.1f,.1f};
    policy.observe(rejected,5,0);
    if(policy.seen[DraftWindow::bin(.99f)]!=1 || policy.seen[DraftWindow::bin(.1f)]!=0)
        throw std::runtime_error("observed a draw after rejection");
    for(int n=0;n<32;++n)policy.observe(high,5,0);
    if(policy.choose(high,5)!=1)throw std::runtime_error("failed to shorten inaccurate drafts");
    policy={};
    for(int n=0;n<32;++n)policy.observe(low,5,5);
    if(policy.choose(low,5)<=1)throw std::runtime_error("failed to calibrate underestimated confidence");
    policy={};
    if(policy.choose(high,5)!=5)throw std::runtime_error("request reset retained observations");
    for(int bad:{-1,6}) {
        bool rejected_input=false;
        try {policy.observe(high,5,bad);}catch(const std::invalid_argument&){rejected_input=true;}
        if(!rejected_input)throw std::runtime_error("invalid acceptance count accepted");
    }
    std::printf("draft window: limits, calibration, first-rejection masking and reset pass\n");return 0;
}
int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--cold")return cold_bench();
        if(argc==2 && std::string(argv[1])=="--iq2-down-exact")return iq2_exact_bench();
        if(argc==2 && std::string(argv[1])=="--draft-window")return draft_window_test();
        if(argc==2 && std::string(argv[1])=="--group-pipeline")basalt::core::diagnostics().exp["ds-cpu-group-pipeline"]=1;
        else if(argc==2 && std::string(argv[1])=="--group-pipeline-scratch") {
            basalt::core::diagnostics().exp["ds-cpu-group-pipeline"]=1;
            basalt::core::diagnostics().exp["ds-cpu-reuse-scratch"]=1;
        }
        else if(argc!=1)return 2;
        std::mt19937 rng(417);std::normal_distribution<float> nd;
        std::vector<float> x(5120);for(auto& v:x)v=nd(rng);
        for (auto pair : {std::pair<int,int>{19,19},{19,16},{39,39}}) {
            cpu::NativeFmt f;std::string error;
            if(!cpu::native_fmt(pair.first,pair.second,5120,2304,f,error,{8192,4096}))throw std::runtime_error(error);
            f.swiglu_limit=10;
            const auto a=blob(f,rng),b=blob(f,rng),c=blob(f,rng);
            const std::vector<const uint8_t*> pointers={b.data(),a.data(),b.data()};
            const auto wanted=serial(f,x,pointers);
            uint64_t hash=14695981039346656037ull;
            const auto* bytes=reinterpret_cast<const uint8_t*>(wanted.data());
            for(size_t j=0;j<wanted.size()*sizeof(float);++j)hash=(hash^bytes[j])*1099511628211ull;
            std::printf("CPU %d/%d frozen output hash %016llx\n",pair.first,pair.second,(unsigned long long)hash);
            for(int workers:{1,4,15}) {
                basalt::deepseek41::CpuExperts pool(workers);
                auto got=pool.run(f,x.data(),pointers);
                if(got!=wanted)throw std::runtime_error("CPU row scheduling changed expert output");
                for(float v:got)if(!std::isfinite(v))throw std::runtime_error("nonfinite expert result");
                if(!pool.run(f,x.data(),{}).empty())throw std::runtime_error("empty CPU job returned rows");
                if(pool.run(f,x.data(),pointers)!=wanted)throw std::runtime_error("reused CPU pool changed output");
                std::vector<float> window;
                for(int t=0;t<11;++t)for(float v:x)window.push_back(v*(1.f+t*.01f));
                std::vector<basalt::deepseek41::CpuExperts::Job> jobs={{b.data(),1},{a.data(),0},{b.data(),0},{a.data(),1}};
                for(int t=2;t<11;++t)jobs.push_back({b.data(),t});
                const auto got_window=pool.run_window(f,window.data(),11,jobs);
                for(size_t j=0;j<jobs.size();++j) {
                    const std::vector<float> input(window.begin()+jobs[j].token*5120,window.begin()+(jobs[j].token+1)*5120);
                    const auto row=serial(f,input,{jobs[j].blob});
                    if(!std::equal(row.begin(),row.end(),got_window.begin()+j*5120))throw std::runtime_error("window CPU expert grouping changed rows");
                }
                std::printf("CPU %d/%d, 5120/2304, %d workers: byte-identical to serial native rows\n",pair.first,pair.second,workers);
                const auto started=std::chrono::steady_clock::now();
                for(int repeat=0;repeat<8;++repeat)pool.run(f,x.data(),pointers);
                const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-started).count()/8;
                std::printf("micro CPU %d/%d %d workers: %.3f us, %.3f logical GB/s\n",pair.first,pair.second,workers,us,
                            3.*f.bytes/us/1000.);
                if(workers==15)for(int count:{1,2,3}) {
                    std::vector<const uint8_t*> entries={a.data(),b.data(),c.data()};entries.resize(count);
                    const auto reference=serial(f,x,entries);
                    for(int repeat=0;repeat<3;++repeat)
                        if(pool.run(f,x.data(),entries)!=reference)throw std::runtime_error("small CPU batch changed rows");
                    const auto begin=std::chrono::steady_clock::now();
                    for(int repeat=0;repeat<64;++repeat)pool.run(f,x.data(),entries);
                    const double elapsed=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count()/64;
                    if(pool.run(f,x.data(),entries)!=reference)throw std::runtime_error("repeated CPU batch changed rows");
                    std::printf("micro small CPU %d/%d %d experts: %.3f us, %.3f logical GB/s\n",pair.first,pair.second,count,
                                elapsed,count*double(f.bytes)/elapsed/1000.);
                }
            }
            std::vector<float> inputs,reference;
            for(int t=0;t<12;++t) {
                std::vector<float> row(x);
                for(auto& v:row)v=t==11?0.f:v*(1.f+t*.01f);
                inputs.insert(inputs.end(),row.begin(),row.end());
                const auto expected=serial(f,row,{b.data()});reference.insert(reference.end(),expected.begin(),expected.end());
            }
            basalt::deepseek41::CpuExperts window_pool(4);
            for(int width=1;width<=12;++width) {
                std::vector<basalt::deepseek41::CpuExperts::Job> jobs;
                for(int t=0;t<width;++t)jobs.push_back({b.data(),t});
                const auto result=window_pool.run_window(f,inputs.data(),width,jobs);
                if(!std::equal(result.begin(),result.end(),reference.begin()))throw std::runtime_error("CPU window width changed rows");
            }
            if(basalt::core::exp_flag("ds-cpu-reuse-scratch")!=0) {
                using Job=basalt::deepseek41::CpuExperts::Job;
                const auto sparse=window_pool.run_window(f,inputs.data(),6,{Job{b.data(),5},Job{b.data(),0}});
                if(sparse.size()!=10240 || !std::equal(sparse.begin(),sparse.begin()+5120,reference.begin()+5*5120) ||
                   !std::equal(sparse.begin()+5120,sparse.end(),reference.begin()))
                    throw std::runtime_error("sparse activation reuse changed expert rows");
                const auto zero=window_pool.run_window(f,inputs.data()+11*5120,1,{Job{b.data(),0}});
                if(zero.size()!=5120 || !std::equal(zero.begin(),zero.end(),reference.begin()+11*5120))
                    throw std::runtime_error("reused expert scratch changed zero activation");
            }
            std::printf("CPU %d/%d: windows 1..12 and zero activation match serial rows\n",pair.first,pair.second);
        }
        std::printf("deepseek41_cpu_test: 0 failures\n");return 0;
    } catch(const std::exception& e){std::printf("deepseek41_cpu_test: %s\n",e.what());return 1;}
}
