// Mixed-format native experts: real sidecar bytes, CPU serial/batched parity, GPU placement and unequal-size swaps.
#include "basalt/core/model_file.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "models/deepseek41/expert_ram.hpp"
#include "models/deepseek41/prefill_native.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace cpu = basalt::kernels::cpu;
namespace kg = basalt::kernels;
constexpr int H = 5120, FF = 2304, TMAX = 8, K = 6;
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
void ck(cudaError_t e) { require(e == cudaSuccess, cudaGetErrorString(e)); }
struct Host {
    void* p = nullptr;
    explicit Host(size_t n) { ck(cudaHostAlloc(&p,n,cudaHostAllocPortable | cudaHostAllocMapped)); }
    ~Host() { cudaFreeHost(p); }
};
struct Expert {
    kg::NativeExpertLayout gpu;
    cpu::NativeFmt cpu;
    std::vector<uint8_t> bytes;
};

int main(int argc, char** argv) try {
    if (argc != 2) { std::puts("usage: ds41_mixed_expert_test OVERLAY.basalt"); return 77; }
    int devices = 0; ck(cudaGetDeviceCount(&devices));
    if (devices < 2) return 77;
    basalt::core::ModelFile file;
    std::string err;
    require(file.open(argv[1],err),err.c_str());
    std::vector<uint8_t> layout;
    require(file.read("overlay.layout",layout,err),err.c_str());
    std::istringstream lines(std::string(layout.begin(),layout.end()));
    std::string magic; lines >> magic;
    require(magic == "BASALT_DS_EXPERTS_V1","overlay header");
    std::vector<Expert> experts;
    std::map<std::pair<int,int>,bool> seen;
    int l,e,gu,d; uint64_t off,n;
    while (lines >> l >> e >> gu >> d >> off >> n) {
        if (!seen.emplace(std::make_pair(gu,d),true).second) continue;
        Expert v;
        v.gpu = kg::native_expert_layout(gu,d,H,FF,10.f);
        require(cpu::native_fmt(gu,d,H,FF,v.cpu,err,10.f),err.c_str());
        require(n == v.gpu.bytes && n == v.cpu.bytes,"expert bytes");
        v.bytes.resize((size_t)n);
        require(basalt::core::read_range_at(file.range("experts"),off,v.bytes.data(),n,err),err.c_str());
        experts.push_back(std::move(v));
    }
    const int E = (int)experts.size();
    require(E > 1,"need mixed formats");
    size_t largest = 0;
    for (const auto& v:experts) largest = std::max(largest,v.bytes.size());
    Host input(TMAX*H*sizeof(float)), rows0(TMAX*K*H*sizeof(float)), rows1(TMAX*K*H*sizeof(float));
    auto* x = (float*)input.p;
    std::mt19937 rng(192);
    std::normal_distribution<float> normal(0.f,0.5f);
    for (int i=0;i<TMAX*H;++i) x[i]=normal(rng);
    cpu::ExpertPool pool(15,cpu::PoolAffinity::All);
    require(pool.placement_ok(),"CPU placement");
    for (int T : {1,2,4,8}) {
        std::vector<cpu::ExpertJobMulti> jobs(E);
        std::vector<uint8_t> acts((size_t)E*T*cpu::kNativeActBytes);
        std::vector<float> got((size_t)E*T*H), want(got.size()), ff((size_t)T*FF);
        std::vector<uint8_t> hq((size_t)T*cpu::kNativeHBytes);
        for (int i=0;i<E;++i) {
            const auto& v=experts[i]; auto& j=jobs[i];
            j.native_format=&v.cpu; j.blob=v.bytes.data(); j.nt=T;
            float* fp[TMAX]; const void* hp[TMAX]; float* op[TMAX];
            for (int t=0;t<T;++t) {
                auto* a=acts.data()+((size_t)i*T+t)*cpu::kNativeActBytes;
                cpu::native_quant_act(v.cpu,x+(size_t)t*H,a);
                j.nact[t]=a; j.out[t]=got.data()+((size_t)i*T+t)*H;
                fp[t]=ff.data()+(size_t)t*FF; hp[t]=hq.data()+(size_t)t*cpu::kNativeHBytes;
                op[t]=want.data()+((size_t)i*T+t)*H;
            }
            cpu::native_gu_rows(v.cpu,j.blob,j.nact,T,fp,0,FF);
            for (int t=0;t<T;++t) cpu::native_quant_h(v.cpu,fp[t],(void*)hp[t]);
            cpu::native_down_rows(v.cpu,j.blob,hp,T,op,0,H);
        }
        pool.start_split_multi_native(experts.front().cpu,jobs.data(),E);
        pool.finish_split_multi_native();
        require(!std::memcmp(got.data(),want.data(),got.size()*sizeof(float)),"mixed CPU vs serial output");
        for (float f:got) require(std::isfinite(f),"finite CPU output");
        std::printf("CPU: %d formats x %d rows, bitwise equal to serial native kernels\n",E,T);
    }
    basalt::core::PeerExperts primary, peer;
    require(primary.open_plain(0,1,E,H,FF,(float*)rows0.p,err),err.c_str());
    require(peer.open_plain(1,1,E,H,FF,(float*)rows1.p,err),err.c_str());
    require(primary.open_slots(std::vector<int64_t>(E,largest),err),err.c_str());
    require(peer.open_slots(std::vector<int64_t>(E,largest),err),err.c_str());
    require(!peer.launch_rows(0,experts[0].gpu,nullptr,nullptr,peer.kRowBatch+1,nullptr,err),"reject oversized row batch");
    require(!peer.launch_rows(0,experts[0].gpu,nullptr,nullptr,-1,nullptr,err),"reject negative row batch");
    auto bad = experts[0].gpu;
    ++bad.n_ff;
    const kg::NativeExpertLayout* bad_layout[] = {&bad};
    const int32_t first_id[] = {0}, first_kind[] = {2};
    require(!peer.launch_window(0,experts[0].gpu,nullptr,first_id,1,1,first_kind,err,bad_layout),"reject incompatible geometry");
    Host stage(largest), readback(largest+64);
    for (int i=0;i<E;++i) {
        std::memcpy(stage.p,experts[i].bytes.data(),experts[i].bytes.size());
        require(primary.fill_slot(i,(uint8_t*)stage.p,experts[i].bytes.size()),"primary fill");
        require(peer.fill_slot(i,(uint8_t*)stage.p,experts[i].bytes.size()),"peer fill");
        primary.admit(0,i,i); peer.admit(0,i,i);
    }
    float *dx = nullptr, *dy = nullptr;
    void *xq = nullptr, *scratch = nullptr;
    ck(cudaSetDevice(0));
    ck(cudaMalloc((void**)&dx,H*sizeof(float)));
    ck(cudaMalloc((void**)&dy,H*sizeof(float)));
    ck(cudaMalloc(&xq,(size_t)H/32*36));
    ck(cudaMalloc(&scratch,kg::gated_mlp_scratch_bytes(1,FF)));
    for (int swapped=0;swapped<2;++swapped) {
        if (swapped) {
            for (int i=0;i<E;++i) peer.admit(0,i,-1);
            for (int i=0;i<E;++i) {
                const int in=(i+1)%E;
                std::memcpy(stage.p,experts[in].bytes.data(),experts[in].bytes.size());
                std::memset(readback.p,0x5a,largest+64);
                peer.exchange_slot(i,(uint8_t*)readback.p,experts[i].bytes.size(),(uint8_t*)stage.p,experts[in].bytes.size());
                require(peer.read_wait(err),err.c_str());
                require(!std::memcmp(readback.p,experts[i].bytes.data(),experts[i].bytes.size()),"evicted bytes changed");
                for (size_t z=experts[i].bytes.size();z<largest+64;++z)
                    require(((uint8_t*)readback.p)[z]==0x5a,"eviction overrun");
                peer.admit(0,in,i);
            }
        }
        for (int T:{1,2,4,8}) {
            std::vector<int32_t> ids(T*K),kind(T*K,2),mask(T*K);
            std::vector<const kg::NativeExpertLayout*> formats(T*K);
            for (int r=0;r<T*K;++r) { ids[r]=(r*3+r/K)%E; formats[r]=&experts[ids[r]].gpu; }
            // The established homogeneous path, called once per format, is the arithmetic reference.
            for (int i=0;i<E;++i) {
                for (int r=0;r<T*K;++r) mask[r]=ids[r]==i?2:0;
                require(primary.launch_window(0,experts[i].gpu,x,ids.data(),T,K,mask.data(),err),err.c_str());
                require(primary.finish(nullptr,err),err.c_str());
            }
            std::vector<float> reference((float*)rows0.p,(float*)rows0.p+(size_t)T*K*H);
            // The primary's replacement path uses one native gated MLP per row.
            for (int r=0;r<T*K;++r) {
                ck(cudaMemcpy(dx,x+(size_t)(r/K)*H,H*sizeof(float),cudaMemcpyHostToDevice));
                kg::quantize_q8_1_rows(dx,1,H,xq,nullptr);
                kg::gated_mlp_clamped(experts[ids[r]].gpu,10.f,primary.slot_ptr(0,ids[r]),1,1,xq,scratch,dy,nullptr);
                ck(cudaMemcpy((float*)rows0.p+(size_t)r*H,dy,H*sizeof(float),cudaMemcpyDeviceToHost));
            }
            require(!std::memcmp(rows0.p,reference.data(),reference.size()*sizeof(float)),"primary row vs grouped output");
            require(primary.launch_window(0,experts[0].gpu,x,ids.data(),T,K,kind.data(),err,formats.data()),err.c_str());
            require(primary.finish(nullptr,err),err.c_str());
            const auto start=std::chrono::steady_clock::now();
            require(peer.launch_window(0,experts[0].gpu,x,ids.data(),T,K,kind.data(),err,formats.data()),err.c_str());
            require(peer.finish(nullptr,err),err.c_str());
            const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
            require(!std::memcmp(rows0.p,reference.data(),reference.size()*sizeof(float)),"mixed primary output");
            require(!std::memcmp(rows1.p,reference.data(),reference.size()*sizeof(float)),"mixed helper output");
            std::printf("GPU: rows=%d swapped=%d, primary/helper bitwise equal; helper %.1f us\n",T,swapped,us);
        }
    }
    cudaFree(scratch); cudaFree(xq); cudaFree(dy); cudaFree(dx);
    // Long prompt batches retain the single-row result, including repeated and permuted token indices.
    constexpr int B = basalt::models::deepseek41::kNativePromptRows;
    Host batch_x((size_t)B*H*4), mapped_q(kg::native_q8_1_bytes(H)*B);
    auto* batch = (float*)batch_x.p;
    for (int i=0;i<B*H;++i) batch[i]=normal(rng);
    for (int device : {0,1}) {
        ck(cudaSetDevice(device));
        float *bx=nullptr,*by=nullptr;
        void *bq=nullptr,*bg=nullptr,*bs=nullptr,*single=nullptr;
        int32_t *bt=nullptr,*bd=nullptr;
        ck(cudaMalloc((void**)&bx,(size_t)B*H*4));
        ck(cudaMalloc((void**)&by,(size_t)B*H*4));
        ck(cudaMalloc(&bq,kg::native_q8_1_bytes(H)*B));
        ck(cudaMalloc(&bg,kg::native_q8_1_bytes(H)*B));
        ck(cudaMalloc(&bs,kg::gated_mlp_scratch_bytes(B,FF)+32));
        ck(cudaMalloc(&single,kg::gated_mlp_scratch_bytes(1,FF)));
        ck(cudaMalloc((void**)&bt,B*4)); ck(cudaMalloc((void**)&bd,B*4));
        ck(cudaMemcpy(bx,batch,(size_t)B*H*4,cudaMemcpyHostToDevice));
        kg::quantize_q8_1_rows(bx,B,H,bq,nullptr);
        ck(cudaMemcpy(mapped_q.p,bq,kg::native_q8_1_bytes(H)*B,cudaMemcpyDeviceToHost));
        for (int e=0;e<E;++e) {
            const void* blob=device ? peer.slot_ptr(0,e) : primary.slot_ptr(0,e);
            std::vector<float> reference((size_t)B*H),got(reference.size());
            std::vector<int32_t> tok(B),dst(B);
            for (int t=0;t<B;++t) {
                tok[t]=(t*7+3)%(B/2); dst[t]=(t*13)%B;
                kg::gated_mlp_clamped(experts[e].gpu,10.f,blob,1,1,
                    (uint8_t*)bq+kg::native_q8_1_bytes(H)*tok[t],single,by,nullptr);
                ck(cudaMemcpy(reference.data()+(size_t)dst[t]*H,by,H*4,cudaMemcpyDeviceToHost));
            }
            ck(cudaMemcpy(bt,tok.data(),B*4,cudaMemcpyHostToDevice));
            ck(cudaMemcpy(bd,dst.data(),B*4,cudaMemcpyHostToDevice));
            kg::gather_q8_1_rows(mapped_q.p,bt,B,H,bg,nullptr);
            std::vector<uint8_t> gathered(kg::native_q8_1_bytes(H)*B);
            ck(cudaMemcpy(gathered.data(),bg,gathered.size(),cudaMemcpyDeviceToHost));
            const size_t row_bytes=kg::native_q8_1_bytes(H);
            for (int t=0;t<B;++t)
                require(!std::memcmp(gathered.data()+t*row_bytes,(uint8_t*)mapped_q.p+tok[t]*row_bytes,row_bytes),
                        "q8 gather vs CPU byte-copy reference");
            for (bool mapped : {false,true}) {
                for (int count : {1,31,80,256}) {
                    ck(cudaMemset(by,0,(size_t)B*H*4));
                    basalt::models::deepseek41::native_prompt_rows(experts[e].gpu,10.f,blob,mapped?mapped_q.p:bq,
                        bt,bd,count,bs,by,nullptr);
                    ck(cudaMemcpy(got.data(),by,(size_t)B*H*4,cudaMemcpyDeviceToHost));
                    for (int t=0;t<count;++t)
                        require(!std::memcmp(got.data()+(size_t)dst[t]*H,reference.data()+(size_t)dst[t]*H,H*4),
                                "native prompt batch vs single-row output");
                }
            }
            if (e==0) {
                cudaEvent_t begin,end; ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));
                ck(cudaEventRecord(begin));
                for (int t=0;t<B;++t)
                    kg::gated_mlp_clamped(experts[e].gpu,10.f,blob,1,1,
                        (uint8_t*)bq+row_bytes*tok[t],single,by,nullptr);
                ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));
                float serial_ms=0,batch_ms=0;ck(cudaEventElapsedTime(&serial_ms,begin,end));
                ck(cudaEventRecord(begin));
                for (int repeat=0;repeat<10;++repeat)
                    basalt::models::deepseek41::native_prompt_rows(experts[e].gpu,10.f,blob,bq,bt,bd,B,bs,by,nullptr);
                ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));
                ck(cudaEventElapsedTime(&batch_ms,begin,end));
                const double us=batch_ms*100.;
                std::printf("prompt GPU %d, 256 rows: serial %.1f us, batch %.1f us, %.2f TFLOP/s\n",
                    device,serial_ms*1000.,us,6.*H*FF*B/(us*1e6));
                cudaEventDestroy(end);cudaEventDestroy(begin);
            }
        }
        std::printf("prompt GPU %d: %d formats, 1/31/80/256 rows, permuted device/mapped inputs: bitwise equal\n",device,E);
        cudaFree(bd);cudaFree(bt);cudaFree(single);cudaFree(bs);cudaFree(bg);cudaFree(bq);cudaFree(by);cudaFree(bx);
    }
    // GPU-exclusive homes drop the allocation and reconstruct the same bytes from VRAM.
    for (int round = 0; round < 4; ++round) for (int i = 0; i < E; ++i) {
        auto ram = std::make_unique<basalt::models::deepseek41::ExpertRam>();
        require(ram->allocate(experts[i].bytes.size()),"recreate exclusive expert RAM");
        require(!ram->ready,"RAM must not publish before the device copy");
        require(peer.read_slot(peer.slot_ptr(0,i),ram->data,experts[i].bytes.size(),err),err.c_str());
        require(peer.read_wait(err),err.c_str());
        require(!std::memcmp(ram->data,experts[i].bytes.data(),experts[i].bytes.size()),"VRAM restored bytes changed");
        ram->ready = true;
        require(ram->allocated() >= experts[i].bytes.size(),"resident accounting");
        ram.reset();
        require(!ram,"exclusive home retained its host allocation");
    }
    std::puts("ds41_mixed_expert_test: ok");
    return 0;
} catch (const std::exception& e) { std::printf("FAIL: %s\n",e.what()); return 1; }
