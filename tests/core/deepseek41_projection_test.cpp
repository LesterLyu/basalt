#include "basalt/deepseek41/device_model.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>

namespace ds=basalt::deepseek41;
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static float rounded(float value,bool bf16) {
    if(!bf16)return ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
    uint32_t bits;std::memcpy(&bits,&value,4);bits+=0x7fff+((bits>>16)&1);bits&=0xffff0000u;
    std::memcpy(&value,&bits,4);return value;
}
static void check(ds::PackedWeights& weights,ds::DeviceModel& d,const char* name,std::mt19937& rng,bool strided=false) {
    const auto& full=d.tensor(name);const int in=int(full.shape[0]);
    const int first_row=strided?7*1024:0,n=strided?1024:int(full.shape[1]),tokens=33;
    const size_t row_bytes=full.bytes/full.shape[1];
    const ds::DeviceTensor matrix{full.data+size_t(first_row)*row_bytes,full.type,{uint64_t(in),uint64_t(n)},size_t(n)*row_bytes};
    const int xs=strided?32768:in,ys=strided?8192:n,xoff=strided?7*4096:0,yoff=strided?7*1024:0;
    std::normal_distribution<float> normal;std::vector<float> x(size_t(tokens)*xs),y(size_t(tokens)*ys,123.f);
    for(auto& v:x)v=normal(rng);
    auto* dx=d.alloc<float>(x.size());auto* dy=d.alloc<float>(y.size());
    d.upload(dx,x.data(),x.size()*4);d.upload(dy,y.data(),y.size()*4);
    d.set_prefill_tensor(true);d.linear(matrix,dx+xoff,dy+yoff,tokens,xs,ys);d.read(y.data(),dy,y.size()*4);
    const auto source=weights.target_tensor(name);const auto* traits=ggml_get_type_traits(ggml_type(full.type));
    std::vector<float> row(in);double maximum=0,relative=0;
    for(int r:{0,17,n/2,n-1}) {
        const uint8_t* raw=source.data()+size_t(first_row+r)*row_bytes;
        if(full.type==0)std::memcpy(row.data(),raw,in*4);
        else if(full.type==30)for(int c=0;c<in;++c){uint16_t h;std::memcpy(&h,raw+c*2,2);uint32_t bits=uint32_t(h)<<16;std::memcpy(&row[c],&bits,4);}
        else traits->to_float(raw,row.data(),in);
        for(int t=0;t<tokens;++t) {
            double ref=0,l1=0;
            for(int c=0;c<in;++c) {
                float a=x[size_t(t)*xs+xoff+c],b=row[c];
                if(full.type!=0){a=rounded(a,full.type==30);if(full.type!=30)b=rounded(b,false);}
                ref+=double(a)*b;l1+=std::abs(double(a)*b);
            }
            const double error=std::abs(y[size_t(t)*ys+yoff+r]-ref);
            maximum=std::max(maximum,error);relative=std::max(relative,error/std::max(l1,1e-12));
            if(error>1e-5+3e-6*l1)throw std::runtime_error("projection violates CPU-double bound");
        }
    }
    if(strided)for(int t=0;t<tokens;++t)for(int r=0;r<yoff;++r)
        if(y[size_t(t)*ys+r]!=123.f)throw std::runtime_error("strided GEMM overwrote another group");
    std::printf("%s type=%u N=%d K=%d T=%d strided=%d max_abs=%.7g L1_rel=%.7g\n",name,full.type,n,in,tokens,strided,maximum,relative);std::fflush(stdout);
}
static void grouped(ds::PackedWeights& weights,ds::DeviceModel& d,const std::string& name,std::mt19937& rng,bool small=false,int tokens=1) {
    const auto& full=d.tensor(name);const int groups=8,in=small?512:int(full.shape[0]),out=small?17:1024;
    const size_t row_bytes=basalt::kernels::native_mmvq_weight_bytes(int(full.type),in,1);
    const ds::DeviceTensor matrix{full.data,full.type,{uint64_t(in),uint64_t(out*groups)},row_bytes*out*groups};
    std::normal_distribution<float> normal;std::vector<float> x(size_t(groups)*tokens*in),reference(size_t(groups)*tokens*out),actual(reference.size());
    for(auto& v:x)v=normal(rng);
    auto* dx=d.alloc<float>(x.size());auto* dy=d.alloc<float>(reference.size());
    auto* quant=d.allocate(size_t(in/32)*36*groups*tokens,ds::MemoryKind::scratch);
    d.upload(dx,x.data(),x.size()*4);d.set_prefill_tensor(false);
    const auto serial=[&] {
        for(int g=0;g<groups;++g) {
            const ds::DeviceTensor slice{matrix.data+size_t(g)*out*row_bytes,matrix.type,{uint64_t(in),uint64_t(out)},out*row_bytes};
            d.linear(slice,dx+size_t(g)*in,dy+size_t(g)*out,tokens,size_t(groups)*in,size_t(groups)*out);
        }
    };
    serial();d.read(reference.data(),dy,reference.size()*4);
    d.linear_groups(matrix,dx,dy,groups,tokens);d.read(actual.data(),dy,actual.size()*4);
    if(std::memcmp(reference.data(),actual.data(),actual.size()*4))throw std::runtime_error("grouped projection changed output bits");
    basalt::kernels::quantize_q8_1_rows(dx,groups*tokens,in,quant,d.stream());
    std::vector<uint8_t> q(size_t(in/32)*36*groups*tokens);d.read(q.data(),quant,q.size());
    const auto source=weights.target_tensor(name.c_str());const auto* traits=ggml_get_type_traits(ggml_type(full.type));
    std::vector<float> row(in);double maximum=0,relative=0;
    for(int t=0;t<tokens;++t)for(int g=0;g<groups;++g)for(int r:{0,out/2,out-1}) {
        traits->to_float(source.data()+size_t(g*out+r)*row_bytes,row.data(),in);
        double ref=0,l1=0;
        for(int k=0;k<in;++k) {
            const auto* block=q.data()+(size_t(t*groups+g)*in/32+k/32)*36;uint16_t h;std::memcpy(&h,block,2);
            const double value=double(ggml_fp16_to_fp32(h))*int8_t(block[4+k%32])*row[k];
            ref+=value;l1+=std::abs(value);
        }
        const double error=std::abs(actual[(t*groups+g)*out+r]-ref);maximum=std::max(maximum,error);relative=std::max(relative,error/std::max(l1,1e-12));
        if(error>1e-5+3e-6*l1)throw std::runtime_error("grouped projection violates CPU-double bound");
    }
    cudaEvent_t start,end;ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));
    float times[2]{};
    for(int mode=0;mode<2;++mode) {
        for(int i=0;i<10;++i)if(mode)d.linear_groups(matrix,dx,dy,groups,tokens);else serial();
        ck(cudaEventRecord(start,static_cast<cudaStream_t>(d.stream())));
        for(int i=0;i<100;++i)if(mode)d.linear_groups(matrix,dx,dy,groups,tokens);else serial();
        ck(cudaEventRecord(end,static_cast<cudaStream_t>(d.stream())));ck(cudaEventSynchronize(end));
        ck(cudaEventElapsedTime(&times[mode],start,end));times[mode]*=10;
    }
    ck(cudaEventDestroy(start));ck(cudaEventDestroy(end));
    std::printf("grouped type=%u N=%d K=%d groups=%d tokens=%d bitwise max_abs=%.7g L1_rel=%.7g serial_us=%.3f grouped_us=%.3f grouped_GB_s=%.2f\n",
                full.type,out,in,groups,tokens,maximum,relative,times[0],times[1],matrix.bytes/(times[1]*1e3));std::fflush(stdout);
}
int main(int argc,char** argv) {
    if(argc!=2 && argc!=3)return 2;
    try {
        const int device=argc==3?std::stoi(argv[2]):0;
        ck(cudaSetDevice(device));ds::PackedWeights weights(argv[1]);ds::DeviceModel d(weights,false,device,33);
        d.prepare_prefill();std::mt19937 rng(417);
        check(weights,d,"blk.0.attn_q_a.weight",rng);
        check(weights,d,"blk.0.attn_q_b.weight",rng);
        check(weights,d,"blk.0.attn_output_a.weight",rng,true);
        check(weights,d,"blk.1.engram_kv.weight",rng);
        check(weights,d,"blk.0.ffn_down_shexp.weight",rng);
        bool checked[2]{};
        for(int l=0;l<40;++l) {
            const std::string name="blk."+std::to_string(l)+".attn_output_a.weight";const auto type=d.tensor(name).type;
            if((type==13 || type==14) && !checked[type-13]) {for(int tokens=1;tokens<=6;++tokens) {grouped(weights,d,name,rng,false,tokens);grouped(weights,d,name,rng,true,tokens);}checked[type-13]=true;}
        }
        if(!checked[0] || !checked[1])throw std::runtime_error("missing grouped projection format coverage");
        std::printf("deepseek41_projection_test: 0 failures\n");return 0;
    }catch(const std::exception& e){std::printf("deepseek41_projection_test: %s\n",e.what());return 1;}
}
