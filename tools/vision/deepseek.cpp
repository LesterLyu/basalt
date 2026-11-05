#include "deepseek_tower.hpp"
#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/vision/preprocess.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <bit>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>

namespace {
using namespace basalt;
using namespace basalt::vision;

uint16_t bf16(float f) {
    const auto u=std::bit_cast<uint32_t>(f);
    return uint16_t((u+0x7fffu+((u>>16)&1u))>>16);
}
struct Projector {
    GgufFile file;
    std::map<std::string,std::vector<std::string>> names;
    VisionGeo geo;
    explicit Projector(const std::string& path):file(path) {
        const auto* type=file.get("clip.projector_type");
        if(!type || type->s!="deepseek41v")throw std::runtime_error("expected deepseek41v projector");
        geo.layers=32;
        for(const auto& [key,value]:std::map<std::string,int>{
            {"block_count",32},{"embedding_length",1024},{"attention.head_count",16},
            {"feed_forward_length",2816},{"patch_size",14},{"projector.scale_factor",3},{"projection_dim",5120}}) {
            const auto* v=file.get("clip.vision."+key);
            if(!v || v->u!=uint64_t(value))throw std::runtime_error("invalid projector geometry: "+key);
        }
        names={{"aligner.w1.weight",{"mm.1.weight"}},{"aligner.w1.bias",{"mm.1.bias"}},
               {"aligner.w2.weight",{"mm.2.weight"}},{"aligner.w2.bias",{"mm.2.bias"}},
               {"vision.norm.weight",{"v.post_ln.weight"}},{"vision.patch_embed.proj.weight",{"v.patch_embd.weight"}},
               {"vision.patch_embed.proj.bias",{"v.patch_embd.bias"}}};
        for(int i=0;i<geo.layers;++i) {
            const std::string a="vision.blocks."+std::to_string(i)+".",b="v.blk."+std::to_string(i)+".";
            names[a+"norm1.weight"]={b+"ln1.weight"};names[a+"norm2.weight"]={b+"ln2.weight"};
            names[a+"attn.wqkv.weight"]={b+"attn_qkv.weight"};names[a+"attn.wqkv.bias"]={b+"attn_qkv.bias"};
            names[a+"attn.wo.weight"]={b+"attn_out.weight"};names[a+"attn.wo.bias"]={b+"attn_out.bias"};
            names[a+"mlp.w1.weight"]={b+"ffn_gate.weight",b+"ffn_up.weight"};
            names[a+"mlp.w2.weight"]={b+"ffn_down.weight"};
        }
    }
    bool read(const std::string& name,std::vector<int64_t>& shape,void* out) const {
        const auto it=names.find(name);
        const std::vector<std::string> parts=it==names.end()?std::vector<std::string>{name}:it->second;
        shape.clear();
        auto* dst=static_cast<uint16_t*>(out);
        for(const auto& part:parts) {
            const auto* t=file.find(part);
            if(!t || (t->type!=0 && t->type!=1 && t->type!=30))return false;
            const auto n=t->elements();
            const uint64_t bytes=n*(t->type==0?4:2),length=file.file_size()-file.data_start();
            if(!n || t->offset>length || bytes>length-t->offset)return false;
            std::vector<int64_t> dims(t->shape.rbegin(),t->shape.rend());
            if(part=="v.patch_embd.weight") {
                if(dims!=std::vector<int64_t>{1024,3,14,14})return false;
                dims={1024,588};
            }
            if(shape.empty())shape=dims;
            else {
                if(shape.size()!=dims.size() || !std::equal(shape.begin()+1,shape.end(),dims.begin()+1))return false;
                shape[0]+=dims[0];
            }
            if(dst) {
                const auto* raw=file.tensor_data(*t);
                if(t->type==30)std::memcpy(dst,raw,size_t(bytes));
                else for(uint64_t i=0;i<n;++i) {
                    float f;
                    if(t->type==0)std::memcpy(&f,raw+4*i,4);
                    else {
                        __half_raw h;
                        std::memcpy(&h.x,raw+2*i,2);
                        f=__half2float(__half(h));
                    }
                    dst[i]=bf16(f);
                }
                dst+=n;
            }
        }
        return !shape.empty();
    }
};

int run(int argc,char** argv) {
    if(argc!=2 && (argc!=4 || std::string_view(argv[2])!="--max-tokens"))
        throw std::runtime_error("usage: basalt-deepseek-vision MMPROJ.gguf [--max-tokens N]");
    Projector projector(argv[1]);
    if(argc==4)projector.geo.max_tokens=std::stoi(argv[3]);
    VisionTower tower;
    std::string err;
    if(!tower.load(projector.geo,0,[&](const auto& name,auto& shape,void* dst){
        return projector.read(name,shape,dst);
    },err))throw std::runtime_error(err);
    std::vector<uint16_t> marks[3];
    const char* names[]={"v.token_embd.img_start","v.image_newline","v.token_embd.img_end"};
    for(int i=0;i<3;++i) {
        std::vector<int64_t> shape;
        if(!projector.read(names[i],shape,nullptr) || shape!=std::vector<int64_t>{5120})
            throw std::runtime_error("missing projector image marker");
        marks[i].resize(5120);
        if(!projector.read(names[i],shape,marks[i].data()))throw std::runtime_error("image marker read failed");
    }
    std::cout<<"READY 5120 "<<tower.room_bytes()<<" mem-events"<<std::endl;
    for(std::string line;std::getline(std::cin,line) && line!="QUIT";) {
        try {
            const bool stats=line.rfind("ENCSTAT ",0)==0;
            const size_t begin=stats?8:4;
            const auto split=line.find('\t',begin);
            if((!stats && line.rfind("ENC ",0)!=0) || split==std::string::npos)
                throw std::runtime_error("expected ENC image<TAB>output");
            const auto input=std::filesystem::path(std::u8string(line.begin()+begin,line.begin()+split));
            const auto output=std::filesystem::path(std::u8string(line.begin()+split+1,line.end()));
            std::ifstream in(input,std::ios::binary|std::ios::ate);
            const auto bytes=in.tellg();
            if(bytes<=0 || bytes>64*1024*1024)throw std::runtime_error("invalid image file size");
            std::vector<uint8_t> data(static_cast<size_t>(bytes));
            in.seekg(0);
            if(!in.read(reinterpret_cast<char*>(data.data()),bytes))throw std::runtime_error("short image read");
            Image image;
            if(!decode_image(data.data(),data.size(),image,err))throw std::runtime_error(err);
            if(image.w%14 || image.h%14)throw std::runtime_error("DeepSeek image must be patch-aligned");
            int nh=0,nw=0;
            std::vector<uint16_t> rows;
            {
                uint8_t* gpu=nullptr;
                const auto alloc=cudaMalloc(&gpu,tower.room_bytes());
                if(alloc!=cudaSuccess)
                    throw std::runtime_error(std::string("vision workspace: ")+cudaGetErrorString(alloc));
                const std::unique_ptr<uint8_t,decltype([](uint8_t* p){cudaFree(p);})> room(gpu);
                if(stats)std::cout<<"MEM "<<tower.room_bytes()<<std::endl;
                if(!tower.encode(image.rgb.data(),image.w,image.h,room.get(),rows,nh,nw,err))
                    throw std::runtime_error(err);
            }
            const uint32_t count=uint32_t(nh*(nw+1)+2);
            const uint32_t header[]={0x31495644,count,5120};
            std::ofstream out(output,std::ios::binary);
            auto put=[&](const uint16_t* row,size_t n){out.write(reinterpret_cast<const char*>(row),n*2);};
            out.write(reinterpret_cast<const char*>(header),sizeof(header));
            put(marks[0].data(),5120);
            for(int y=0;y<nh;++y) {
                put(rows.data()+size_t(y)*nw*5120,size_t(nw)*5120);
                put(marks[1].data(),5120);
            }
            put(marks[2].data(),5120);
            out.close();
            if(!out)throw std::runtime_error("embedding write failed");
            std::cout<<"OK "<<count<<" "<<nw<<" "<<nh<<std::endl;
        } catch(const std::exception& e) {std::cout<<"ERR "<<e.what()<<std::endl;}
    }
    return 0;
}
}
int main(int argc,char** argv) {
    try{return run(argc,argv);}
    catch(const std::exception& e){std::cout<<"ERR "<<e.what()<<std::endl;return 1;}
}
