// The existing peer tier at both model geometries, optionally with real DSpark weights resident beside it.
#include "basalt/core/peer_experts.hpp"
#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/f16_bits.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "../../src/deepseek41/expert_tiers.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace cpu = basalt::kernels::cpu;
namespace k = basalt::kernels;
namespace core = basalt::core;
static void require(bool ok, const std::string& why) {
    if (!ok) { std::printf("FAIL: %s\n", why.c_str()); std::exit(1); }
}
static void ck(cudaError_t e) { require(e == cudaSuccess, cudaGetErrorString(e)); }
template<class T> static T* alloc(size_t n) { T* p = nullptr; ck(cudaMalloc(&p, n * sizeof(T))); return p; }

class Source : public core::ExpertSource {
public:
    int experts=3;
    std::vector<std::vector<uint8_t>> data;
    const uint8_t* blob(int64_t l, int64_t e) override { return data.at(size_t(l * experts + e)).data(); }
};

static void matrix(std::vector<uint8_t>& out, int type, int rows, int n, std::mt19937& rng) {
    const int block = type == 20 || type == 39 ? 32 : 256;
    const size_t bs = k::iq_row_bytes(type, block), begin = out.size();
    out.resize(begin + size_t(rows) * k::iq_row_bytes(type, n));
    for (size_t i = begin; i < out.size(); ++i) out[i] = uint8_t(rng());
    for (size_t i = begin; i < out.size(); i += bs) {
        if (type == 39) out[i] = uint8_t(114 + rng() % 8);
        else {
            const uint16_t d = k::f16_from_f32((rng() & 1 ? -1.f : 1.f) / 256.f);
            std::memcpy(out.data() + i, &d, 2);
        }
    }
}

static void check(int h, int ff, int gu, int down, std::mt19937& rng) {
    constexpr int nl = 2, ne = 3, nt = 5, top = 2;
    cpu::NativeFmt f;
    std::string err;
    const cpu::NativeBufferLimits limits = h == 2560 ? cpu::NativeBufferLimits{} : cpu::NativeBufferLimits{8192, 4096};
    require(cpu::native_fmt(gu, down, h, ff, f, err, limits), err);
    f.swiglu_limit = h == 5120 ? 10.f : 0.f;
    std::ostringstream text;
    text << "# experts v6: (n_expert 3)\n";
    for (int l = 0; l < nl; ++l) text << l << ' ' << gu << ' ' << down << ' ' << l * ne * f.bytes << ' ' << f.bytes << '\n';
    require(cpu::expert_layout_load_text(text.str(), nl, ne, err, {h, ff, limits, f.swiglu_limit}), err);
    require(cpu::expert_layout().n_embd == h, "layout lost its hidden dimension");
    Source src;
    src.data.resize(nl * ne);
    for (auto& b : src.data) {
        matrix(b, gu, 2 * ff, h, rng);
        matrix(b, down, h, ff, rng);
        require(b.size() == f.bytes, "blob byte count");
    }
    ck(cudaSetDevice(0));
    core::ExpertCache primary;
    require(primary.open(1, nl, ne, f.bytes, err), err);
    require(primary.admit(0, 0) >= 0, "primary admission");
    core::PeerExperts peer;
    require(peer.open(1, {{0, 0}, {0, 1}, {1, 0}, {0, 2}, {1, 1}}, primary, src, nl, ne, 1024, 2, err), err);
    require(peer.has(0, 1) && peer.has(1, 0) && !peer.has(0, 0), "ranked fill or exclusion");
    std::printf("Peer hidden=%d ff=%d %d/%d: %lld slots, physical budget %.1f MiB\n", h, ff, gu, down,
                (long long) peer.resident(), peer.available_bytes() / 1048576.0);
    {
        constexpr int tokens=29,k=6,batch=13;
        std::vector<float> input(tokens*h,.125f),serial(tokens*k*h,123.f),parallel(serial);
        std::vector<int32_t> ids(tokens*k,0),kind(tokens*k,0);
        for(int t=0;t<tokens;++t) {ids[t*k+1]=1;kind[t*k+1]=2;}
        const auto run=[&](std::vector<float>& out) {
            std::string error;
            for(int first=0;first<tokens;first+=batch) {
                require(peer.launch(0,input.data()+first*h,ids.data()+first*k,std::min(batch,tokens-first),k,kind.data()+first*k,error),error);
                require(peer.finish(out.data()+first*k*h,error),error);
            }
        };
        run(serial);
        auto work=std::async(std::launch::async,[&]{run(parallel);});
        ck(cudaSetDevice(0));auto* other=alloc<float>(h);ck(cudaMemset(other,0,h*4));ck(cudaDeviceSynchronize());
        work.get();cudaFree(other);
        require(serial==parallel,"background helper changed batch outputs or untouched rows");
    }
    std::vector<float> x(size_t(nt) * h);
    std::normal_distribution<float> nd;
    for (auto& v : x) v = nd(rng) * (h == 5120 ? 8.f : 1.f);
    std::vector<int32_t> ids(nt * top), kinds(nt * top, 0);
    for (int t = 0; t < nt; ++t) kinds[t * top + 1] = 2;
    for (int wanted : {1, 2}) {
        if (wanted == 2) {
            std::vector<float> usage(nl * ne, 0.f);
            usage[2] = 20.f;
            std::vector<int32_t> res0(nl * ne, -1);
            res0[0] = 0;
            require(peer.adapt(usage.data(), res0.data(), 1, err), err);
            peer.apply_pending(true);
            require(peer.has(0, 2) && !peer.has(0, 1), "adaptive swap did not land");
        }
        for (int t = 0; t < nt; ++t) ids[t * top + 1] = wanted;
        std::vector<float> got(size_t(nt) * top * h, 123.f), expected(got);
        require(peer.launch(0, x.data(), ids.data(), nt, top, kinds.data(), err), err);
        require(peer.finish(got.data(), err), err);
        ck(cudaSetDevice(0));
        auto* dx = alloc<float>(x.size());
        auto* q = alloc<uint8_t>(size_t(nt) * h / 32 * 36);
        auto* blob = alloc<uint8_t>(f.bytes);
        auto* scratch = alloc<uint8_t>(k::native_expert_scratch_bytes(nt, ff));
        auto* out = alloc<float>(got.size());
        auto* ptr = alloc<unsigned long long>(1);
        auto* start = alloc<int32_t>(2);
        auto* groups = alloc<int32_t>(1);
        auto* dst = alloc<int32_t>(nt);
        auto* tok = alloc<int32_t>(nt);
        const unsigned long long address = (unsigned long long) blob;
        const int32_t starts[] = {0, nt}, count = 1;
        std::vector<int32_t> d(nt), t(nt);
        for (int i = 0; i < nt; ++i) { d[i] = i * top + 1; t[i] = i; }
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(blob, src.blob(0, wanted), f.bytes, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(out, expected.data(), expected.size() * 4, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(ptr, &address, 8, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(start, starts, 8, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(groups, &count, 4, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(dst, d.data(), nt * 4, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(tok, t.data(), nt * 4, cudaMemcpyHostToDevice));
        k::quantize_q8_1_rows(dx, nt, h, q, nullptr);
        const auto layout = k::native_expert_layout(gu, down, h, ff, f.swiglu_limit);
        k::native_grouped_set_mma_hint(true);
        k::native_expert_grouped(layout, ptr, start, groups, dst, tok, 1, nt, q, scratch, out, nullptr);
        k::native_grouped_set_mma_hint(false);
        ck(cudaMemcpy(expected.data(), out, expected.size() * 4, cudaMemcpyDeviceToHost));
        require(std::memcmp(expected.data(), got.data(), got.size() * 4) == 0, "main/peer result or scatter differs");
        require(peer.copy_to(0, wanted, blob, nullptr, err), err);
        ck(cudaDeviceSynchronize());
        std::vector<uint8_t> promoted(f.bytes);
        ck(cudaMemcpy(promoted.data(), blob, f.bytes, cudaMemcpyDeviceToHost));
        require(std::memcmp(promoted.data(), src.blob(0, wanted), f.bytes) == 0, "peer promotion bytes differ");
        cudaFree(dx); cudaFree(q); cudaFree(blob); cudaFree(scratch); cudaFree(out);
        cudaFree(ptr); cudaFree(start); cudaFree(groups); cudaFree(dst); cudaFree(tok);
    }
    peer.close();
    if(h==5120 && gu==19 && down==16) {
        ck(cudaSetDevice(0));
        require(primary.fill_slot_blocking(primary.slot_of(0,0),src.blob(0,0),err),err);
        const auto path=std::filesystem::temp_directory_path()/
            ("basalt-exclusive-peer-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        {
            std::ofstream file(path,std::ios::binary);
            for(const auto& bytes:src.data)file.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());
            require(bool(file),"write exclusive peer fixture");
        }
        core::FileExpertSource resident;
        require(resident.open(core::FileRange(path.string()),nl,ne,err),err);
        resident.set_spares(8);
        require(peer.open(1,{{0,1},{1,0}},primary,resident,nl,ne,1024,2,err,true),err);
        require(resident.pin_cache_complement(primary,err,true,{{0,1},{1,0}}),err);
        peer.set_exclusive(&resident);
        const auto spares=resident.spares_free(),reads=resident.file_reads();
        require(spares>=8,"exclusive helper has no landing places");
        std::vector<float> usage(nl*ne,0.f);usage[2]=20.f;
        std::vector<int32_t> res0(nl*ne,-1);res0[0]=0;
        require(peer.adapt(usage.data(),res0.data(),1,err),err);
        peer.apply_pending(true);
        require(peer.swaps()>0 && peer.has(0,2) && !peer.has(0,1),"exclusive helper swap did not land");
        require(resident.has_resident(0,1) && !resident.has_resident(0,2),"exclusive RAM ownership did not move");
        require(std::memcmp(resident.resident_blob(0,1),src.blob(0,1),f.bytes)==0,"evicted expert bytes differ");
        ck(cudaSetDevice(1));std::vector<uint8_t> bytes(f.bytes);
        ck(cudaMemcpy(bytes.data(),peer.slot_ptr(0,2),f.bytes,cudaMemcpyDeviceToHost));
        require(bytes==src.data[2],"incoming expert bytes differ");
        require(resident.file_reads()==reads && resident.spares_free()==spares,"exclusive swap read disk or leaked landing place");
        ck(cudaSetDevice(0));
        usage[1]=20.f;usage[2]=100.f;uint64_t primary_swaps=0,promotions=0;
        require(basalt::deepseek41::adapt_primary(primary,resident,peer,usage.data(),nl,ne,8,res0.data(),nullptr,primary_swaps,promotions,err),err);
        require(primary_swaps==1 && primary.slot_of(0,1)==0 && primary.slot_of(0,0)<0,"primary heat admission failed");
        require(res0[1]==0 && res0[0]<0 && peer.has(0,2),"primary mapping or peer exclusion failed");
        require(resident.has_resident(0,0) && !resident.has_resident(0,1),"primary RAM ownership failed");
        require(std::memcmp(resident.resident_blob(0,0),src.blob(0,0),f.bytes)==0,"primary victim bytes differ");
        ck(cudaMemcpy(bytes.data(),primary.device_slot(0),f.bytes,cudaMemcpyDeviceToHost));
        require(bytes==src.data[1],"primary incoming bytes differ");
        require(resident.file_reads()==reads && resident.spares_free()==spares,"primary swap read disk or leaked landing place");
        require(basalt::deepseek41::adapt_primary(primary,resident,peer,usage.data(),nl,ne,8,res0.data(),nullptr,primary_swaps,promotions,err),err);
        require(primary_swaps==1,"stable heat caused another primary exchange");
        basalt::core::diagnostics().exp["ds-primary-promote"]=1;
        basalt::core::diagnostics().exp["ds-promotion-check"]=1;
        require(basalt::deepseek41::adapt_primary(primary,resident,peer,usage.data(),nl,ne,8,res0.data(),nullptr,primary_swaps,promotions,err),err);
        require(primary_swaps==2 && promotions==1 && primary.slot_of(0,2)==0 && primary.slot_of(0,1)<0,"hot peer expert was not promoted");
        require(peer.has(0,1) && !peer.has(0,2) && res0[2]==0 && res0[1]<0,"promotion residency or demotion failed");
        require(!resident.has_resident(0,1) && !resident.has_resident(0,2) && resident.has_resident(0,0),"promotion duplicated or lost a RAM place");
        ck(cudaMemcpy(bytes.data(),primary.device_slot(0),f.bytes,cudaMemcpyDeviceToHost));
        require(bytes==src.data[2],"promoted expert bytes differ");
        {const core::OnDevice on(1);ck(cudaMemcpy(bytes.data(),peer.slot_ptr(0,1),f.bytes,cudaMemcpyDeviceToHost));}
        require(bytes==src.data[1],"demoted expert bytes differ");
        require(resident.file_reads()==reads && resident.spares_free()==spares,"promotion read disk or leaked landing place");
        require(basalt::deepseek41::adapt_primary(primary,resident,peer,usage.data(),nl,ne,8,res0.data(),nullptr,primary_swaps,promotions,err),err);
        require(primary_swaps==2 && promotions==1,"stable peer heat caused another promotion");
        basalt::core::diagnostics().exp["ds-primary-promote"]=0;
        basalt::core::diagnostics().exp["ds-promotion-check"]=0;
        peer.close();resident.close();std::filesystem::remove(path);
        std::printf("Exclusive helper swap: resident ownership, exact bytes, no disk reads, landing places restored\n");
    }
    primary.close();
    std::printf("Peer fill, compute, scatter, adaptive swap and promotion: byte-identical\n");
}

static void primary_batches(std::mt19937& rng) {
    constexpr int nl=3,ne=16,each=8,h=512,ff=256;
    std::ostringstream text;text<<"# experts v6: (n_expert 16)\n";
    uint64_t offset=0;std::string err;std::vector<int64_t> sizes;
    Source bytes;bytes.experts=ne;bytes.data.resize(nl*ne);
    for(int l=0;l<nl;++l) {
        cpu::NativeFmt f;
        require(cpu::native_fmt(l==2?39:19,l==0?19:l==1?16:39,h,ff,f,err),err);
        text<<l<<' '<<f.gu_type<<' '<<f.d_type<<' '<<offset<<' '<<f.bytes<<'\n';offset+=ne*f.bytes;
        for(int e=0;e<ne;++e) {auto& blob=bytes.data[l*ne+e];blob.resize(f.bytes);for(auto& v:blob)v=uint8_t(rng());}
        for(int e=0;e<each;++e)sizes.push_back(int64_t(f.bytes));
    }
    require(cpu::expert_layout_load_text(text.str(),nl,ne,err,{h,ff}),err);
    const auto path=std::filesystem::temp_directory_path()/
        ("basalt-primary-swaps-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    {std::ofstream out(path,std::ios::binary);for(const auto& blob:bytes.data)out.write(reinterpret_cast<const char*>(blob.data()),blob.size());require(bool(out),"write primary batch fixture");}
    ck(cudaSetDevice(0));core::ExpertCache cache;
    require(cache.open_sized(sizes,nl,ne,err),err);cache.set_per_layer_admission(true);
    for(int l=0;l<nl;++l)for(int e=0;e<each;++e) {
        const int slot=cache.admit(l,e);require(slot>=0,"batch admission");require(cache.fill_slot_blocking(slot,bytes.blob(l,e),err),err);
    }
    core::FileExpertSource source;require(source.open(core::FileRange(path.string()),nl,ne,err),err);source.set_spares(8);
    require(source.pin_cache_complement(cache,err,true),err);
    const auto spares=source.spares_free(),reads=source.file_reads();
    core::PeerExperts peer;std::vector<float> heat(nl*ne,0);std::vector<int32_t> res(nl*ne,-1);
    for(int l=0;l<nl;++l)for(int e=0;e<ne;++e) {res[l*ne+e]=cache.slot_of(l,e);if(e>=each)heat[l*ne+e]=float(20+e);}
    uint64_t swaps=0,promotions=0;
    require(basalt::deepseek41::adapt_primary(cache,source,peer,heat.data(),nl,ne,23,res.data(),nullptr,swaps,promotions,err),err);
    require(swaps==23,"primary batch limit");
    require(basalt::deepseek41::adapt_primary(cache,source,peer,heat.data(),nl,ne,8,res.data(),nullptr,swaps,promotions,err),err);
    require(swaps==24,"primary batch remainder");
    for(int l=0;l<nl;++l)for(int e=0;e<each;++e) {
        const int in=ne-1-e,slot=cache.slot_of(l,in);require(slot>=0 && cache.slot_of(l,e)<0,"batched primary ownership");
        require(res[l*ne+in]==slot && res[l*ne+e]<0,"batched primary residency");
        require(source.has_resident(l,e) && !source.has_resident(l,in),"batched RAM ownership");
        std::vector<uint8_t> got(bytes.data[l*ne+in].size());ck(cudaMemcpy(got.data(),cache.device_slot(slot),got.size(),cudaMemcpyDeviceToHost));
        require(got==bytes.data[l*ne+in],"batched primary bytes");
        require(std::memcmp(source.resident_blob(l,e),bytes.blob(l,e),got.size())==0,"batched eviction bytes");
    }
    require(source.file_reads()==reads && source.spares_free()==spares,"batched swaps read disk or leaked places");
    std::vector<std::pair<int32_t,int32_t>> helpers;
    for(int l=0;l<nl;++l)for(int e=0;e<4;++e)helpers.emplace_back(l,e);
    require(peer.open(1,helpers,cache,source,nl,ne,1024,int(helpers.size()),err,true),err);
    peer.set_exclusive(&source);
    for(const auto& [l,e]:helpers)require(source.spare_release(l,e),"helper's existing RAM place was not released");
    const auto promotion_spares=source.spares_free(),promotion_reads=source.file_reads();
    for(const auto& [l,e]:helpers)heat[l*ne+e]=float(100-e);
    basalt::core::diagnostics().exp["ds-primary-promote"]=1;
    basalt::core::diagnostics().exp["ds-promotion-check"]=1;
    require(basalt::deepseek41::adapt_primary(cache,source,peer,heat.data(),nl,ne,12,res.data(),nullptr,swaps,promotions,err),err);
    require(promotions==12 && swaps==36,"multi-format promotion batch limit");
    for(int l=0;l<nl;++l)for(int e=0;e<4;++e) {
        const int out=each+e,slot=cache.slot_of(l,e);
        require(slot>=0 && cache.slot_of(l,out)<0 && peer.has(l,out) && !peer.has(l,e),"multi-format promotion ownership");
        require(!source.has_resident(l,e) && !source.has_resident(l,out),"multi-format promotion RAM exclusion");
        std::vector<uint8_t> got(bytes.data[l*ne+e].size());
        ck(cudaMemcpy(got.data(),cache.device_slot(slot),got.size(),cudaMemcpyDeviceToHost));
        require(got==bytes.data[l*ne+e],"multi-format promoted bytes");
        {const core::OnDevice on(1);ck(cudaMemcpy(got.data(),peer.slot_ptr(l,out),got.size(),cudaMemcpyDeviceToHost));}
        require(got==bytes.data[l*ne+out],"multi-format demoted bytes");
    }
    require(source.file_reads()==promotion_reads && source.spares_free()==promotion_spares,"multi-format promotions read disk or leaked places");
    basalt::core::diagnostics().exp["ds-primary-promote"]=0;
    basalt::core::diagnostics().exp["ds-promotion-check"]=0;
    peer.close();
    source.close();cache.close();std::filesystem::remove(path);
    std::printf("Primary batches: 24 RAM exchanges and 12 peer promotions, three blob sizes, exact bytes and restored spares\n");
}
int main(int argc, char** argv) {
    basalt::core::diagnostics().exp["ds-primary-promote"]=0;
    int devices = 0;
    ck(cudaGetDeviceCount(&devices));
    if (devices < 2) return 77;
    void* resident = nullptr;
    if (argc == 3 && std::string(argv[1]) == "--dspark") {
        basalt::GgufFile f(argv[2]);
        require(f.get("general.architecture") && f.get("general.architecture")->s == "dflash", "not a DSpark GGUF");
        const uint64_t bytes = f.file_size() - f.data_start();
        require(!f.tensors().empty() && bytes < (12ull << 30), "invalid DSpark payload");
        ck(cudaSetDevice(1));
        ck(cudaMalloc(&resident, size_t(bytes)));
        const auto& first = f.tensors().front();
        const uint8_t* data = f.tensor_data(first) - first.offset;
        const size_t chunk = 64ull << 20;
        std::vector<uint8_t> stage(chunk);
        for (uint64_t off = 0; off < bytes; off += chunk) {
            const size_t n = size_t(std::min<uint64_t>(chunk, bytes - off));
            std::memcpy(stage.data(), data + off, n);
            ck(cudaMemcpy((uint8_t*) resident + off, stage.data(), n, cudaMemcpyHostToDevice));
        }
        std::printf("DSpark raw weights resident on device 1: %.3f GiB (no draft execution)\n", bytes / double(1ull << 30));
    } else if (argc != 1) return 2;
    std::mt19937 rng(173);
    check(2560, 640, 19, 20, rng);
    check(5120, 2304, 19, 19, rng);
    check(5120, 2304, 19, 16, rng);
    check(5120, 2304, 39, 39, rng);
    primary_batches(rng);
    if (resident) { ck(cudaSetDevice(1)); ck(cudaFree(resident)); }
    std::printf("peer_geometry_test: passed\n");
}
