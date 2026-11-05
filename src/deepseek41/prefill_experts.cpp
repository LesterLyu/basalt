#include "prefill_experts.hpp"
#include "profile.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/deepseek41/execution_ops.hpp"
#include "basalt/log.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <deque>

namespace basalt::deepseek41 {
namespace mmq=basalt::prefill::mmq;
namespace {
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
constexpr int group_size=16,max_ring_size=16;
constexpr int H=5120,FF=2304,top_max=6;
constexpr int routed_experts=384;
}
struct PrefillExperts::Impl {
    DeviceModel& device;
    const kernels::cpu::ExpertLayout& layout;
    std::unique_ptr<mmq::Context> context;
    cudaStream_t copy=nullptr,peer_copy=nullptr;
    cudaEvent_t pack_done[2]{},batch_ready=nullptr;
    bool pack_used[2]{};
    uint8_t *second_gu=nullptr,*second_down=nullptr;
    const int ring_capacity=std::clamp(int(core::exp_flag("ds-prefill-ring-depth",4)),4,max_ring_size);
    int ring_size=ring_capacity;
    cudaEvent_t copied[max_ring_size]{},used[max_ring_size]{};
    cudaEvent_t peer_copied[max_ring_size]{};
    bool copied_ready[max_ring_size]{},used_ready[max_ring_size]{};
    uint8_t* host[max_ring_size]{};
    uint8_t *ring,*gu_weights,*down_weights,*xq,*hq;
    float *gu,*hidden;
    int32_t *token,*destination,*identity,*bounds,*relative;
    size_t blob_stride,gu_stride,down_stride;
    uint64_t cursor=0,peer_cursor=0;
    int row_capacity=0;
    std::vector<int> route_entries;
    Impl(DeviceModel& d,const kernels::cpu::ExpertLayout& l,int capacity):device(d),layout(l) {
        const core::OnDevice on(d.device());
        context=std::make_unique<mmq::Context>();
        blob_stride=gu_stride=down_stride=0;
        for(const auto& f:l.fmt) {
            expect(mmq::fits(f.gu_type,2*FF) && mmq::fits(f.d_type,H),"unsupported prompt MMQ expert type");
            blob_stride=std::max(blob_stride,size_t(f.bytes));
            gu_stride=std::max(gu_stride,mmq::matrix_bytes(f.gu_type,2*FF,H));
            down_stride=std::max(down_stride,mmq::matrix_bytes(f.d_type,H,FF));
        }
        blob_stride=(blob_stride+255)/256*256;
        constexpr auto mmq_buffers=MemoryKind::mmq;
        ring=d.allocate(ring_capacity*blob_stride,mmq_buffers);
        gu_weights=d.allocate(group_size*gu_stride+512,mmq_buffers);
        down_weights=d.allocate(group_size*down_stride+512,mmq_buffers);
        const size_t rows=size_t(capacity)*top_max;
        row_capacity=int(rows);
        xq=d.allocate(mmq::q8_bytes(rows,H),mmq_buffers);
        hq=d.allocate(mmq::q8_bytes(rows,FF),mmq_buffers);
        gu=d.alloc<float>(rows*2*FF,mmq_buffers);hidden=d.alloc<float>(rows*FF,mmq_buffers);
        token=d.alloc<int32_t>(rows,mmq_buffers);destination=d.alloc<int32_t>(rows,mmq_buffers);
        identity=d.alloc<int32_t>(rows,mmq_buffers);
        bounds=d.alloc<int32_t>(385,mmq_buffers);relative=d.alloc<int32_t>(24*17,mmq_buffers);
        mmq::iota(identity,rows,d.stream());
        try {
            ck(cudaStreamCreateWithFlags(&copy,cudaStreamNonBlocking));
            if(core::exp_flag("ds-prefill-host-dma")!=0) {
                second_gu=d.allocate(group_size*gu_stride+512,mmq_buffers);
                second_down=d.allocate(group_size*down_stride+512,mmq_buffers);
                for(auto& ev:pack_done)ck(cudaEventCreateWithFlags(&ev,cudaEventDisableTiming));
                if(core::exp_flag("ds-prefill-peer-stream")!=0)
                    ck(cudaStreamCreateWithFlags(&peer_copy,cudaStreamNonBlocking));
#if CUDART_VERSION >= 13030
                if(core::exp_flag("ds-prefill-batch-dma")!=0)ck(cudaEventCreateWithFlags(&batch_ready,cudaEventDisableTiming));
#endif
            }
            for(int i=0;i<ring_capacity;++i) {
                ck(cudaHostAlloc(reinterpret_cast<void**>(&host[i]),blob_stride,cudaHostAllocPortable));
                ck(cudaEventCreateWithFlags(&copied[i],cudaEventDisableTiming));
                ck(cudaEventCreateWithFlags(&used[i],cudaEventDisableTiming));
                if(peer_copy)ck(cudaEventCreateWithFlags(&peer_copied[i],cudaEventDisableTiming));
            }
        }catch(...){close();throw;}
        d.synchronize();
    }
    void close() {
        const core::OnDevice on(device.device());
        if(copy)cudaStreamSynchronize(copy);
        if(peer_copy)cudaStreamSynchronize(peer_copy);
        cudaStreamSynchronize(static_cast<cudaStream_t>(device.stream()));
        for(int i=0;i<ring_capacity;++i) {
            if(host[i])cudaFreeHost(host[i]);host[i]=nullptr;
            if(copied[i])cudaEventDestroy(copied[i]);copied[i]=nullptr;
            if(used[i])cudaEventDestroy(used[i]);used[i]=nullptr;
            if(peer_copied[i])cudaEventDestroy(peer_copied[i]);peer_copied[i]=nullptr;
        }
        for(auto& ev:pack_done) {if(ev)cudaEventDestroy(ev);ev=nullptr;}
        if(batch_ready)cudaEventDestroy(batch_ready);batch_ready=nullptr;
        if(copy)cudaStreamDestroy(copy);copy=nullptr;
        if(peer_copy)cudaStreamDestroy(peer_copy);peer_copy=nullptr;
    }
    ~Impl(){close();}
    void stage_packed(int layer,int expert,core::FileExpertSource& source,size_t bytes,size_t gu_bytes,
                      size_t down_off,size_t down_bytes,uint8_t* gu,uint8_t* down,ExecutionStats& stats,cudaStream_t compute) {
        const int slot=int(cursor++%ring_size);
        const uint8_t* src=source.blob(layer,expert);expect(src,"missing packed prompt expert");
        if(!source.pinned(layer,expert)) {
            if(copied_ready[slot])ck(cudaEventSynchronize(copied[slot]));
            std::memcpy(host[slot],src,bytes);src=host[slot];
        }
        ck(cudaMemcpyAsync(gu,src,gu_bytes,cudaMemcpyHostToDevice,copy));
        ck(cudaMemcpyAsync(down,src+down_off,down_bytes,cudaMemcpyHostToDevice,copy));
        ck(cudaEventRecord(copied[slot],copy));copied_ready[slot]=true;
        ck(cudaStreamWaitEvent(compute,copied[slot],0));
        stats.streamed_bytes+=bytes;
    }
    const uint8_t* stage(int layer,int expert,core::FileExpertSource& source,size_t bytes,ExecutionStats& stats,cudaStream_t compute) {
        const int slot=int(cursor++%ring_size);
        if(used_ready[slot])ck(cudaStreamWaitEvent(copy,used[slot],0));
        const uint8_t* src=source.blob(layer,expert);expect(src,"missing prompt expert");
        if(!source.pinned(layer,expert)) {
            if(copied_ready[slot])ck(cudaEventSynchronize(copied[slot]));
            std::memcpy(host[slot],src,bytes);src=host[slot];
        }
        auto* dst=ring+size_t(slot)*blob_stride;
        ck(cudaMemcpyAsync(dst,src,bytes,cudaMemcpyHostToDevice,copy));
        ck(cudaEventRecord(copied[slot],copy));copied_ready[slot]=true;
        ck(cudaStreamWaitEvent(compute,copied[slot],0));
        stats.streamed_bytes+=bytes;return dst;
    }
};
PrefillExperts::PrefillExperts(DeviceModel& d,const kernels::cpu::ExpertLayout& l,int capacity)
    :impl_(std::make_unique<Impl>(d,l,capacity)){}
PrefillExperts::~PrefillExperts(){const core::OnDevice on(impl_->device.device());impl_.reset();}
size_t PrefillExperts::pool_bytes()const{return impl_->context->reserved_bytes();}
void PrefillExperts::run(int layer,const float* x,std::span<const int32_t> ids,const std::vector<int32_t>& kind,
                         int tokens,int top,core::ExpertCache& cache,core::FileExpertSource& source,float* output,ExecutionStats& stats) {
    auto& m=*impl_;auto& d=m.device;const core::OnDevice on(d.device());
    const auto stream=static_cast<cudaStream_t>(d.stream());
    std::map<int,std::vector<int>> groups;
    for(int j=0;j<tokens*top;++j)if(kind[j]!=2)groups[ids[j]].push_back(j);
    d.zero(output,size_t(tokens)*top*H*sizeof(float));if(groups.empty())return;
    std::vector<int32_t> order,src,dst,off{0},rel;
    for(const auto& [expert,members]:groups) {
        order.push_back(expert);
        for(int entry:members) {src.push_back(entry/top);dst.push_back(entry);}
        off.push_back(int(src.size()));
    }
    const int n=int(order.size()),rows=int(src.size());rel.resize(((n+15)/16)*17);
    for(int first=0;first<n;first+=group_size)for(int j=0;j<=group_size;++j)
        rel[(first/group_size)*17+j]=off[std::min(n,first+j)]-off[first];
    d.upload(m.token,src.data(),src.size()*4);d.upload(m.destination,dst.data(),dst.size()*4);
    d.upload(m.bounds,off.data(),off.size()*4);d.upload(m.relative,rel.data(),rel.size()*4);
    const auto& f=m.layout.fmt[size_t(layer)];
    const size_t gu_bytes=mmq::matrix_bytes(f.gu_type,2*FF,H),dn_bytes=mmq::matrix_bytes(f.d_type,H,FF);
    mmq::quantize(x,m.token,m.xq,f.gu_type,H,H,rows,d.stream());
    for(int first=0;first<n;first+=group_size) {
        const int ng=std::min(group_size,n-first),r0=off[first],nr=off[first+ng]-r0;
        int max_rows=0;
        for(int j=0;j<ng;++j) {
            const int e=order[first+j],candidate=cache.slot_of(layer,e);
            const int slot=candidate<cache.slots()?candidate:core::kNotResident;
            const uint8_t* blob=slot>=0?cache.device_slot(slot):m.stage(layer,e,source,f.bytes,stats,stream);
            mmq::gather_native(blob,blob+f.up_off,gu_bytes/2,blob+f.down_off,dn_bytes,
                               m.gu_weights+size_t(j)*gu_bytes,m.down_weights+size_t(j)*dn_bytes,d.stream());
            if(slot<0) {const int ring_slot=int((m.cursor-1)%m.ring_size);ck(cudaEventRecord(m.used[ring_slot],stream));m.used_ready[ring_slot]=true;}
            max_rows=std::max(max_rows,off[first+j+1]-off[first+j]);
        }
        d.zero(m.gu_weights+size_t(ng)*gu_bytes,512);d.zero(m.down_weights+size_t(ng)*dn_bytes,512);
        mmq::Product gu;
        gu.w=m.gu_weights;gu.type=f.gu_type;gu.w_rows=2*FF;gu.w_cols=H;gu.expert_bytes=gu_bytes;gu.n=ng;
        gu.xq=m.xq;gu.bounds=m.bounds+first;gu.ids=m.identity;gu.total_rows=rows;gu.max_rows=max_rows;
        gu.dst=m.gu;gu.ld_dst=2*FF;m.context->run(gu,d.stream());
        mmq::swiglu(m.gu+size_t(r0)*2*FF,m.hidden,nr,FF,false,d.stream(),nullptr,nullptr,0,10);
        mmq::quantize(m.hidden,nullptr,m.hq,f.d_type,FF,FF,nr,d.stream());
        mmq::Product dn;
        dn.w=m.down_weights;dn.type=f.d_type;dn.w_rows=H;dn.w_cols=FF;dn.expert_bytes=dn_bytes;dn.n=ng;
        dn.xq=m.hq;dn.bounds=m.relative+(first/group_size)*17;dn.ids=m.destination+r0;
        dn.total_rows=nr;dn.max_rows=max_rows;dn.dst=output;dn.ld_dst=H;m.context->run(dn,d.stream());
    }
    // The layer's host routing vectors remain alive until their asynchronous copies are consumed.
    d.synchronize();
}
void PrefillExperts::run_long(int layer,const float* x,const std::vector<int32_t>& ids,const float* weights,int tokens,int top,
                              core::ExpertCache& cache,core::FileExpertSource& source,core::PeerExperts& helper,
                              float* parts,double* accumulator,ExecutionStats& stats,int64_t primary_limit,Profile* profile,
                              const std::function<void()>& between_groups,void* background_stream) {
    auto& m=*impl_;auto& d=m.device;const core::OnDevice on(d.device());
    const auto resident_slot=[&](int expert) {
        const auto slot=cache.slot_of(layer,expert);
        return slot<primary_limit?slot:core::kNotResident;
    };
    m.ring_size=core::exp_flag("ds-prefill-ring-wide",1)!=0?m.ring_capacity:4;
    const auto stream=static_cast<cudaStream_t>(background_stream?background_stream:d.stream());
    const auto upload=[&](void* dst,const void* src,size_t bytes) {
        ck(cudaMemcpyAsync(dst,src,bytes,cudaMemcpyHostToDevice,stream));
    };
    struct Plan {std::vector<int32_t> src,dst,bounds{0};};std::deque<Plan> plans;
    // Retain every asynchronous upload's source until all its consumers finish, including on an exception.
    struct Complete {
        cudaStream_t stream,copy,peer;
        ~Complete(){cudaStreamSynchronize(stream);cudaStreamSynchronize(copy);if(peer)cudaStreamSynchronize(peer);}
    } complete{stream,m.copy,m.peer_copy};
    std::map<int,std::vector<int>> groups;
    std::array<int,routed_experts+1> offsets{};
    std::vector<int> order;order.reserve(routed_experts);
    const int entries=tokens*top;
    const bool buckets=core::exp_flag("ds-prefill-route-buckets",1)!=0;
    if(buckets) {
        // IDs were checked by the caller. Preserve token order within each expert.
        for(int j=0;j<entries;++j)++offsets[ids[j]+1];
        for(int e=0;e<routed_experts;++e) {
            if(offsets[e+1])order.push_back(e);
            offsets[e+1]+=offsets[e];
        }
        auto cursor=offsets;
        m.route_entries.resize(entries);
        for(int j=0;j<entries;++j)m.route_entries[cursor[ids[j]]++]=j;
    } else {
        for(int j=0;j<entries;++j)groups[ids[j]].push_back(j);
        for(const auto& [e,members]:groups)order.push_back(e);
    }
    const auto members_of=[&](int expert)->std::span<const int> {
        if(!buckets)return groups.at(expert);
        return std::span<const int>(m.route_entries).subspan(offsets[expert],offsets[expert+1]-offsets[expert]);
    };
    // Similar row counts keep each bounded MMQ pass populated instead of padding cold experts beside hot ones.
    std::sort(order.begin(),order.end(),[&](int a,int b){
        const auto x=members_of(a).size(),y=members_of(b).size();return x!=y?x>y:a<b;
    });
    const auto& f=m.layout.fmt[size_t(layer)];
    const size_t gu_bytes=mmq::matrix_bytes(f.gu_type,2*FF,H),dn_bytes=mmq::matrix_bytes(f.d_type,H,FF);
    const bool host_dma=m.second_gu && core::exp_flag("ds-prefill-host-dma")!=0 && f.up_off==gu_bytes/2 &&
                        f.down_off==gu_bytes && size_t(f.bytes)>=gu_bytes+dn_bytes;
    // Only use the separate ring cursor when host experts DMA directly into pack
    // banks. In that mode the staging ring itself is exclusive to peer weights.
    const bool peer_stream=host_dma && m.peer_copy && core::exp_flag("ds-prefill-peer-stream")!=0;
    if(host_dma && layer==0)basalt::log::info("basalt deepseek41","direct host DMA to MMQ pack banks");
#if CUDART_VERSION >= 13030
    const bool batch_dma=host_dma && m.batch_ready && core::exp_flag("ds-prefill-batch-dma")!=0;
    bool batch_logged=false;
#endif
    if(core::exp_flag("ds-prefill-peer-audit")!=0 && helper.valid()) {
        uint64_t peers=0,entries=0,cheaper=0,weight_bytes=0,io_bytes=0;
        std::vector<bool> inputs(size_t(tokens),false);
        for(int expert:order) {
            if(resident_slot(expert)>=0 || !helper.has(layer,expert))continue;
            const auto members=members_of(expert);
            ++peers;entries+=members.size();weight_bytes+=f.bytes;
            cheaper+=2*members.size()*H*sizeof(float)<size_t(f.bytes);
            for(int member:members)inputs[size_t(member/top)]=true;
        }
        const uint64_t unique=std::count(inputs.begin(),inputs.end(),true);
        io_bytes=(entries+unique)*H*sizeof(float);
        basalt::log::info("basalt deepseek41","prefill peer audit layer=%d experts=%llu rows=%llu unique_input_rows=%llu weight_bytes=%llu min_io_bytes=%llu smaller_expert_roundtrips=%llu",
                         layer,(unsigned long long)peers,(unsigned long long)entries,(unsigned long long)unique,
                         (unsigned long long)weight_bytes,(unsigned long long)io_bytes,(unsigned long long)cheaper);
    }
    for(int first=0;first<int(order.size());first+=group_size) {
        const int ng=std::min(group_size,int(order.size())-first);
        const size_t pack_stamp=profile?profile->begin_async("expert_pack_and_wait",stream):size_t(-1);
        const int bank=(first/group_size)&1;
        auto* gu_buffer=host_dma && bank?m.second_gu:m.gu_weights;
        auto* down_buffer=host_dma && bank?m.second_down:m.down_weights;
        if(host_dma && m.pack_used[bank])ck(cudaStreamWaitEvent(m.copy,m.pack_done[bank],0));
        uint32_t copied_in_batch=0;
#if CUDART_VERSION >= 13030
        if(batch_dma) {
            std::array<void*,group_size*2> dsts{};
            std::array<const void*,group_size*2> srcs{};
            std::array<size_t,group_size*2> lengths{};
            size_t copies=0;
            for(int j=0;j<ng;++j) {
                const int e=order[first+j];
                if(resident_slot(e)>=0 || (helper.valid() && helper.has(layer,e)) || !source.pinned(layer,e))continue;
                const auto* blob=source.blob(layer,e);expect(blob,"missing pinned prompt expert");
                dsts[copies]=gu_buffer+size_t(j)*gu_bytes;srcs[copies]=blob;lengths[copies++]=gu_bytes;
                dsts[copies]=down_buffer+size_t(j)*dn_bytes;srcs[copies]=blob+f.down_off;lengths[copies++]=dn_bytes;
                copied_in_batch|=uint32_t{1}<<j;
            }
            if(copies) {
                // All sources are fixed pinned expert blobs, and destinations do not overlap.
                cudaMemcpyAttributes attrs{};attrs.srcAccessOrder=cudaMemcpySrcAccessOrderStream;
                size_t attrs_index=0;
                const auto status=cudaMemcpyBatchAsync(dsts.data(),srcs.data(),lengths.data(),copies,&attrs,&attrs_index,1,m.copy);
                if(status==cudaErrorNotSupported)copied_in_batch=0;
                else {
                    ck(status);ck(cudaEventRecord(m.batch_ready,m.copy));ck(cudaStreamWaitEvent(stream,m.batch_ready,0));
                    stats.streamed_bytes+=uint64_t(copies/2)*f.bytes;
                    if(!batch_logged && layer==0) {
                        basalt::log::info("basalt deepseek41","batched pinned host DMA to MMQ pack banks");
                        batch_logged=true;
                    }
                }
            }
        }
#endif
        int largest=0;
        for(int j=0;j<ng;++j) {
            const int e=order[first+j],slot=resident_slot(e);const uint8_t* blob=nullptr;
            const bool peer=slot<0 && helper.valid() && helper.has(layer,e);
            int ring_slot=-1;
            if(host_dma && slot<0 && !peer) {
                if(!(copied_in_batch & (uint32_t{1}<<j)))
                    m.stage_packed(layer,e,source,f.bytes,gu_bytes,f.down_off,dn_bytes,
                                   gu_buffer+size_t(j)*gu_bytes,down_buffer+size_t(j)*dn_bytes,stats,stream);
                largest=std::max(largest,int(members_of(e).size()));
                continue;
            }
            if(slot>=0)blob=cache.device_slot(slot);
            else if(peer) {
                ring_slot=int((peer_stream?m.peer_cursor++:m.cursor++)%m.ring_size);
                const auto copy_stream=peer_stream?m.peer_copy:m.copy;
                const auto ready=peer_stream?m.peer_copied[ring_slot]:m.copied[ring_slot];
                if(m.used_ready[ring_slot])ck(cudaStreamWaitEvent(copy_stream,m.used[ring_slot],0));
                auto* dst=m.ring+size_t(ring_slot)*m.blob_stride;std::string error;
                expect(helper.copy_to(layer,e,dst,copy_stream,error),error);
                ck(cudaEventRecord(ready,copy_stream));
                if(!peer_stream)m.copied_ready[ring_slot]=true;
                ck(cudaStreamWaitEvent(stream,ready,0));blob=dst;stats.peer_streamed_bytes+=f.bytes;
            } else {
                blob=m.stage(layer,e,source,f.bytes,stats,stream);
                ring_slot=int((m.cursor-1)%m.ring_size);
            }
            mmq::gather_native(blob,blob+f.up_off,gu_bytes/2,blob+f.down_off,dn_bytes,
                               gu_buffer+size_t(j)*gu_bytes,down_buffer+size_t(j)*dn_bytes,stream);
            if(ring_slot>=0) {ck(cudaEventRecord(m.used[ring_slot],stream));m.used_ready[ring_slot]=true;}
            largest=std::max(largest,int(members_of(e).size()));
        }
        ck(cudaMemsetAsync(gu_buffer+size_t(ng)*gu_bytes,0,512,stream));
        ck(cudaMemsetAsync(down_buffer+size_t(ng)*dn_bytes,0,512,stream));
        if(profile)profile->end_async(pack_stamp,stream);
        const int per=m.row_capacity/ng;
        for(int start=0;start<largest;start+=per) {
            auto& plan=plans.emplace_back();auto& src=plan.src;auto& dst=plan.dst;auto& bounds=plan.bounds;int max_rows=0;
            for(int j=0;j<ng;++j) {
                const auto members=members_of(order[first+j]);
                const int end=std::min(int(members.size()),start+per),begin=std::min(int(members.size()),start);
                for(int r=begin;r<end;++r) {src.push_back(members[r]/top);dst.push_back(members[r]);}
                bounds.push_back(int(src.size()));max_rows=std::max(max_rows,end-begin);
            }
            const int nr=int(src.size());
            upload(m.token,src.data(),nr*4);upload(m.destination,dst.data(),nr*4);upload(m.bounds,bounds.data(),bounds.size()*4);
            {AsyncGpuScope stamp(profile,"expert_quant_input",stream);mmq::quantize(x,m.token,m.xq,f.gu_type,H,H,nr,stream);}
            mmq::Product gu;gu.w=gu_buffer;gu.type=f.gu_type;gu.w_rows=2*FF;gu.w_cols=H;gu.expert_bytes=gu_bytes;gu.n=ng;
            gu.xq=m.xq;gu.bounds=m.bounds;gu.ids=m.identity;gu.total_rows=nr;gu.max_rows=max_rows;gu.dst=m.gu;gu.ld_dst=2*FF;
            {AsyncGpuScope stamp(profile,"expert_mmq_gate_up",stream);m.context->run(gu,stream);}
            {AsyncGpuScope stamp(profile,"expert_activate_quant",stream);
                if(core::exp_flag("ds-prefill-fused-swiglu-quant",1)!=0)
                    mmq::quantize_swiglu(m.gu,m.hq,f.d_type,FF,nr,10,stream);
                else {
                    mmq::swiglu(m.gu,m.hidden,nr,FF,false,stream,nullptr,nullptr,0,10);
                    mmq::quantize(m.hidden,nullptr,m.hq,f.d_type,FF,FF,nr,stream);
                }
            }
            mmq::Product dn;dn.w=down_buffer;dn.type=f.d_type;dn.w_rows=H;dn.w_cols=FF;dn.expert_bytes=dn_bytes;dn.n=ng;
            dn.xq=m.hq;dn.bounds=m.bounds;dn.ids=m.identity;dn.total_rows=nr;dn.max_rows=max_rows;dn.dst=parts;dn.ld_dst=H;
            {AsyncGpuScope stamp(profile,"expert_mmq_down",stream);m.context->run(dn,stream);}
            {AsyncGpuScope stamp(profile,"expert_accumulate",stream);
                accumulate_experts_f64(parts,m.destination,weights,accumulator,nr,H,top,stream);}
        }
        // The copy stream may overwrite this bank only after the previous MMQ readers finish.
        if(host_dma) {ck(cudaEventRecord(m.pack_done[bank],stream));m.pack_used[bank]=true;}
        if(between_groups)between_groups();
    }
    ck(cudaStreamSynchronize(stream));
}
}
