#include "model_state.hpp"
#include "basalt/core/vmm.hpp"
#include "basalt/platform/memory.hpp"
#include <algorithm>

namespace basalt::deepseek41 {
namespace {
constexpr int kv_step=16384;

int64_t vram_room(int reserve_mib) {
    size_t free=0,total=0;
    cuda_check(cudaMemGetInfo(&free,&total));
    return std::max<int64_t>(0,int64_t(free)-(int64_t(reserve_mib)<<20));
}
}

void Model::Impl::initialize_kv(SequenceState& s) {
    if(!options.elastic_kv)return;
    const core::OnDevice on(0);
    std::vector<uint64_t> rows;
    for(const auto& l:s.layers)if(l.kv_source) {
        rows.push_back(288*(2/l.ratio));
        rows.push_back(68*(2/l.ratio));
    }
    s.kv=std::make_unique<core::KvArena>();
    std::string error;
    expect(s.kv->open(rows,(options.context+1)/2,2,error),"elastic KV: "+error);
    s.kv->set_extent_cells(kv_step);
    const int base=std::min(options.context,kv_step);
    expect(s.kv->map(base,error),"elastic KV base: "+error);
    basalt::log::info("basalt deepseek41","elastic KV: startup base %.1f MiB per slot, %lld tokens backed",
                      s.kv->mapped_bytes()/1048576.,(long long)s.kv->cells());
    size_t i=0;
    for(auto& l:s.layers)if(l.kv_source) {
        l.compressed=s.kv->array(i++);
        l.index_keys=s.kv->array(i++);
    }
}

void Model::Impl::regrow_experts(uint64_t released_bytes) {
    if(!cache.segmented() || cache.slots()==cache.full_slots())return;
    const core::OnDevice on(0);
    // The driver can report an older free-memory estimate just after unmapping.
    const auto room=std::max(vram_room(options.reserve_mib),int64_t(released_bytes));
    const auto before=cache.slots();
    if(room<cache.segment_bytes())return;
    std::string error;
    (void)cache.grow(cache.mapped_bytes()+room,error);
    if(cache.slots()==before)return;
    for(int l=0;l<40;++l)for(int e=0;e<384;++e) {
        const int slot=cache.slot_of(l,e);
        if(slot<before || slot>=cache.slots())continue;
        expect(cache.fill_slot(slot,source.blob(l,e),primary->stream(),error,
                               int64_t(packed.expert_layout().blob_bytes(l))),error);
    }
    primary->synchronize();
    for(int l=0;l<40;++l)for(int e=0;e<384;++e) {
        const int slot=cache.slot_of(l,e);
        if(slot>=before && slot<cache.slots() && elastic_homes[size_t(slot)]) {
            expect(source.home_release(l,e),"elastic KV: missing expert RAM home");
            elastic_homes[size_t(slot)]=0;
        }
    }
    source.homes_reclaim();
    refresh_primary_lookup();
    primary->synchronize();
    basalt::log::info("basalt deepseek41","elastic KV: expert slots %lld -> %lld",(long long)before,(long long)cache.slots());
}

bool Model::Impl::fit_kv(int end_position,std::string& error,bool during_prefill) {
    if(!kv || end_position<=kv->cells())return true;
    if(during_prefill || borrowed_prefill_bytes) {
        error="prefill workspace loan is active";
        return false;
    }
    const core::OnDevice on(0);
    const int want=std::min(options.context,((end_position+kv_step-1)/kv_step)*kv_step);
    const int64_t before=cache.slots(),old_cells=kv->cells();
    const uint64_t old_bytes=kv->mapped_bytes();
    const auto room=vram_room(options.reserve_mib);
    const auto expert_bytes=cache.mapped_bytes();
    int64_t chosen=want,add=int64_t(kv->grow_bytes(chosen));
    const int64_t gran=cache.segment_bytes();
    const auto can_lend=[&](int64_t bytes) {
        if(bytes<=room)return true;
        if(gran<=0 || bytes-room>cache.mapped_bytes())return false;
        const int64_t keep=(cache.mapped_bytes()-(bytes-room))/gran*gran;
        const int64_t fixed=(cache.full_bytes()+gran-1)/gran*gran-cache.elastic_bytes();
        return keep>=fixed &&
               cache.slots_after_shrink(keep)>=std::min<int64_t>(128,cache.full_slots());
    };
    // If the growth cushion does not fit, try exactly the requested position first.
    if(!can_lend(add)) {chosen=end_position;add=int64_t(kv->grow_bytes(chosen));}
    if(!can_lend(add)) {
        error="not enough VRAM for position "+std::to_string(end_position)+" with the expert-cache floor";
        return false;
    }
    if(add>room) {
        if(helper.valid())helper.apply_pending(true);
        primary->synchronize();
        const auto keep_for=[&](int64_t bytes) {return (expert_bytes-(bytes-room))/gran*gran;};
        const auto save_experts=[&](int64_t keep) {
            if(!source.complement_ready())return true;
            const auto next=cache.slots_after_shrink(keep);
            std::vector<std::pair<int32_t,int32_t>> missing;
            std::vector<int32_t> slots;
            for(int l=0;l<40;++l)for(int e=0;e<384;++e) {
                const auto slot=cache.slot_of(l,e);
                if(slot>=next && slot<before && !source.has_resident(l,e)) {
                    missing.emplace_back(l,e);slots.push_back(slot);
                }
            }
            if(!missing.empty()) {
                std::vector<uint8_t*> homes;
                if(!source.homes_take(missing,error,homes,true,kExpertRamFloor))return false;
                const auto stream=static_cast<cudaStream_t>(primary->stream());
                try {
                    for(size_t i=0;i<slots.size();++i)
                        cuda_check(cudaMemcpyAsync(homes[i],cache.device_slot(slots[i]),
                            packed.expert_layout().blob_bytes(missing[i].first),cudaMemcpyDeviceToHost,stream));
                    primary->synchronize();
                } catch(...) {
                    cudaStreamSynchronize(stream);
                    source.homes_abort();
                    throw;
                }
                source.homes_commit();
                for(int32_t slot:slots)elastic_homes[size_t(slot)]=1;
            }
            return true;
        };
        auto keep=keep_for(add);
        if(!save_experts(keep)) {
            // The rounded growth may fit in VRAM but evict more experts than RAM can hold.
            const auto exact_add=int64_t(kv->grow_bytes(end_position));
            if(chosen==end_position || exact_add>=add || !can_lend(exact_add))return false;
            if(exact_add>room) {
                const auto exact_keep=keep_for(exact_add);
                if(cache.slots_after_shrink(exact_keep)==cache.slots_after_shrink(keep) ||
                   !save_experts(exact_keep))return false;
                keep=exact_keep;
            }
            basalt::log::info("basalt deepseek41","elastic KV: slot %d retry without growth cushion (%lld -> %d tokens)",
                              active_slot,(long long)chosen,end_position);
            chosen=end_position;
            add=exact_add;
            error.clear();
        }
        if(add>room) {
            const bool shrunk=cache.shrink(keep,error);
            refresh_primary_lookup();
            primary->synchronize();
            if(!shrunk) {
                regrow_experts(uint64_t(expert_bytes-cache.mapped_bytes()));
                if(memory_changed)memory_changed();
                return false;
            }
        }
    }
    if(!kv->map(chosen,error)) {
        std::string rollback;
        expect(kv->unmap(old_cells,rollback),"elastic KV rollback: "+rollback);
        const int64_t net=expert_bytes-cache.mapped_bytes()-int64_t(kv->mapped_bytes()-old_bytes);
        regrow_experts(uint64_t(std::max<int64_t>(net,0)));
        if(memory_changed)memory_changed();
        return false;
    }
    if(memory_changed)memory_changed();
    basalt::log::info("basalt deepseek41","elastic KV: slot %d backed to %lld tokens (+%.1f MiB), expert slots %lld -> %lld",
                      active_slot,(long long)kv->cells(),double(kv->mapped_bytes()-old_bytes)/1048576.,
                      (long long)before,(long long)cache.slots());
    return true;
}

uint64_t Model::Impl::trim_kv(int position,bool replenish) {
    if(!kv || !kv->mapped_bytes())return 0;
    position=std::max(position,std::min(options.context,kv_step));
    const core::OnDevice on(0);
    const auto bytes=kv->mapped_bytes();
    if(kv->bytes_for(position)>=bytes)return 0;
    std::string error;
    expect(kv->unmap(position,error),"elastic KV: "+error);
    const auto released=bytes-kv->mapped_bytes();
    if(!released)return 0;
    if(replenish)regrow_experts(released);
    if(memory_changed)memory_changed();
    basalt::log::info("basalt deepseek41","elastic KV: slot %d returned %.1f MiB, %lld tokens backed",
                      active_slot,released/1048576.,(long long)kv->cells());
    return released;
}
}
