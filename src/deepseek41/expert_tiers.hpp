#pragma once
#include "basalt/core/expert_source.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cstring>

namespace basalt::deepseek41 {
// Between forwards, admit hot RAM or peer experts into cold same-layer primary slots.
inline bool adapt_primary(core::ExpertCache& cache,core::FileExpertSource& source,core::PeerExperts& peer,
                          const float* usage,int layers,int experts,int limit,int32_t* residency,
                          cudaStream_t stream,uint64_t& completed,uint64_t& promoted,std::string& error) {
    if(!source.complement_ready() || limit<=0)return true;
    const bool promote=peer.exclusive() && core::exp_flag("ds-primary-promote",1)!=0;
    struct Swap { float gain;int layer,in,out,slot;uint64_t offset=0;bool spare=false,peer=false; };
    std::vector<Swap> swaps;
    std::vector<std::pair<float,int>> candidates,victims;
    for(int l=0;l<layers;++l) {
        candidates.clear();victims.clear();
        for(int e=0;e<experts;++e) {
            const float heat=usage[l*experts+e];
            const auto slot=cache.slot_of(l,e);
            // An unmapped slot retains its owner for refill and is not part of the exchange pool.
            if(slot>=cache.slots())continue;
            if(slot>=0)victims.emplace_back(heat,e);
            else if(heat>=2 && ((promote && peer.has(l,e)) ||
                    (!(peer.valid() && peer.has(l,e)) && source.has_resident(l,e))))candidates.emplace_back(heat,e);
        }
        std::sort(candidates.begin(),candidates.end(),[](auto a,auto b){return a.first!=b.first?a.first>b.first:a.second<b.second;});
        std::sort(victims.begin(),victims.end());
        for(size_t i=0;i<std::min(candidates.size(),victims.size());++i) {
            const float gain=candidates[i].first-victims[i].first;
            if(gain<1.5f)break;
            swaps.push_back({gain,l,candidates[i].second,victims[i].second,cache.slot_of(l,victims[i].second)});
            swaps.back().peer=peer.valid() && peer.has(l,candidates[i].second);
        }
    }
    std::sort(swaps.begin(),swaps.end(),[](const Swap& a,const Swap& b){
        if(a.gain!=b.gain)return a.gain>b.gain;
        return a.layer!=b.layer?a.layer<b.layer:a.in<b.in;
    });
    if(swaps.size()>size_t(limit))swaps.resize(size_t(limit));
    const auto checked=[&](cudaError_t status){if(status==cudaSuccess)return true;error=cudaGetErrorString(status);return false;};
    // Bound outstanding evictions by the existing spare places, recycling them after each batch.
    for(size_t first=0;first<swaps.size();first+=8) {
        const size_t end=std::min(first+8,swaps.size());
        std::vector<Swap> pending;
        for(size_t i=first;i<end;++i) {
            auto s=swaps[i];
            const size_t bytes=kernels::cpu::expert_layout().blob_bytes(s.layer);
            if(!source.has_resident(s.layer,s.out)) {
                auto* dst=source.spare_take(s.layer,s.offset);
                if(!dst)continue;
                s.spare=true;
                if(!checked(cudaMemcpyAsync(dst,cache.device_slot(s.slot),bytes,cudaMemcpyDeviceToHost,stream))) {
                    source.spare_return(s.layer,s.offset);return false;
                }
            }
            pending.push_back(s);
            if(s.peer) {
                if(!peer.copy_to(s.layer,s.in,cache.device_slot(s.slot),stream,error))return false;
            } else if(!checked(cudaMemcpyAsync(cache.device_slot(s.slot),source.resident_blob(s.layer,s.in),bytes,cudaMemcpyHostToDevice,stream)))return false;
        }
        if(!checked(cudaStreamSynchronize(stream)))return false;
        for(const auto& s:pending) {
            if((s.spare && !source.spare_land(s.layer,s.out,s.offset)) ||
               ((!s.peer || source.has_resident(s.layer,s.in)) && !source.spare_release(s.layer,s.in))) {
                error="primary expert exchange lost a RAM place";return false;
            }
            cache.replace(s.layer,s.out,s.in);
            residency[s.layer*experts+s.out]=-1;residency[s.layer*experts+s.in]=s.slot;
            if(s.peer) {
                // The promoted peer slot takes the primary victim through its existing refill path.
                if(core::exp_flag("ds-promotion-check")!=0 && promoted<8) {
                    const size_t bytes=kernels::cpu::expert_layout().blob_bytes(s.layer);
                    std::vector<uint8_t> old(bytes),current(bytes);
                    {const core::OnDevice on(peer.device());
                    if(!checked(cudaMemcpy(old.data(),peer.slot_ptr(s.layer,s.in),bytes,cudaMemcpyDeviceToHost)))return false;}
                    if(!checked(cudaMemcpy(current.data(),cache.device_slot(s.slot),bytes,cudaMemcpyDeviceToHost)))return false;
                    if(std::memcmp(old.data(),current.data(),bytes)!=0) {error="peer promotion changed expert bytes";return false;}
                }
                if(!peer.reassign(s.layer,s.in,s.layer,s.out,source.resident_blob(s.layer,s.out),error))return false;
                ++promoted;
            }
            ++completed;
        }
        // Recycle the victims' RAM places before another bounded batch borrows them.
        if(promote)peer.apply_pending(true);
        if(promote && core::exp_flag("ds-promotion-check")!=0)for(const auto& s:pending)if(s.peer) {
            if(peer.has(s.layer,s.in) || !peer.has(s.layer,s.out) || source.has_resident(s.layer,s.out)) {
                error="peer promotion lost exclusive tier ownership";return false;
            }
        }
    }
    return true;
}
}
