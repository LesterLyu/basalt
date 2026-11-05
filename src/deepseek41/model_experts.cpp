#include "model_state.hpp"
#include "basalt/core/vmm.hpp"
#include <algorithm>
#include <cstring>
#include <map>
#include <future>

namespace basalt::deepseek41 {
void Model::Impl::initialize_cache() {
    std::string error;std::vector<uint8_t> text;
    expect(packed.file().read("experts.layout",text,error),error);
    kernels::cpu::ExpertGeometry geometry;geometry.n_embd=5120;geometry.n_ff=2304;geometry.buffers={8192,4096};geometry.swiglu_limit=10;
    expect(kernels::cpu::expert_layout_load_text(std::string(text.begin(),text.end()),40,384,error,geometry),error);
    expect(source.open(packed.expert_file(),40,384,error),error);
    const auto& layout=packed.expert_layout();
    if(options.slots>1 || core::exp_flag("ds-device-route")!=0) {
        primary_lookup=primary->alloc<unsigned long long>(40*384,MemoryKind::state);
        host_primary_lookup=std::make_unique<HostTransferStorage>(40*384*sizeof(unsigned long long),true);
        route_transport=std::make_unique<RouteTransport>();
    }
    if(core::exp_flag("ds-attention-predecode",1)!=0 || core::exp_flag("ds-decode-matrix")!=0 || core::exp_flag("ds-decode-split",1)!=0)
        attention_scratch=primary->alloc<float>(kAttentionScratch);
    if(core::exp_flag("ds-attention-check")!=0)attention_check_output=primary->alloc<float>(64*512);
    if(options.tensor_prefill) {
        primary->prepare_prefill();
        prompt_experts=std::make_unique<PrefillExperts>(*primary,layout,window_capacity);
        if(core::exp_flag("ds-index-mma",long_work?1:0)!=0)
            prompt_index=std::make_unique<PrefillIndex>(primary->stream(),[&](size_t bytes){return primary->allocate(bytes,MemoryKind::scratch);});
    }
    batch_stage=options.slots>1 && core::exp_flag("ds-batch-stage")!=0;
    if(options.stream_prefill_experts || batch_stage) {
        for(int l=0;l<40;++l)prefill_expert_stride=std::max(prefill_expert_stride,size_t(layout.blob_bytes(l)));
        prefill_expert_stride=(prefill_expert_stride+255)/256*256;
        prefill_experts=primary->allocate(4*prefill_expert_stride,MemoryKind::scratch);
    }
    if(core::exp_flag("ds-decode-matrix")!=0)primary->prepare_decode_attention(attention_scratch);
    const core::OnDevice on(0);
    size_t free=0,total=0;cuda_check(cudaMemGetInfo(&free,&total));
    const uint64_t reserve=uint64_t(options.reserve_mib)<<20;
    expect(free>reserve,"no VRAM left for target expert cache");
    uint64_t budget=free-reserve;if(options.cache_limit_bytes)budget=std::min(budget,options.cache_limit_bytes);
    uint64_t layer_sum=0;for(int l=0;l<40;++l)layer_sum+=layout.blob_bytes(l);
    const int each=int(std::min<uint64_t>(384,budget/layer_sum));
    expect(each>0,"VRAM budget cannot hold one expert per target layer");
    std::vector<int64_t> sizes;
    int extra=0;
    if(kv) {
        // A shrinking tail removes whole rounds of experts instead of emptying the last layers.
        for(int e=0;e<each;++e)for(int l=0;l<40;++l)sizes.push_back(int64_t(layout.blob_bytes(l)));
        const auto gran=core::vmm::granularity();
        uint64_t used=uint64_t(each)*layer_sum;
        while(each<384 && extra<40) {
            const auto bytes=layout.blob_bytes(extra);
            if((used+bytes+gran-1)/gran*gran>budget)break;
            sizes.push_back(int64_t(bytes));
            used+=bytes;++extra;
        }
        const auto tail=kv->bytes_for(options.context)*uint64_t(options.slots)+gran;
        cache.set_elastic_tail(int64_t((tail+gran-1)/gran*gran));
    } else for(int l=0;l<40;++l)for(int e=0;e<each;++e)sizes.push_back(int64_t(layout.blob_bytes(l)));
    expect(cache.open_sized(sizes,40,384,error),error);cache.set_per_layer_admission(!kv);
    const auto fill=[&](int l,int e) {
        const int slot=cache.admit(l,e);expect(slot>=0,"target expert admission failed");
        expect(cache.fill_slot_queued(slot,source.blob(l,e),error,int64_t(layout.blob_bytes(l))),error);
    };
    if(kv) {
        elastic_homes.resize(size_t(cache.full_slots()));
        for(int e=0;e<each;++e)for(int l=0;l<40;++l)fill(l,e);
        for(int l=0;l<extra;++l)fill(l,each);
    } else for(int l=0;l<40;++l)for(int e=0;e<each;++e)fill(l,e);
    expect(cache.sync_queued(error),error);
    std::vector<std::pair<int32_t,int32_t>> peer_pairs;
    if(options.helper_device>=0) {
        expect(helper.prepare(options.helper_device,source,40,384,error),error);
        helper.set_single_token_graphs(core::exp_flag("ds-helper-graph")!=0);
        const uint64_t peer_reserve=uint64_t(options.helper_reserve_mib)<<20;
        uint64_t peer_budget=helper.available_bytes()>peer_reserve?helper.available_bytes()-peer_reserve:0;
        if(options.cache_limit_bytes)peer_budget=std::min(peer_budget,options.cache_limit_bytes);
        for(int e=each;e<384;++e)for(int l=0;l<40;++l) {
            if(e==each && l<extra)continue;
            const uint64_t bytes=core::PeerExperts::slot_bytes(l);
            if(bytes<=peer_budget) {peer_budget-=bytes;peer_pairs.emplace_back(l,e);}
        }
        const double bench_cap=core::exp_flag("ds-bench-helper-experts");
        if(bench_cap>0 && bench_cap<double(peer_pairs.size()))peer_pairs.resize(size_t(bench_cap));
        if(peer_pairs.empty())helper.close();
        else expect(helper.fill(peer_pairs,error,true),error);
    }
    if(options.resident_experts) {
        // Exclusive swaps need a RAM landing place for each evicted expert.
        if(!options.freeze_placement)source.set_spares(8);
        source.release_mapped_cache();
        basalt::log::info("basalt deepseek41","resident expert RAM floor %llu MiB",
                          (unsigned long long)(kExpertRamFloor>>20));
        if(source.pin_cache_complement(cache,error,true,peer_pairs,-1,kExpertRamFloor,0,nullptr,
                                       core::FileExpertSource::ResidentHeadroom::exact)) {
            if(helper.valid())helper.set_exclusive(&source);
        } else basalt::log::warn("basalt deepseek41","resident expert allocation: %s",error.c_str());
    }
    basalt::log::info("basalt deepseek41","primary experts %lld, helper experts %zu",(long long)cache.resident(),peer_pairs.size());
}
void Model::Impl::refresh_primary_lookup() {
    if(!primary_lookup)return;
    auto* table=reinterpret_cast<unsigned long long*>(host_primary_lookup->data());
    for(int l=0;l<40;++l)for(int e=0;e<384;++e) {
        const int slot=primary_slot(l,e);
        table[l*384+e]=slot>=0?reinterpret_cast<unsigned long long>(cache.device_slot(slot)):0;
    }
    primary->upload(primary_lookup,table,40*384*sizeof(unsigned long long));
}
void Model::Impl::grouped_device(DeviceModel& d,Work& w,Layer& l,int count) {
    const int entries=count*l.top;
    d.zero(w.gpu_parts,size_t(entries)*5120*sizeof(float));
    plan_primary_experts(primary_lookup+l.id*384,w.ids,384,l.top,w.group_pointer,w.group_start,w.group_count,
                         w.entry_token,w.entry_destination,d.stream(),count);
    kernels::quantize_q8_1_rows(w.x,count,5120,w.quant,d.stream());
    const auto& fmt=packed.expert_layout().fmt[size_t(l.id)];
    const auto native=kernels::native_expert_layout(fmt.gu_type,fmt.d_type,5120,2304,10);
    kernels::native_expert_grouped(native,w.group_pointer,w.group_start,w.group_count,w.entry_destination,w.entry_token,
                                   entries,entries,w.quant,w.expert_scratch,w.gpu_parts,d.stream(),entries);
}
void Model::Impl::grouped(DeviceModel& d,Work& w,Layer& l,int count,bool draft) {
    const int entries=count*l.top;
    std::map<int,std::vector<int>> groups;
    for(int j=0;j<entries;++j)if(draft || w.host_kind[j]==1)groups[w.host_ids[j]].push_back(j);
    w.host_pointer.clear();w.host_start.clear();w.host_entry_token.clear();w.host_entry_destination.clear();
    w.host_start.push_back(0);
    const auto& layout=draft?packed.draft_expert_layout():packed.expert_layout();
    for(const auto& [expert,members]:groups) {
        const uint8_t* ptr=draft?draft_experts+layout.blob_offset(l.id,expert):cache.device_slot(cache.slot_of(l.id,expert));
        w.host_pointer.push_back(reinterpret_cast<unsigned long long>(ptr));
        for(int entry:members) {w.host_entry_token.push_back(entry/l.top);w.host_entry_destination.push_back(entry);}
        w.host_start.push_back(int(w.host_entry_token.size()));
    }
    w.host_group_count=int(groups.size());
    d.zero(w.gpu_parts,size_t(entries)*5120*sizeof(float));
    if(groups.empty())return;
    d.upload(w.group_pointer,w.host_pointer.data(),groups.size()*sizeof(unsigned long long));
    d.upload(w.group_start,w.host_start.data(),w.host_start.size()*sizeof(int32_t));
    d.upload(w.group_count,&w.host_group_count,sizeof(int32_t));
    d.upload(w.entry_token,w.host_entry_token.data(),w.host_entry_token.size()*sizeof(int32_t));
    d.upload(w.entry_destination,w.host_entry_destination.data(),w.host_entry_destination.size()*sizeof(int32_t));
    kernels::quantize_q8_1_rows(w.x,count,5120,w.quant,d.stream());
    const auto& fmt=layout.fmt[size_t(l.id)];
    const auto native=kernels::native_expert_layout(fmt.gu_type,fmt.d_type,5120,2304,10);
    kernels::native_expert_grouped(native,w.group_pointer,w.group_start,w.group_count,w.entry_destination,w.entry_token,
                                   entries,entries,w.quant,w.expert_scratch,w.gpu_parts,d.stream(),int64_t(groups.size()));
}
void Model::Impl::expert_layer(DeviceModel& d,Work& w,Layer& l,int count,bool draft) {
    ProfileScope timer(profile,draft?"draft_expert":"target_expert");
    const core::OnDevice on(d.device());const int entries=count*l.top;
    const auto shared=[&] {shared_experts(d,w,l,count);};
    if(draft && core::exp_flag("ds-draft-device-body",1)!=0) {
        expect(count<=6,"invalid device draft expert window");shared();
        d.zero(w.gpu_parts,size_t(entries)*5120*sizeof(float));
        plan_primary_experts(draft_lookup+l.id*128,w.ids,128,l.top,w.group_pointer,w.group_start,w.group_count,
                             w.entry_token,w.entry_destination,d.stream(),count);
        kernels::quantize_q8_1_rows(w.x,count,5120,w.quant,d.stream());
        const auto& fmt=packed.draft_expert_layout().fmt[size_t(l.id)];
        const auto native=kernels::native_expert_layout(fmt.gu_type,fmt.d_type,5120,2304,10);
        kernels::native_expert_grouped(native,w.group_pointer,w.group_start,w.group_count,w.entry_destination,w.entry_token,
                                       entries,entries,w.quant,w.expert_scratch,w.gpu_parts,d.stream(),entries);
        combine_experts(w.gpu_parts,nullptr,w.weights,w.shared,w.branch,5120,l.top,count,d.stream());
        return;
    }
    const bool early=count<=6 && !draft && route_transport && (!drafter || core::exp_flag("ds-verify-route",1)!=0);
    {
    ProfileScope routed(profile,draft?"draft_route_wait":"target_route_wait");
    const auto copy_stream=early?route_transport->stream:static_cast<cudaStream_t>(d.stream());
    if(early) {
        cuda_check(cudaEventRecord(route_transport->ready,static_cast<cudaStream_t>(d.stream())));
        cuda_check(cudaStreamWaitEvent(copy_stream,route_transport->ready));
    }
    const size_t transfer=count==1 && !draft?profile.begin_async("route_transfer",copy_stream):size_t(-1);
    cuda_check(cudaMemcpyAsync(w.host_ids.data(),w.ids,size_t(entries)*sizeof(int32_t),cudaMemcpyDeviceToHost,
                               copy_stream));
    if(!draft)cuda_check(cudaMemcpyAsync(w.host_x.data(),w.x,size_t(count)*5120*sizeof(float),cudaMemcpyDeviceToHost,
                                         copy_stream));
    profile.end_async(transfer,copy_stream);
    if(early) {
        resident_experts(d,w,l,count);
        cuda_check(cudaStreamSynchronize(copy_stream));
    } else d.synchronize();
    }
    if(draft) {
        shared();
        grouped(d,w,l,count,true);
        combine_experts(w.gpu_parts,nullptr,w.weights,w.shared,w.branch,5120,l.top,count,d.stream());
        return;
    }
    const bool skip_host=early && core::exp_flag("ds-decode-skip-host-tail")!=0;
    const bool sparse_host=count<=6 && w.mapped_parts && core::exp_flag("ds-decode-sparse-host-parts")!=0;
    uint64_t host_rows=0;
    bool all_primary=skip_host;
    if(!skip_host && !sparse_host)std::fill(w.host_parts.begin(),w.host_parts.begin()+size_t(entries)*5120,0.f);
    const bool cpu_direct=core::exp_flag("ds-cpu-direct-output")!=0;
    std::vector<CpuExperts::Job> jobs;std::vector<int> destinations;
    for(int j=0;j<entries;++j) {
        const int expert=w.host_ids[j];expect(expert>=0 && expert<384,"invalid routed expert");
        usage[size_t(l.id)*384+expert]+=1;
        int kind=primary_slot(l.id,expert)>=0?1:helper.valid() && helper.has(l.id,expert)?2:0;
        if(!prefill_active)++decode_expert_routes[size_t(kind)];
        if(!kind && prefill_streaming && count>11) {kind=3;++execution.streamed_entries;}
        w.host_kind[j]=kind;
        all_primary=all_primary && kind==1;
        if(sparse_host && kind!=1)host_rows|=uint64_t{1}<<j;
        if(kind==1)++execution.primary_entries;
        if(kind==0)++execution.cpu_entries;
        if(!kind) {const auto* blob=source.blob(l.id,expert);expect(blob,"missing CPU expert");jobs.push_back({blob,j/l.top,cpu_direct?w.host_parts.data()+size_t(j)*5120:nullptr});
            if(!cpu_direct)destinations.push_back(j);}
    }
    std::string error;
    if(all_primary) {
        ProfileScope skipped(profile,"all_primary_tail");
        // The resident expert work was already queued by the early path.
        combine_experts(w.gpu_parts,nullptr,w.weights,w.shared,w.branch,5120,l.top,count,d.stream());
        return;
    }
    if(skip_host && !sparse_host)std::fill(w.host_parts.begin(),w.host_parts.begin()+size_t(entries)*5120,0.f);
    // Keep the existing helper's finite staging buffers; larger prompt windows use several helper batches.
    const int helper_batch=kernels::cpu::MAXT*10/l.top;
    std::future<void> helper_work;
    const bool direct=(count==1 && w.mapped_parts) || sparse_host;
    if(helper.valid()) {
        ProfileScope submitted(profile,"helper_launch");
        if(count>helper_batch)helper_work=std::async(std::launch::async,[&] {
            std::string peer_error;
            for(int first=0;first<count;first+=helper_batch) {
                expect(helper.launch(l.id,w.host_x.data()+size_t(first)*5120,w.host_ids.data()+first*l.top,
                                     std::min(helper_batch,count-first),l.top,w.host_kind.data()+first*l.top,peer_error),peer_error);
                expect(helper.finish(w.host_parts.data()+size_t(first)*l.top*5120,peer_error),peer_error);
            }
        });
        else expect(helper.launch(l.id,w.host_x.data(),w.host_ids.data(),count,l.top,w.host_kind.data(),error,
                                  direct?w.host_parts.data():nullptr),error);
    }
    // Shared experts do not feed routing; overlap independent work after helper dispatch.
    if(!early) {
    if(prefill_tensor && count>11) {
        shared();
        GpuProfileScope gpu_time(profile,"prefill_expert_pipeline",d.stream());
        prompt_experts->run(l.id,w.x,w.host_ids,w.host_kind,count,l.top,cache,source,w.gpu_parts,execution);
    }
    else {
        const auto gpu=[&]{shared();grouped(d,w,l,count,false);};
        if(count==1 && !drafter && !profile.gpu_enabled && core::exp_flag("ds-graph")!=0) {
            if(!l.expert_graph)l.expert_graph=std::make_unique<UpdateGraph>();
            l.expert_graph->run(d.stream(),gpu,&profile,"shared_primary_graph");
        } else gpu();
        if(prefill_streaming && count>11)streamed(d,w,l,count);
    }
    }
    std::vector<float> result;
    {ProfileScope computed(profile,"cpu_experts");result=cpu.run_window(packed.expert_layout().fmt[size_t(l.id)],w.host_x.data(),count,jobs);}
    for(size_t j=0;j<destinations.size();++j)std::memcpy(w.host_parts.data()+size_t(destinations[j])*5120,result.data()+j*5120,5120*sizeof(float));
    if(helper.valid()) {
        ProfileScope waited(profile,"helper_finish_wait");
        if(helper_work.valid())helper_work.get();
        else expect(helper.finish(w.host_parts.data(),error),error);
    }
    AsyncGpuScope combined(count==1?&profile:nullptr,"combine_transfer",d.stream());
    if(sparse_host)
        combine_experts_masked(w.gpu_parts,w.mapped_parts,w.weights,w.shared,w.branch,5120,l.top,count,host_rows,d.stream());
    else {
        if(!direct)d.upload(w.cpu_parts,w.host_parts.data(),size_t(entries)*5120*sizeof(float));
        combine_experts(w.gpu_parts,direct?w.mapped_parts:w.cpu_parts,w.weights,w.shared,w.branch,5120,l.top,count,d.stream());
    }
}
void Model::Impl::streamed(DeviceModel& d,Work& w,Layer& l,int count) {
    ProfileScope timer(profile,"prefill_streamed_experts");
    std::map<int,std::vector<int>> groups;
    for(int j=0;j<count*l.top;++j)if(w.host_kind[j]==3)groups[w.host_ids[j]].push_back(j);
    if(groups.empty())return;
    d.synchronize();
    kernels::quantize_q8_1_rows(w.x,count,5120,w.quant,d.stream());
    const auto& fmt=packed.expert_layout().fmt[size_t(l.id)];
    const auto native=kernels::native_expert_layout(fmt.gu_type,fmt.d_type,5120,2304,10);
    const size_t bytes=packed.expert_layout().blob_bytes(l.id);
    auto next=groups.begin();
    while(next!=groups.end()) {
        w.host_pointer.clear();w.host_start={0};w.host_entry_token.clear();w.host_entry_destination.clear();
        for(int slot=0;slot<4 && next!=groups.end();++slot,++next) {
            const uint8_t* blob=source.blob(l.id,next->first);expect(blob,"missing streamed expert");
            auto* dst=prefill_experts+size_t(slot)*prefill_expert_stride;
            d.upload(dst,blob,bytes);execution.streamed_bytes+=bytes;
            w.host_pointer.push_back(reinterpret_cast<unsigned long long>(dst));
            for(int entry:next->second) {w.host_entry_token.push_back(entry/l.top);w.host_entry_destination.push_back(entry);}
            w.host_start.push_back(int(w.host_entry_token.size()));
        }
        w.host_group_count=int(w.host_pointer.size());
        d.upload(w.group_pointer,w.host_pointer.data(),w.host_pointer.size()*sizeof(unsigned long long));
        d.upload(w.group_start,w.host_start.data(),w.host_start.size()*sizeof(int32_t));
        d.upload(w.group_count,&w.host_group_count,sizeof(int32_t));
        d.upload(w.entry_token,w.host_entry_token.data(),w.host_entry_token.size()*sizeof(int32_t));
        d.upload(w.entry_destination,w.host_entry_destination.data(),w.host_entry_destination.size()*sizeof(int32_t));
        const int entries=int(w.host_entry_token.size());
        kernels::native_expert_grouped(native,w.group_pointer,w.group_start,w.group_count,w.entry_destination,w.entry_token,
                                       w.host_group_count,entries,w.quant,w.expert_scratch,w.gpu_parts,d.stream(),w.host_group_count);
        // All consumers finish before the finite staging slots and metadata are reused.
        d.synchronize();
    }
}
} // namespace basalt::deepseek41
