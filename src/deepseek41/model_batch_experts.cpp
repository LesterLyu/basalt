#include "model_state.hpp"
#include <algorithm>
#include <cstring>
#include <numeric>

namespace basalt::deepseek41 {
void Model::Impl::shared_experts(DeviceModel& d,Work& w,Layer& l,int count) {
    d.linear(*l.shared_gate,w.x,w.gate,count);d.linear(*l.shared_up,w.x,w.up,count);
    kernels::native_swiglu_f32(w.gate,w.up,w.activated,count*2304,10,d.stream());
    d.linear(*l.shared_down,w.activated,w.shared,count);
}
void Model::Impl::resident_experts(DeviceModel& d,Work& w,Layer& l,int count) {
    const auto gpu=[&]{shared_experts(d,w,l,count);grouped_device(d,w,l,count);};
    if(!profile.gpu_enabled && core::exp_flag("ds-graph")!=0) {
        auto& graph=count==1?l.expert_graph:l.verify_graphs[size_t(count-2)];
        if(!graph)graph=std::make_unique<UpdateGraph>();
        graph->run_static(d.stream(),gpu,&profile,"shared_primary_graph");
    } else {AsyncGpuScope timing(&profile,"shared_primary_submissions",d.stream());gpu();}
}
void Model::Impl::issue_experts(DeviceModel& d,Work& w,Layer& l,int count) {
    const auto stream=route_transport->stream;
    cuda_check(cudaEventRecord(route_transport->ready,static_cast<cudaStream_t>(d.stream())));
    cuda_check(cudaStreamWaitEvent(stream,route_transport->ready));
    cuda_check(cudaMemcpyAsync(w.host_ids.data(),w.ids,size_t(count)*l.top*sizeof(int32_t),cudaMemcpyDeviceToHost,stream));
    cuda_check(cudaMemcpyAsync(w.host_x.data(),w.x,size_t(count)*Shape::hidden*sizeof(float),cudaMemcpyDeviceToHost,stream));
    resident_experts(d,w,l,count);
}
void Model::Impl::batch_staged_experts_for_slot(Work& w,Layer& l,int count,const int* staged,int nstaged) {
    w.host_pointer.clear();w.host_start.assign(1,0);
    w.host_entry_token.clear();w.host_entry_destination.clear();
    for(int i=0;i<nstaged;++i) {
        const auto first=w.host_entry_token.size();
        for(int j=0;j<count*l.top;++j)if(w.host_ids[j]==staged[i]) {
            w.host_entry_token.push_back(j/l.top);
            w.host_entry_destination.push_back(j);
        }
        if(w.host_entry_token.size()==first)continue;
        w.host_pointer.push_back(reinterpret_cast<unsigned long long>(prefill_experts+size_t(i)*prefill_expert_stride));
        w.host_start.push_back(int(w.host_entry_token.size()));
    }
    if(w.host_pointer.empty())return;
    auto& d=*primary;
    w.host_group_count=int(w.host_pointer.size());
    const int entries=int(w.host_entry_token.size());
    d.upload(w.group_pointer,w.host_pointer.data(),w.host_pointer.size()*sizeof(unsigned long long));
    d.upload(w.group_start,w.host_start.data(),w.host_start.size()*sizeof(int32_t));
    d.upload(w.group_count,&w.host_group_count,sizeof(int32_t));
    d.upload(w.entry_token,w.host_entry_token.data(),entries*sizeof(int32_t));
    d.upload(w.entry_destination,w.host_entry_destination.data(),entries*sizeof(int32_t));
    const auto& fmt=packed.expert_layout().fmt[size_t(l.id)];
    const auto native=kernels::native_expert_layout(fmt.gu_type,fmt.d_type,Shape::hidden,Shape::ff,10);
    // The resident launch has already quantized these rows and cleared the output.
    kernels::native_expert_grouped(native,w.group_pointer,w.group_start,w.group_count,w.entry_destination,w.entry_token,
                                   w.host_group_count,entries,w.quant,w.expert_scratch,w.gpu_parts,d.stream(),w.host_group_count);
}
void Model::Impl::batch_experts(const std::vector<int>& slots,const std::vector<int>& counts,int layer) {
    ProfileScope timer(profile,"batch_experts");
    {ProfileScope waited(profile,"batch_route_wait");cuda_check(cudaStreamSynchronize(route_transport->stream));}
    constexpr int top=6,h=Shape::hidden;
    const int rows=std::accumulate(counts.begin(),counts.end(),0),entries=rows*top;
    expect(size_t(rows)*h<=batch_x.size(),"expert batch exceeds transfer capacity");
    std::fill(batch_parts.begin(),batch_parts.begin()+size_t(entries)*h,0.f);
    std::vector<CpuExperts::Job> jobs;std::vector<int> destinations;
    std::array<int,384> frequency{};
    std::array<int,4> staged{};
    int nstaged=0;
    const bool stage=batch_stage && slots.size()>1;
    const auto cpu_job=[&](int expert,int dst) {
        ++execution.cpu_entries;const auto* blob=source.blob(layer,expert);expect(blob,"missing CPU expert");
        jobs.push_back({blob,dst/top});destinations.push_back(dst);
    };
    int row=0;
    for(size_t i=0;i<slots.size();++i) {
        select_slot(slots[i]);auto& w=*main_work;
        std::memcpy(batch_x.data()+size_t(row)*h,w.host_x.data(),size_t(counts[i])*h*sizeof(float));
        std::memcpy(batch_ids.data()+row*top,w.host_ids.data(),size_t(counts[i])*top*sizeof(int32_t));
        for(int j=0;j<counts[i]*top;++j) {
            const int expert=w.host_ids[j],dst=row*top+j;
            expect(expert>=0 && expert<384,"invalid routed expert");
            usage[size_t(layer)*384+expert]+=1;
            const int kind=primary_slot(layer,expert)>=0?1:helper.valid() && helper.has(layer,expert)?2:0;
            ++decode_expert_routes[size_t(kind)];
            batch_kind[size_t(dst)]=kind;
            if(kind==1)++execution.primary_entries;
            if(!kind) {
                if(stage)++frequency[size_t(expert)];
                else cpu_job(expert,dst);
            }
        }
        row+=counts[i];
    }
    if(stage) {
        const auto stream=route_transport->stream;
        const auto bytes=packed.expert_layout().blob_bytes(layer);
        while(nstaged<int(staged.size())) {
            const auto best=std::max_element(frequency.begin(),frequency.end());
            if(*best<2)break;
            const int expert=int(best-frequency.begin()),members=*best;
            *best=0;
            // Only pinned RAM is eligible. Leave other misses to the CPU.
            if(!source.pinned(layer,expert))continue;
            const auto* blob=source.resident_blob(layer,expert);
            if(!blob)continue;
            cuda_check(cudaMemcpyAsync(prefill_experts+size_t(nstaged)*prefill_expert_stride,blob,bytes,cudaMemcpyHostToDevice,stream));
            staged[size_t(nstaged++)]=expert;
            ++batch_staged_experts;batch_staged_rows+=members;
        }
        if(nstaged) {
            cuda_check(cudaEventRecord(route_transport->ready,stream));
            cuda_check(cudaStreamWaitEvent(static_cast<cudaStream_t>(primary->stream()),route_transport->ready));
            for(size_t i=0;i<slots.size();++i) {
                select_slot(slots[i]);
                batch_staged_experts_for_slot(*main_work,layers[size_t(layer)],counts[i],staged.data(),nstaged);
            }
        }
        for(int j=0;j<entries;++j)if(!batch_kind[size_t(j)]) {
            const int expert=batch_ids[j];
            if(std::find(staged.begin(),staged.begin()+nstaged,expert)==staged.begin()+nstaged)cpu_job(expert,j);
        }
    }
    std::string error;
    const int helper_batch=kernels::cpu::MAXT*10/top;
    if(helper.valid()) {
        ProfileScope launch(profile,"batch_helper_launch");
        if(rows<=helper_batch)expect(helper.launch(layer,batch_x.data(),batch_ids.data(),rows,top,batch_kind.data(),error),error);
        else for(int first=0;first<rows;first+=helper_batch) {
            expect(helper.launch(layer,batch_x.data()+size_t(first)*h,batch_ids.data()+first*top,
                                 std::min(helper_batch,rows-first),top,batch_kind.data()+first*top,error),error);
            expect(helper.finish(batch_parts.data()+size_t(first)*top*h,error),error);
        }
    }
    std::vector<float> result;
    {ProfileScope computed(profile,"batch_cpu_experts");
    result=cpu.run_window(packed.expert_layout().fmt[size_t(layer)],batch_x.data(),rows,jobs);}
    for(size_t i=0;i<jobs.size();++i)
        std::memcpy(batch_parts.data()+size_t(destinations[i])*h,result.data()+i*h,h*sizeof(float));
    if(helper.valid() && rows<=helper_batch) {
        ProfileScope waited(profile,"batch_helper_finish");expect(helper.finish(batch_parts.data(),error),error);
    }
    row=0;
    for(size_t i=0;i<slots.size();++i) {
        select_slot(slots[i]);auto& w=*main_work;auto& d=*primary;
        const size_t bytes=size_t(counts[i])*top*h*sizeof(float);
        std::memcpy(w.host_parts.data(),batch_parts.data()+size_t(row)*top*h,bytes);
        const float* parts=w.mapped_parts;
        if(!parts) {d.upload(w.cpu_parts,w.host_parts.data(),bytes);parts=w.cpu_parts;}
        combine_experts(w.gpu_parts,parts,w.weights,w.shared,w.branch,h,top,counts[i],d.stream());
        row+=counts[i];
    }
}
} // namespace basalt::deepseek41
