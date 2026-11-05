#include "model_state.hpp"
#include <algorithm>
#include <cmath>
#include <array>

namespace basalt::deepseek41 {
void Model::Impl::transfer(DeviceModel& dst,void* to,DeviceModel& src,const void* from,size_t bytes) {
    src.synchronize();
    const core::OnDevice on(dst.device());
    if(dst.device()==src.device())dst.copy(to,from,bytes);
    else {
        std::string error;
        auto& copy=dst.device()==0?to_primary:to_draft;
        expect(copy.copy(to,from,bytes,static_cast<cudaStream_t>(dst.stream()),error),error);
    }
    dst.synchronize();
}
void Model::Impl::seed_draft(int p0,int count) {
    ProfileScope timer(profile,"draft_seed");
    if(helper.valid())helper.apply_pending(true);
    auto& d=*drafter;auto& w=*draft_work;
    const core::OnDevice on(d.device());
    GpuProfileScope gpu_time(profile,"draft_prefix_seed",d.stream());
    for(int first=0;first<count;first+=Shape::draft_window) {
        const int rows=std::min(Shape::draft_window,count-first);
        {ProfileScope copied(profile,"draft_seed_transfer");
        transfer(d,w.anchors,*primary,main_work->anchors+size_t(first)*15360,size_t(rows)*15360*sizeof(float));}
        d.linear("fc.weight",w.anchors,draft_main_x,rows);
        RowFinish f;f.norm=d.f32("enc.output_norm.weight",{5120});
        finish_rows(draft_main_x,rows,5120,5120,f,nullptr,0,d.stream());
        for(auto& l:draft_layers) {
            d.linear(*l.kv,draft_main_x,w.kv,rows);
            f={};f.norm=l.kv_norm;f.frequencies=w.freq_window;f.rotary=64;f.first_position=p0+first;f.quant=CacheFormat::window_fp8;
            finish_rows(w.kv,rows,512,512,f,l.extended+128*528,528,d.stream());
            if(core::exp_flag("ds-draft-device-body",1)!=0)update_window_ring(l.ring,l.extended,p0+first,rows,528,d.stream());
            else for(int t=0;t<rows;++t)d.copy(l.ring+((p0+first+t)%128)*528,l.extended+(128+t)*528,528);
        }
        d.synchronize();
    }
}
DraftResult Model::Impl::draft(int32_t anchor) {
    ProfileScope timer(profile,"draft_total");
    expect(!broken && !pending && drafter,"DSpark requires a committed target prefix");
    expect(anchor>=0 && anchor<Shape::vocab && pos<=options.context-5,"invalid draft anchor or context");
    try {
        if(helper.valid()) {
            ProfileScope wait(profile,"draft_helper_pending");
            helper.apply_pending(true);
        }
        primary->synchronize();auto& d=*drafter;auto& w=*draft_work;
        const core::OnDevice on(d.device());
        std::vector<int32_t> block_tokens(5,packed.contract().noise_token);block_tokens[0]=anchor;
        embeddings(block_tokens,d,w);
        RowFinish f;f.norm=d.f32("output_norm.weight",{5120});
        const auto body=[&] {
            for(auto& l:draft_layers)block(d,w,l,pos,5,true);
            if(draft_gpu_stages)draft_gpu_stages->record(int(draft_layers.size()),0,d.stream());
            hc_reduce(w.h,w.pre,nullptr,w.head_x,5120,5,1e-20f,d.stream());
            d.copy(w.x,w.head_x,5*5120*sizeof(float));
            finish_rows(w.x,5,5120,5120,f,nullptr,0,d.stream());
        };
        {
        ProfileScope submitted(profile,"draft_body_and_hidden_transfer");
        if(core::exp_flag("ds-draft-device-body",1)!=0 && core::exp_flag("ds-graph")!=0 && !profile.gpu_enabled) {
            if(!draft_body_graph)draft_body_graph=std::make_unique<UpdateGraph>();
            draft_body_graph->run(d.stream(),body);
        } else body();
        // Cache placement belongs to the target. Draft weights and scratch
        // are separate, so the target can finish its swaps during this graph.
        finish_deferred_adapt();
        transfer(*primary,main_work->head_x,d,w.x,5*5120*sizeof(float));
        if(draft_gpu_stages)draft_gpu_stages->collect();
        }
        {
        ProfileScope head(profile,"draft_head_and_logits_transfer");
        primary->linear("output.weight",main_work->head_x,main_work->logits,5);
        logit_rows=0;
        transfer(d,draft_logits,*primary,main_work->logits,5*Shape::vocab*sizeof(float));
        }
        const auto& embedding=d.tensor("markov_w1.weight",{256,Shape::vocab});
        expect(embedding.type==30,"Markov embedding must be BF16");
        const auto& transition=d.tensor("markov_w2.weight",{256,Shape::vocab});
        DraftResult result;int32_t previous=anchor;
        ProfileScope markov(profile,"draft_markov");
        if(core::exp_flag("ds-draft-device-picks",1)!=0) {
            d.upload(draft_picks,&anchor,sizeof(anchor));d.zero(draft_picks+6,sizeof(int32_t));
            const auto submit=[&] {
                for(int t=0;t<5;++t) {
                    gather_bf16_row(reinterpret_cast<const uint16_t*>(embedding.data),draft_picks+t,Shape::vocab,256,markov_x,d.stream());
                    auto* logits=draft_logits+size_t(t)*Shape::vocab;
                    if(markov_cache)
                        cached_markov_add(reinterpret_cast<const uint16_t*>(transition.data),markov_x,draft_picks+t,
                                          markov_cache_keys,markov_cache,logits,d.stream());
                    else {
                        d.linear(transition,markov_x,markov_bias,1);
                        add_rows(logits,markov_bias,Shape::vocab,d.stream());
                    }
                    finite_argmax(draft_logits+size_t(t)*Shape::vocab,Shape::vocab,draft_picks+t+1,draft_picks+6,
                                  draft_pick_workspace,finite_argmax_workspace_bytes(Shape::vocab),d.stream());
                    d.copy(confidence_input+size_t(t)*(5120+256),w.head_x+t*5120,5120*sizeof(float));
                    d.copy(confidence_input+size_t(t)*(5120+256)+5120,markov_x,256*sizeof(float));
                }
                d.linear("conf_proj.weight",confidence_input,confidence_output,5);
            };
            if(core::exp_flag("ds-graph")!=0 && !profile.gpu_enabled) {
                if(!draft_pick_graph)draft_pick_graph=std::make_unique<UpdateGraph>();
                draft_pick_graph->run_static(d.stream(),submit,nullptr,nullptr);
            } else submit();
            std::array<int32_t,7> picks{};d.read(picks.data(),draft_picks,sizeof(picks));
            expect(!picks[6],"nonfinite DSpark logits");
            const auto confidence=d.read(confidence_output,5);
            for(int t=0;t<5;++t) {
                expect(picks[t+1]>=0 && picks[t+1]<Shape::vocab,"invalid DSpark draft token");
                expect(std::isfinite(confidence[t]),"nonfinite DSpark confidence");
                result.tokens.push_back(picks[t+1]);result.confidence.push_back(1.f/(1.f+std::exp(-confidence[t])));
            }
            ++execution.draft_rounds;return result;
        }
        for(int t=0;t<5;++t) {
            kernels::iq_dequant_f32(30,embedding.data+size_t(previous)*512,256,markov_x,d.stream());
            d.linear("markov_w2.weight",markov_x,markov_bias,1);
            add_rows(draft_logits+size_t(t)*Shape::vocab,markov_bias,Shape::vocab,d.stream());
            const auto logits=d.read(draft_logits+size_t(t)*Shape::vocab,Shape::vocab);
            expect(std::all_of(logits.begin(),logits.end(),[](float x){return std::isfinite(x);}),"nonfinite DSpark logits");
            const int32_t next=int32_t(std::max_element(logits.begin(),logits.end())-logits.begin());
            d.copy(confidence_input,w.head_x+t*5120,5120*sizeof(float));
            d.copy(confidence_input+5120,markov_x,256*sizeof(float));
            d.linear("conf_proj.weight",confidence_input,confidence_output,1);
            const float logit=d.read(confidence_output,1)[0];
            expect(std::isfinite(logit),"nonfinite DSpark confidence");
            result.tokens.push_back(next);result.confidence.push_back(1.f/(1.f+std::exp(-logit)));previous=next;
        }
        d.synchronize();++execution.draft_rounds;return result;
    } catch(...) {broken=true;throw;}
}
} // namespace basalt::deepseek41
