#include "model_state.hpp"
#include <algorithm>
#include <numeric>

namespace basalt::deepseek41 {
void Model::Impl::initialize_slots() {
    sequences.resize(size_t(options.slots));
    for(int slot=1;slot<options.slots;++slot) {
        auto& s=sequences[size_t(slot)];
        s.history=s.saved_history=hasher.initial_history();
        s.main_work=std::make_unique<Work>(*primary,options.context,false,window_capacity,options.tensor_prefill);
        for(int l=0;l<40;++l)s.layers.push_back(make_layer(*primary,l,false));
        initialize_kv(s);
        if(drafter) {
            s.draft_work=std::make_unique<Work>(*drafter,options.context,true,Shape::draft_window);
            for(int l=0;l<3;++l)s.draft_layers.push_back(make_layer(*drafter,l,true));
            std::vector<int32_t> selection(5*512,0);
            for(int t=0;t<5;++t)for(int j=0;j<5;++j)selection[t*512+j]=j;
            drafter->upload(s.draft_work->selected,selection.data(),selection.size()*sizeof(int32_t));
            drafter->synchronize();
        }
        s.host_engram.resize(size_t(window_capacity)*48*144);
        s.host_embedding.resize(size_t(window_capacity)*Shape::hidden);
        s.row_ids.resize(size_t(window_capacity)*48);
    }
    primary->synchronize();
    if(options.slots==1)return;
    const size_t rows=size_t(options.slots)*6,nx=rows*Shape::hidden,np=nx*6,ni=rows*6;
    batch_transfer=std::make_unique<HostTransferStorage>((nx+np)*sizeof(float)+ni*sizeof(int32_t),true);
    auto* p=batch_transfer->data();
    batch_x={reinterpret_cast<float*>(p),nx};batch_parts={reinterpret_cast<float*>(p+nx*sizeof(float)),np};
    batch_ids={reinterpret_cast<int32_t*>(p+(nx+np)*sizeof(float)),ni};batch_kind.resize(ni);
    basalt::log::info("basalt deepseek41","slots %d, context per slot %d, shared expert tiers",options.slots,options.context);
}
void Model::Impl::select_slot(int slot) {
    expect(slot>=0 && slot<options.slots,"invalid slot");
    if(slot==active_slot)return;
    std::swap(static_cast<SequenceState&>(*this),sequences[size_t(active_slot)]);
    std::swap(static_cast<SequenceState&>(*this),sequences[size_t(slot)]);
    active_slot=slot;
}
void Model::Impl::forward_batch(const std::vector<int>& slots,const std::vector<std::vector<int32_t>>& tokens) {
    ProfileScope timer(profile,"target_batch_forward");
    expect(slots.size()==tokens.size() && !slots.empty() && slots.size()<=size_t(options.slots),"invalid forward batch");
    expect(primary_lookup && route_transport && !prefill_active,"decode batch requires device routing");
    std::vector<int> counts;
    ++execution.batch_rounds;
    try {
        const core::OnDevice on(0);
        for(size_t i=0;i<slots.size();++i) {
            select_slot(slots[i]);
            expect(!tokens[i].empty() && tokens[i].size()<=6,"invalid batch verify width");
            std::string error;
            expect(fit_kv(pos+int(tokens[i].size()),error),error);
            begin_forward(tokens[i]);counts.push_back(int(tokens[i].size()));
        }
        for(int layer=0;layer<40;++layer) {
            for(size_t i=0;i<slots.size();++i) {
                select_slot(slots[i]);auto& w=*main_work;auto& l=layers[size_t(layer)];
                if(l.engram)engram(l,counts[i]);
                if(layer>=37)stream_mean(w.h,w.anchors,Shape::hidden,counts[i],15360,size_t(layer-37)*Shape::hidden,primary->stream());
                enqueue_prefix(*primary,w,l,pos,counts[i],false);
                issue_experts(*primary,w,l,counts[i]);
            }
            batch_experts(slots,counts,layer);
            for(size_t i=0;i<slots.size();++i) {
                select_slot(slots[i]);auto& w=*main_work;
                hc_expand(w.branch,w.h,w.post,w.comb,w.h,Shape::hidden,counts[i],primary->stream());
                std::swap(w.pre,w.f_pre);
            }
        }
        for(int slot:slots) {select_slot(slot);finish_forward(LogitsMode::all,false);}
    } catch(...) {broken=true;throw;}
}
std::vector<DecodeResult> Model::decode_batch(const std::vector<DecodeRequest>& requests) {
    if(requests.empty())return {};
    expect(requests.size()<=size_t(slots()),"too many decode requests");
    std::vector<bool> occupied(size_t(slots()),false);
    for(const auto& r:requests) {
        expect(r.slot>=0 && r.slot<slots() && !occupied[size_t(r.slot)],"duplicate or invalid decode slot");
        occupied[size_t(r.slot)]=true;
        expect(r.anchor>=0 && r.anchor<Shape::vocab && r.max_tokens>0,"invalid decode request");
        const auto& s=r.slot==impl_->active_slot?static_cast<const SequenceState&>(*impl_):impl_->sequences[size_t(r.slot)];
        expect(!impl_->broken && !s.pending && s.pos<context(),"unfinished or full decode slot");
    }
    struct Restore {Model& model;int slot;~Restore(){model.select_slot(slot);}} restore{*this,current_slot()};
    if(requests.size()==1) {
        const auto& r=requests[0];select_slot(r.slot);return {decode(r.anchor,r.parameters,r.max_tokens,r.stop_token,r.proposals,r.filter,r.penalty_tail)};
    }
    std::vector<int> ids;
    std::vector<std::vector<int32_t>> inputs;
    std::vector<std::vector<float>> confidence(requests.size());
    const bool adaptive=impl_->drafter && core::exp_flag("ds-adaptive-window",1)!=0;
    for(size_t i=0;i<requests.size();++i) {
        const auto& r=requests[i];select_slot(r.slot);
        expect(!impl_->pending && !impl_->broken,"unfinished decode slot");
        const int limit=std::min({r.max_tokens,6,context()-position()});
        expect(limit>0,"context capacity reached");
        std::vector<int32_t> input={r.anchor};
        if(!r.proposals.empty() && r.anchor!=r.stop_token) {
            expect(!impl_->drafter,"external proposals require DSpark disabled");
            input.insert(input.end(),r.proposals.begin(),r.proposals.begin()+std::min(r.proposals.size(),size_t(limit-1)));
        } else if(impl_->drafter && limit>1 && position()<=context()-5 && r.anchor!=r.stop_token) {
            const auto proposals=draft(r.anchor);
            const int count=adaptive?impl_->draft_window.choose(proposals.confidence,limit-1):limit-1;
            input.insert(input.end(),proposals.tokens.begin(),proposals.tokens.begin()+count);
            if(adaptive)confidence[i]=proposals.confidence;
        }
        ids.push_back(r.slot);inputs.push_back(std::move(input));
    }
    impl_->forward_batch(ids,inputs);
    std::vector<DecodeResult> results(requests.size());
    for(size_t i=0;i<requests.size();++i) {
        select_slot(requests[i].slot);
        const auto& r=requests[i];
        const auto picks=sample(r.parameters,int(inputs[i].size()),r.penalty_tail,inputs[i]);
        results[i]=verify_picks(inputs[i],picks,r.stop_token,r.filter);
    }
    // No expert ownership changes until every slot's GPU forward and target sampling have completed.
    for(size_t i=0;i<requests.size();++i) {
        select_slot(requests[i].slot);commit(int(results[i].tokens.size()));
        if(adaptive)impl_->draft_window.observe(confidence[i],results[i].proposed,results[i].accepted);
    }
    return results;
}
} // namespace basalt::deepseek41
