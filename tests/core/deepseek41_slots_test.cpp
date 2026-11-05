#include "basalt/deepseek41/model.hpp"
#include "basalt/core/diag.hpp"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <tuple>

namespace ds=basalt::deepseek41;
static void check(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
static void snapshot_test(ds::Model& model) {
    const std::vector<int32_t> cycle={0,128803,8420,304,223,397,16,128804,128822};
    const std::vector<int32_t> suffix={219,4321,42,128799,87};
    for(int position:{97,145}) {
        std::vector<int32_t> prompt;
        for(int i=0;i<position;++i)prompt.push_back(cycle[size_t(i)%cycle.size()]);
        model.select_slot(0);model.reset();model.prefill(prompt);
        auto snapshot=model.save_prefix();
        check(snapshot.bytes()==model.prefix_snapshot_bytes(),"snapshot size mismatch");
        const auto run=[&] {
            const auto logits=model.prefill(suffix,0,ds::LogitsMode::all).logits;
            auto next=model.sample({})[0];
            const auto draft=model.draft(next);
            std::vector<int32_t> ids;
            while(ids.size()<12) {
                const auto r=model.decode_greedy(next,std::min(6,12-int(ids.size())));
                ids.insert(ids.end(),r.tokens.begin(),r.tokens.end());next=r.next;
            }
            return std::tuple{logits,draft.tokens,draft.confidence,ids};
        };
        const auto live=run();
        model.select_slot(1);model.reset();model.prefill(cycle);
        model.restore_prefix(snapshot);
        check(model.position()==position,"restored cursor differs");
        check(run()==live,"snapshot changed logits, DSpark proposals, confidence or target continuation");
        model.select_slot(0);
        check(model.position()==position+int(suffix.size())+12,"restoring slot one changed slot zero");
        std::printf("snapshot at %d: %zu bytes; cross-slot logits, DSpark and 12 IDs identical\n",position,snapshot.bytes());
        std::fflush(stdout);
    }
}
static void grammar_window_test() {
    const std::vector<int32_t> input={10,20,30};
    const std::vector<int32_t> picks={20,30,40};
    std::vector<int> seen;
    auto filter=[&](int row,int32_t token) {seen.push_back(row);return token;};
    const auto one=ds::Model::verify_picks(std::span(input).first(1),std::span(picks).first(1),1,filter);
    check(one.tokens==std::vector<int32_t>{10} && one.next==20 && seen==std::vector<int>{0},
          "a full one-token window did not keep its next draw");
    seen.clear();
    const auto full=ds::Model::verify_picks(input,picks,1,filter);
    check(full.tokens==input && full.next==40 && seen==std::vector<int>{0,1,2},
          "a full draft window did not keep its next draw");
    seen.clear();
    const auto corrected=ds::Model::verify_picks(input,picks,1,[&](int row,int32_t token) {
        seen.push_back(row);return row==1?99:token;
    });
    check(corrected.tokens==std::vector<int32_t>{10,20} && corrected.next==99 && seen==std::vector<int>{0,1},
          "the grammar consumed a discarded speculative row");
    seen.clear();
    const std::vector<int32_t> stop={1};
    const auto ended=ds::Model::verify_picks(stop,picks,1,filter);
    check(ended.tokens==stop && seen.empty(),"sampling continued after an accepted EOS");
    std::puts("grammar windows: single token, full draft, rejected draft and EOS pass");
}
static void elastic_snapshot(const char* path) {
    ds::RunOptions options;
    options.context=524288;options.slots=2;options.elastic_kv=true;
    options.helper_device=-2;options.draft_device=0;options.dspark=true;options.freeze_placement=true;
    options.reserve_mib=600;options.cache_limit_bytes=12ull<<30;options.resident_experts=false;
    basalt::core::diagnostics().exp["ds-graph"]=1;
    ds::Model model(path,options);
    const auto memory=[&](std::string_view name) {
        uint64_t bytes=0;
        for(const auto& m:model.memory())if(m.device==0 && m.component==name)bytes+=m.bytes;
        return bytes;
    };
    const auto base=memory("kv_state"), full_experts=memory("experts");
    const std::vector<int32_t> prefix={0,128803,8420,304,223,397,16,128804,128822,1000,1042,57,324,21,32,61,28};
    const std::vector<int32_t> suffix={219,4321,42,128799,87};
    model.prefill(prefix);
    auto snapshot=model.save_prefix();
    std::string error;
    check(memory("kv_state")==base,"short prompt grew the startup KV base");
    check(model.prepare_kv(65537,error),error.c_str());
    const auto grown=memory("kv_state"), experts_before_release=memory("experts");
    check(grown>base,"growing the slot did not back more pages");
    check(model.trim_kv(false)==grown-base,"trim did not report the released pages");
    check(memory("kv_state")==base,"trim did not keep exactly the base KV");
    check(memory("experts")==experts_before_release,"experts grew before the other slot could use the pages");
    model.select_slot(1);
    check(model.prepare_kv(65537,error),error.c_str());
    check(memory("kv_state")==grown,"the second slot did not map the same amount of KV");
    check(memory("experts")==experts_before_release,"the second slot evicted experts despite the released pages");
    check(model.release_idle_kv()==grown-base,"resetting the second slot did not retain its base");
    model.replenish_expert_cache();
    model.select_slot(0);
    model.trim_kv();
    const auto run=[&]{
        auto logits=model.prefill(suffix).logits;
        auto next=model.sample({})[0];
        const auto draft=model.draft(next);
        std::vector<int32_t> ids;
        while(ids.size()<4) {
            const auto step=model.decode_greedy(next,4-int(ids.size()));
            ids.insert(ids.end(),step.tokens.begin(),step.tokens.end());next=step.next;
        }
        return std::tuple{std::move(logits),std::move(ids),draft.tokens};
    };
    const auto reference=run();
    model.select_slot(1);
    model.prefill({0,21,222,77});
    check(memory("kv_state")==base,"slot one did not retain its startup KV");
    model.select_slot(0);
    model.reset();
    model.select_slot(1);
    check(model.position()==4,"releasing slot zero changed slot one");
    model.restore_prefix(snapshot);
    check(run()==reference,"trim/restore changed the continuation or logits");
    model.reset();
    check(memory("kv_state")==base,"reset did not retain exactly the startup KV");
    check(memory("experts")==full_experts,"expert slots did not return after both arenas trimmed to the base");
    std::printf("elastic KV: startup %zu, expanded %zu bytes; slot handoff, snapshot and four token IDs pass\n",
                size_t(base),size_t(grown));
}
int main(int argc,char** argv) {
    if(argc==3 && std::string(argv[2])=="--elastic-kv") {
        try {elastic_snapshot(argv[1]);return 0;}
        catch(const std::exception& e) {std::puts(e.what());return 1;}
    }
    if(argc==2 && std::string(argv[1])=="--grammar-verify") {
        try {grammar_window_test();return 0;}
        catch(const std::exception& e) {std::puts(e.what());return 1;}
    }
    if(argc!=2 && (argc!=3 || std::string(argv[2])!="--prefix-snapshot"))return 2;
    try {
        ds::RunOptions options;options.context=256;options.slots=2;options.cache_limit_bytes=1ull<<30;
        options.freeze_placement=true;options.resident_experts=false;options.helper_device=1;options.draft_device=1;
        basalt::core::diagnostics().exp["ds-graph"]=1;
        ds::Model model(argv[1],options);
        if(argc==3){snapshot_test(model);return 0;}
        const std::vector<std::vector<int32_t>> prompts={{0,128803,8420,304,223,397,16,128804,128822},
            {0,1234,219,4321,42,128799,87,902,404,19,657,223,42}};
        const auto start=[&](int slot,const basalt::kernels::SamplerParams& p) {
            model.select_slot(slot);model.reset();model.prefill(prompts[size_t(slot)]);return model.sample(p)[0];
        };
        for(bool greedy:{true,false}) {
            basalt::kernels::SamplerParams sampler[2];
            for(int slot=0;slot<2;++slot) {
                sampler[slot].greedy=greedy;sampler[slot].temperature=greedy?0.f:.6f;
                sampler[slot].top_k=20;sampler[slot].top_p=.95f;sampler[slot].seed=173+slot;
            }
            const int length=greedy?30:12;
            std::vector<int32_t> reference[2];
            for(int slot=0;slot<2;++slot) {
                auto p=sampler[slot];auto next=start(slot,p);p.counter=1;
                for(int i=0;i<length+2;++i) {
                    const auto r=model.decode(next,p,1);reference[slot].push_back(r.tokens[0]);next=r.next;++p.counter;
                }
            }
            int32_t next[2];std::vector<int32_t> outputs[2];
            for(int slot=0;slot<2;++slot) {next[slot]=start(slot,sampler[slot]);sampler[slot].counter=1;}
            const auto before=model.stats().batch_rounds;
            while(outputs[0].size()<size_t(length) || outputs[1].size()<size_t(length)) {
                std::vector<ds::DecodeRequest> requests;
                // Reverse request order as well as varying each verify width.
                for(int slot:{1,0})if(outputs[slot].size()<size_t(length))
                    requests.push_back({slot,next[slot],sampler[slot],std::min(slot?5:6,length-int(outputs[slot].size())),-1});
                const int original=model.current_slot();const auto result=model.decode_batch(requests);
                check(model.current_slot()==original,"batch changed selected slot");
                for(size_t i=0;i<requests.size();++i) {
                    const int slot=requests[i].slot;const auto& r=result[i];
                    outputs[slot].insert(outputs[slot].end(),r.tokens.begin(),r.tokens.end());
                    next[slot]=r.next;sampler[slot].counter+=r.tokens.size();
                }
            }
            for(int slot=0;slot<2;++slot) {
                check(std::equal(outputs[slot].begin(),outputs[slot].end(),reference[slot].begin()),"batched target sequence differs");
                model.select_slot(slot);check(model.position()==int(prompts[slot].size())+length,"slot cursor crossed sequences");
            }
            check(model.stats().batch_rounds>before,"no batched target forwards ran");
            const int position_b=model.position();
            auto replacement=sampler[0];replacement.counter=0;next[0]=start(0,replacement);replacement.counter=1;
            model.select_slot(1);check(model.position()==position_b,"resetting slot zero changed slot one");
            const auto continued=model.decode_batch({{0,next[0],replacement,2,-1},{1,next[1],sampler[1],2,-1}});
            check(std::equal(continued[1].tokens.begin(),continued[1].tokens.end(),reference[1].begin()+length),"slot reuse changed the other continuation");
            std::printf("slots: %s, reversed order, mixed widths, reset/reuse and continuation pass\n",greedy?"greedy":"sampled");std::fflush(stdout);
        }
        const auto position=model.position();
        try {model.decode_batch({{0,42,{},1,-1},{0,42,{},1,-1}});throw std::runtime_error("duplicate accepted");}
        catch(const std::invalid_argument&) {}
        catch(const std::runtime_error& e) {check(std::string(e.what())!="duplicate accepted","duplicate accepted");}
        check(model.position()==position,"invalid batch modified cursor");
        uint64_t weights=0,state=0;for(const auto& m:model.memory()) {
            if(m.component=="weights" || m.component=="dspark")weights+=m.bytes;
            if(m.component=="kv_state")state+=m.bytes;
        }
        std::printf("slots: shared weights %.3f MiB, both-slot state %.3f MiB, batch rounds %llu\n",weights/1048576.,state/1048576.,
                    (unsigned long long)model.stats().batch_rounds);
        std::printf("deepseek41_slots_test: passed\n");return 0;
    }catch(const std::exception& e) {std::printf("deepseek41_slots_test: %s\n",e.what());return 1;}
}
