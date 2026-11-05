#include "basalt/spec/suffix_drafter.hpp"
#include "basalt/deepseek41/model.hpp"
#include "basalt/core/diag.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <array>

namespace ds=basalt::deepseek41;
static int32_t pick(const std::vector<float>& rows,int row=0) {
    const auto begin=rows.begin()+size_t(row)*129280;
    return int32_t(std::max_element(begin,begin+129280)-begin);
}
static std::vector<int32_t> read_prompt(const char* path,size_t minimum=251) {
    std::ifstream file(path);std::string text((std::istreambuf_iterator<char>(file)),{});
    if(!file || text.empty())throw std::runtime_error("cannot read prefill IDS");
    std::replace(text.begin(),text.end(),',',' ');std::istringstream values(text);
    std::vector<int32_t> prompt;int32_t id;while(values>>id)prompt.push_back(id);
    if(prompt.size()<minimum || prompt.size()>size_t(ds::kPromptWindow))throw std::runtime_error("invalid prefill fixture length");
    return prompt;
}
static void quality(const ds::ForwardResult& a,const ds::ForwardResult& b,const std::vector<int32_t>& tokens) {
    if(a.rows!=b.rows || a.rows!=int(tokens.size()))throw std::runtime_error("quality row count mismatch");
    struct Metrics {double kl=0,max_kl=0,nll_a=0,nll_b=0,error=0;int matched=0;};
    std::array<Metrics,8> metrics{};std::vector<std::thread> threads;
    for(int worker=0;worker<8;++worker)threads.emplace_back([&,worker] {
        auto& m=metrics[worker];
        for(int t=worker;t<a.rows;t+=8) {
            const auto* x=a.logits.data()+size_t(t)*129280;const auto* y=b.logits.data()+size_t(t)*129280;
            const int pa=int(std::max_element(x,x+129280)-x),pb=int(std::max_element(y,y+129280)-y);
            const double ma=x[pa],mb=y[pb];double sa=0,sb=0,kl=0;
            for(int j=0;j<129280;++j) {sa+=std::exp(x[j]-ma);sb+=std::exp(y[j]-mb);m.error=std::max(m.error,std::abs(double(x[j])-y[j]));}
            const double la=std::log(sa),lb=std::log(sb);
            for(int j=0;j<129280;++j)kl+=std::exp(x[j]-ma)/sa*((x[j]-ma-la)-(y[j]-mb-lb));
            m.kl+=kl;m.max_kl=std::max(m.max_kl,kl);m.matched+=pa==pb;
            if(t+1<a.rows) {m.nll_a+=ma+la-x[tokens[t+1]];m.nll_b+=mb+lb-y[tokens[t+1]];}
        }
    });
    for(auto& thread:threads)thread.join();Metrics total;
    for(const auto& m:metrics) {total.kl+=m.kl;total.max_kl=std::max(total.max_kl,m.max_kl);total.nll_a+=m.nll_a;total.nll_b+=m.nll_b;total.error=std::max(total.error,m.error);total.matched+=m.matched;}
    std::printf("teacher-forced quantized legacy PPL=%.8g tensor PPL=%.8g mean_KLD=%.8g max_KLD=%.8g top1=%d/%d max_abs=%.8g\n",
        std::exp(total.nll_a/(a.rows-1)),std::exp(total.nll_b/(a.rows-1)),total.kl/a.rows,total.max_kl,total.matched,a.rows,total.error);
}
int main(int argc,char** argv) {
    if(argc<2) {std::printf("usage: deepseek41_model_test MODEL [load|window|boundary|draft|prefill IDS]\n");return 2;}
    try {
        const std::string mode=argc>2?argv[2]:"load";
        if(mode=="boundary" && argc==4 && std::string(argv[3])=="batch-rollback")
            basalt::core::diagnostics().exp["ds-rollback-window"]=1;
        const bool boundary_overlay=mode=="boundary" && argc==4 && std::string(argv[3])=="ring-overlay";
        if(boundary_overlay) {
            basalt::core::diagnostics().exp["ds-rollback-window"]=1;
            basalt::core::diagnostics().exp["ds-verify-ring-overlay"]=1;
        }
        if(mode=="draft" && argc==4 && std::string(argv[3])=="adaptive")
            basalt::core::diagnostics().exp["ds-adaptive-window"]=1;
        if(mode=="draft")basalt::core::diagnostics().exp["ds-graph"]=1;
        if(mode=="draft" && argc==4 && std::string(argv[3])=="gpu-timing") {
            basalt::core::diagnostics().exp["ds-cpu-timing"]=1;
            basalt::core::diagnostics().exp["ds-gpu-timing"]=1;
        }
        const bool all_select=mode=="direct" && argc==4 && std::string(argv[3])=="all-select";
        const bool key_attention=mode=="direct" && argc==4 && std::string(argv[3])=="keys";
        if(key_attention)basalt::core::diagnostics().exp["ds-attention-key-tiles"]=1;
        if(mode=="direct" && argc==4 && std::string(argv[3])=="front-metrics")
            basalt::core::diagnostics().exp["ds-front-metrics"]=1;
        if(mode=="direct" && argc==4 && std::string(argv[3])=="bounded-metrics") {
            basalt::core::diagnostics().exp["ds-front-metrics"]=1;
            basalt::core::diagnostics().exp["ds-metrics-from"]=4;
            basalt::core::diagnostics().exp["ds-metrics-count"]=4;
        }
        if(mode=="direct" && argc==4 && std::string(argv[3])=="helper-graph")
            basalt::core::diagnostics().exp["ds-helper-graph"]=1;
        if(mode!="load" && mode!="window" && mode!="boundary" && mode!="draft" && mode!="prefill" && mode!="stream" && mode!="tensor" && mode!="quality" && mode!="attention" && mode!="graph" && mode!="metrics" && mode!="direct" && mode!="matrix" && mode!="split" && mode!="long-quality" && mode!="long-draft" && mode!="verify-bench")return 2;
        if(mode=="graph" || mode=="metrics" || mode=="direct" || mode=="matrix" || mode=="split")basalt::core::diagnostics().exp["ds-graph"]=1;
        if(mode=="metrics" || mode=="direct" || mode=="matrix" || mode=="split") {
            basalt::core::diagnostics().exp["ds-metrics"]=1;
            basalt::core::diagnostics().exp["ds-pinned-transfers"]=1;
            basalt::core::diagnostics().exp["ds-device-route"]=1;
        }
        if(mode=="direct" || mode=="matrix" || mode=="split")basalt::core::diagnostics().exp["ds-direct-host"]=1;
        if(mode=="matrix")basalt::core::diagnostics().exp["ds-decode-matrix"]=1;
        const bool attention_check=mode=="matrix" && argc==4 && std::string(argv[3])=="check";
        if(attention_check) {
            basalt::core::diagnostics().exp["ds-graph"]=0;
            basalt::core::diagnostics().exp["ds-attention-check"]=1;
        }
        if(mode=="split")basalt::core::diagnostics().exp["ds-decode-split"]=1;
        if(mode=="split" && argc==4 && std::string(argv[3])=="keys")basalt::core::diagnostics().exp["ds-attention-key-tiles"]=1;
        ds::RunOptions options;options.context=256;options.resident_experts=false;options.helper_reserve_mib=1024;
        options.cache_limit_bytes=1ull<<30;options.freeze_placement=true;options.dspark=mode=="draft" || mode=="prefill" || mode=="stream" || mode=="tensor" || mode=="quality";
        options.helper_device=options.dspark || mode=="direct" || mode=="matrix" || mode=="split"?1:-2;
        if(mode=="prefill" || mode=="stream" || mode=="tensor" || mode=="quality") {options.context=262144;options.cache_limit_bytes=0;options.resident_experts=true;}
        options.stream_prefill_experts=mode=="stream";
        options.tensor_prefill=mode=="tensor" || mode=="quality";
        if(mode=="tensor" && argc>=5)options.tensor_prefill_tokens=std::stoi(argv[4]);
        const bool no_draft=mode=="tensor" && argc==6 && std::string(argv[5])=="no-draft";
        const bool profile_linear=mode=="attention" && argc==6 && (std::string(argv[5])=="profile" || std::string(argv[5])=="profile-fast");
        const bool peer_audit=mode=="attention" && argc==6 && std::string(argv[5])=="peer-audit";
        const bool expert_breakdown=mode=="attention" && argc==6 && std::string(argv[5])=="expert-breakdown";
        const bool prefill_metrics=expert_breakdown || peer_audit || (mode=="attention" && argc==6 && std::string(argv[5])=="prefill-metrics");
        const bool strided_f16=mode=="attention" && argc==6 && std::string(argv[5])=="strided-f16";
        const bool route_buckets=mode=="attention" && argc==6 && std::string(argv[5])=="route-buckets";
        const bool packed_prob=mode=="attention" && argc==6 && std::string(argv[5])=="packed-prob";
        const bool peer_stream=mode=="attention" && argc==6 && std::string(argv[5])=="peer-stream";
        const bool radix_topk=mode=="attention" && argc==6 && std::string(argv[5])=="radix-topk";
        const bool tuned_overlap=mode=="attention" && argc==6 && std::string(argv[5])=="tuned-overlap";
        const bool hc_token_tile=mode=="attention" && argc==6 && std::string(argv[5])=="hc-token-tile";
        const bool fused_tiles=mode=="attention" && argc==6 && std::string(argv[5])=="token-tiles-fused";
        const bool token_tiles=fused_tiles || (mode=="attention" && argc==6 && std::string(argv[5])=="token-tiles");
        const bool combined_prefill=tuned_overlap || hc_token_tile || token_tiles;
        const bool tuned_prefill=strided_f16 || route_buckets || packed_prob || peer_stream || radix_topk || combined_prefill;
        const bool batch_dma=mode=="attention" && argc==6 && std::string(argv[5])=="batch-dma";
        const bool host_dma=mode=="attention" && argc==6 && std::string(argv[5])=="host-dma";
        const bool window_bf16=mode=="attention" && argc==6 && std::string(argv[5])=="window-bf16";
        const bool compressed_bf16=mode=="attention" && argc==6 && std::string(argv[5])=="compressed-bf16";
        const bool router_tile8=mode=="attention" && argc==6 && std::string(argv[5])=="router-tile8";
        const bool router_linear=mode=="attention" && argc==6 && std::string(argv[5])=="router-linear";
        const bool hc_grouped=mode=="attention" && argc==6 && std::string(argv[5])=="hc-grouped";
        const bool cache_keys=mode=="attention" && argc==6 && std::string(argv[5])=="cache-keys";
        if(cache_keys)basalt::core::diagnostics().exp["ds-index-cache-keys"]=1;
        const bool rotary_table=mode=="attention" && argc==6 && std::string(argv[5])=="rotary-table";
        if(rotary_table)basalt::core::diagnostics().exp["ds-prefill-rotary-table"]=1;
        const bool local_rows=mode=="attention" && argc==6 && std::string(argv[5])=="local-rows";
        const bool topk_merge=mode=="attention" && argc==6 && std::string(argv[5])=="topk-merge";
        const bool wide_ring=mode=="attention" && argc==6 && std::string(argv[5])=="wide-ring";
        if(wide_ring)basalt::core::diagnostics().exp["ds-prefill-ring-depth"]=8;
        if(prefill_metrics)basalt::core::diagnostics().exp["ds-prefill-metrics"]=1;
        if(expert_breakdown) {
            auto& exp=basalt::core::diagnostics().exp;
            exp["ds-prefill-expert-breakdown"]=1;
            exp["ds-prefill-hc-grouped"]=1;
            exp["ds-prefill-tile-router-linear"]=1;
            exp["ds-prefill-router-tile8"]=1;
            exp["ds-prefill-compressed-bf16"]=1;
        }
        if(router_linear || router_tile8)basalt::core::diagnostics().exp["ds-prefill-hc-grouped"]=1;
        if(router_tile8 || compressed_bf16 || window_bf16)basalt::core::diagnostics().exp["ds-prefill-tile-router-linear"]=1;
        if(window_bf16) {
            auto& exp=basalt::core::diagnostics().exp;
            exp["ds-prefill-window-bf16"]=1;
            exp["ds-prefill-router-tile8"]=1;
            exp["ds-prefill-host-dma"]=1;
        }
        if(host_dma || batch_dma || tuned_prefill) {
            auto& exp=basalt::core::diagnostics().exp;
            exp["ds-prefill-hc-grouped"]=1;
            exp["ds-prefill-tile-router-linear"]=1;
            exp["ds-prefill-router-tile8"]=1;
            exp["ds-prefill-compressed-bf16"]=1;
            exp["ds-prefill-host-dma"]=1;
            if(batch_dma || tuned_prefill)exp["ds-prefill-window-bf16"]=1;
            if(batch_dma || tuned_prefill)exp["ds-prefill-batch-dma"]=1;
            if(tuned_prefill && !strided_f16)exp["ds-prefill-strided-f16"]=1;
            if(peer_stream || combined_prefill)exp["ds-prefill-peer-stream"]=1;
            if(combined_prefill) {
                exp["ds-prefill-ring-depth"]=8;
                exp["ds-index-cache-keys"]=1;
            }
            if(hc_token_tile || token_tiles)exp["ds-attention-packed-prob"]=1;
        }
        if(compressed_bf16 || window_bf16) {
            basalt::core::diagnostics().exp["ds-prefill-hc-grouped"]=1;
            basalt::core::diagnostics().exp["ds-prefill-compressed-bf16"]=1;
        }
        if(peer_audit)basalt::core::diagnostics().exp["ds-prefill-peer-audit"]=1;
        const bool once=prefill_metrics || (mode=="attention" && argc==6 && (std::string(argv[5])=="once" || std::string(argv[5])=="profile"));
        const bool fast_only=mode=="attention" && argc==6 && (std::string(argv[5])=="fast-only" || std::string(argv[5])=="profile-fast" || once);
        if(profile_linear)basalt::core::diagnostics().exp["ds-linear-timing"]=1;
        if(no_draft)options.dspark=false;
        if(mode=="attention") {
            options.context=262144;options.cache_limit_bytes=0;options.resident_experts=true;
            options.dspark=false;options.helper_device=1;options.tensor_prefill=true;
            options.tensor_prefill_tokens=argc>=5?std::stoi(argv[4]):512;
            basalt::core::diagnostics().exp["ds-prefill-tensor"]=1;
            if(options.tensor_prefill_tokens>2048)basalt::core::diagnostics().exp["ds-attention-mma"]=1;
        }
        if(mode=="long-quality") {
            options.context=262144;options.cache_limit_bytes=0;options.resident_experts=true;options.helper_device=1;
            options.dspark=false;options.tensor_prefill=true;options.tensor_prefill_tokens=32768;
            basalt::core::diagnostics().exp["ds-prefill-tensor"]=1;basalt::core::diagnostics().exp["ds-attention-mma"]=1;
        }
        if(mode=="long-draft") {
            options.context=8192;options.slots=2;options.cache_limit_bytes=0;
            options.resident_experts=true;options.helper_device=1;options.dspark=true;
            options.tensor_prefill=true;options.tensor_prefill_tokens=4096;
            for(const char* flag:{"ds-graph","ds-prefill-tensor","ds-pinned-transfers","ds-device-route","ds-direct-host"})
                basalt::core::diagnostics().exp[flag]=1;
        }
        if(mode=="verify-bench") {
            options.context=262144;options.cache_limit_bytes=0;options.resident_experts=true;options.helper_device=1;
            options.dspark=!(argc>=5 && (std::string(argv[4])=="select" || std::string(argv[4])=="split" || std::string(argv[4])=="suffix"));options.freeze_placement=false;options.tensor_prefill=true;options.tensor_prefill_tokens=512;
            if(argc>=5 && std::string(argv[4])=="split")basalt::core::diagnostics().exp["ds-attention-key-tiles"]=1;
            if(argc>=5 && std::string(argv[4])=="adaptive")basalt::core::diagnostics().exp["ds-bench-helper-experts"]=748;
            if(argc>=5 && std::string(argv[4])=="rollback") {
                auto& exp=basalt::core::diagnostics().exp;
                exp["ds-bench-helper-experts"]=740;
                for(const char* flag:{"ds-draft-markov-cache","ds-decode-sparse-host-parts","ds-cpu-group-pipeline",
                                      "ds-draft-overlap-adapt","ds-verify-front-width-cache"})exp[flag]=1;
            }
            for(const char* flag:{"ds-graph","ds-prefill-tensor","ds-pinned-transfers","ds-device-route","ds-direct-host"})
                basalt::core::diagnostics().exp[flag]=1;
        }
        const auto start=std::chrono::steady_clock::now();
        ds::Model model(argv[1],options);
        std::printf("Model loaded %.3f s\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());std::fflush(stdout);
        for(const auto& m:model.memory())std::printf("device %d %s %.3f MiB\n",m.device,m.component.c_str(),m.bytes/1048576.);
        if(mode=="load")return 0;
        if(mode=="long-draft") {
            if(argc!=4)throw std::runtime_error("long-draft needs a 2112-token IDS fixture");
            const auto prompt=read_prompt(argv[3]);
            if(prompt.size()!=2112)throw std::runtime_error("long-draft fixture must cross a physical tile in its final 128 rows");
            const auto run=[&](int slot,int window,int& proposed) {
                model.select_slot(slot);model.reset();
                const auto logits=model.prefill(prompt).logits;
                int32_t next=int32_t(std::max_element(logits.begin(),logits.end())-logits.begin());
                std::vector<int32_t> tokens;
                while(tokens.size()<12) {
                    const auto step=model.decode_greedy(next,std::min(window,12-int(tokens.size())));
                    next=step.next;proposed+=step.proposed;
                    tokens.insert(tokens.end(),step.tokens.begin(),step.tokens.end());
                }
                return tokens;
            };
            int no_draft=0,proposed=0;
            const auto single=run(0,1,no_draft);
            const auto speculative=run(1,6,proposed);
            if(no_draft || !proposed || single!=speculative)
                throw std::runtime_error("DSpark after long prefill changed the target continuation");
            std::printf("Long-prefill draft parity: 12 target IDs identical across slots, %d proposals\n",proposed);
            return 0;
        }
        if(mode=="verify-bench") {
            if(argc<4 || argc>6)throw std::runtime_error("verify-bench needs IDS [candidate] [tokens]");
            const std::string selector=argc>=5?argv[4]:"route";
            if(selector!="route" && selector!="front" && selector!="groups" && selector!="select" && selector!="split" && selector!="draft-body" && selector!="draft-picks" && selector!="draft-all" && selector!="adaptive" && selector!="suffix" && selector!="rollback")throw std::runtime_error("invalid verification benchmark selector");
            const char* gate=selector=="rollback"?"ds-rollback-window":selector=="suffix"?"ds-test-suffix":selector=="adaptive"?"ds-adaptive-window":selector=="front"?"ds-verify-front":selector=="groups"?"ds-grouped-window":selector=="select"?"ds-all-select":selector=="split"?"ds-decode-split":selector=="draft-picks"?"ds-draft-device-picks":selector=="draft-body" || selector=="draft-all"?"ds-draft-device-body":"ds-verify-route";
            const auto set_gate=[&](int enabled) {
                basalt::core::diagnostics().exp[gate]=enabled;
                if(selector=="draft-all")basalt::core::diagnostics().exp["ds-draft-device-picks"]=enabled;
                if(selector=="draft-body")basalt::core::diagnostics().exp["ds-draft-device-picks"]=0;
                if(selector=="draft-picks")basalt::core::diagnostics().exp["ds-draft-device-body"]=1;
            };
            std::printf("Verify benchmark %s: three warm-ups per side, six-token primer outside timer\n",selector.c_str());
            const int measured_tokens=argc==6?std::stoi(argv[5]):128;
            if(measured_tokens<16 || measured_tokens>512)throw std::runtime_error("verification benchmark token bound is 16..512");
            const auto prompt=read_prompt(argv[3],1);
            struct Run {std::vector<int32_t> tokens;double ms=0;int accepted=0,proposed=0,passes=0;};
            const auto request=[&](int count) {
                model.reset();const auto row=model.prefill(prompt);
                basalt::kernels::SamplerParams sampler;sampler.greedy=selector!="adaptive" && selector!="rollback";sampler.temperature=sampler.greedy?0.f:.6f;
                sampler.top_k=20;sampler.top_p=.95f;sampler.seed=173;
                int32_t next=sampler.greedy?pick(row.logits):model.sample(sampler)[0];sampler.counter=1;
                basalt::spec::SuffixDrafter suffix(3,32,prompt.size()+count); const bool predict=selector=="suffix" && basalt::core::exp_flag(gate)!=0; if(predict)suffix.append(prompt.data(),prompt.size()); const auto began=std::chrono::steady_clock::now();Run result;
                while(int(result.tokens.size())<count) {
                    int32_t draft[5]; int n=0; if(predict){suffix.append(next);n=suffix.propose(5,draft);} const auto decoded=model.decode(next,sampler,count-int(result.tokens.size()),-1,{draft,size_t(n)}); if(predict && decoded.tokens.size()>1)suffix.append(decoded.tokens.data()+1,decoded.tokens.size()-1);
                    result.tokens.insert(result.tokens.end(),decoded.tokens.begin(),decoded.tokens.end());next=decoded.next;
                    sampler.counter+=decoded.tokens.size();
                    result.accepted+=decoded.accepted;result.proposed+=decoded.proposed;++result.passes;
                }
                result.ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count();return result;
            };
            set_gate(0);
            const auto reference=request(measured_tokens).tokens;request(measured_tokens);request(measured_tokens);
            set_gate(1);
            const bool fixed_reference=selector!="adaptive" && selector!="rollback";
            for(int warm=0;warm<3;++warm)
                if(request(measured_tokens).tokens!=reference && fixed_reference)throw std::runtime_error("verification warm-up changed output");
            double old_ms=0,new_ms=0;int cycle=0;
            std::vector<int32_t> paired_reference;
            bool pair_equal=true;
            for(int enabled:{0,1,1,0,0,1,1,0}) {
                set_gate(enabled);request(6);
                const auto result=request(measured_tokens);
                if(result.tokens!=reference && fixed_reference)throw std::runtime_error("verification benchmark changed output");
                if(paired_reference.empty())paired_reference=result.tokens;
                pair_equal=pair_equal && result.tokens==paired_reference;
                (enabled?new_ms:old_ms)+=result.ms;
                std::printf("Verify ABBA %d candidate=%d decode_ms=%.3f tps=%.3f accept=%d/%d passes=%d ms_pass=%.3f output=%s\n",
                    ++cycle,enabled,result.ms,measured_tokens*1000./result.ms,result.accepted,result.proposed,result.passes,result.ms/result.passes,
                    result.tokens==(selector=="rollback"?paired_reference:reference)?"identical":"sampled-different");std::fflush(stdout);
            }
            if(selector=="rollback" && !pair_equal)std::printf("Verification throughput is not fixed-output: warm sampled sequences changed.\n");
            std::printf("Verify ABBA aggregate: original %.3f tok/s, candidate %.3f tok/s, time %.3f/%.3f ms\n",
                        4*measured_tokens*1000./old_ms,4*measured_tokens*1000./new_ms,old_ms,new_ms);return 0;
        }
        if(mode=="long-quality") {
            if(argc!=4)throw std::runtime_error("long quality needs a short IDS fixture");const auto prompt=read_prompt(argv[3]);
            if(prompt.size()>2048)throw std::runtime_error("long quality readback fixture exceeds 2048 rows");
            const auto reference=model.prefill(prompt,0,ds::LogitsMode::all);model.reset();
            basalt::core::diagnostics().exp["ds-force-long"]=1;
            const auto candidate=model.prefill(prompt,0,ds::LogitsMode::all);
            std::printf("Long-prompt scheduling versus short tensor path, frozen tiers:\n");quality(reference,candidate,prompt);return 0;
        }
        if(mode=="quality") {
            if(argc!=4)throw std::runtime_error("quality needs an IDS fixture");const auto prompt=read_prompt(argv[3]);
            basalt::core::diagnostics().exp["ds-prefill-tensor"]=0;
            const auto reference=model.prefill(prompt,0,ds::LogitsMode::all);model.reset();
            basalt::core::diagnostics().exp["ds-prefill-tensor"]=1;
            const auto candidate=model.prefill(prompt,0,ds::LogitsMode::all);
            quality(reference,candidate,prompt);return 0;
        }
        if(mode=="stream" || mode=="tensor" || mode=="attention") {
            if(argc!=4 && !((mode=="tensor" || mode=="attention") && argc==5) && !no_draft && !fast_only && !wide_ring && !topk_merge && !local_rows && !rotary_table && !cache_keys && !hc_grouped && !router_linear && !router_tile8 && !compressed_bf16 && !window_bf16 && !host_dma && !batch_dma && !tuned_prefill)throw std::runtime_error("stream needs an IDS fixture");
            const auto prompt=read_prompt(argv[3]);
            const std::vector<int32_t> warm=no_draft || mode=="attention"?prompt:std::vector<int32_t>(prompt.begin(),prompt.begin()+128);
            const char* gate=hc_token_tile?"ds-prefill-hc-token-tile":radix_topk?"ds-index-topk-radix":peer_stream?"ds-prefill-peer-stream":packed_prob?"ds-attention-packed-prob":route_buckets?"ds-prefill-route-buckets":strided_f16?"ds-prefill-strided-f16":batch_dma?"ds-prefill-batch-dma":host_dma?"ds-prefill-host-dma":window_bf16?"ds-prefill-window-bf16":compressed_bf16?"ds-prefill-compressed-bf16":router_tile8?"ds-prefill-router-tile8":router_linear?"ds-prefill-tile-router-linear":hc_grouped?"ds-prefill-hc-grouped":cache_keys?"ds-index-cache-keys":rotary_table?"ds-prefill-rotary-table":local_rows?"ds-attention-local-rows":topk_merge?"ds-index-topk-merge":wide_ring?"ds-prefill-ring-wide":mode=="attention"?"ds-attention-predecode":mode=="tensor"?"ds-prefill-tensor":"ds-prefill-stream";
            const auto set_gate=[&](int enabled) {
                auto& exp=basalt::core::diagnostics().exp;
                if(tuned_overlap) {
                    exp["ds-prefill-peer-stream"]=enabled;
                    exp["ds-prefill-ring-wide"]=enabled;
                    exp["ds-attention-packed-prob"]=enabled;
                    exp["ds-index-cache-keys"]=enabled;
                } else if(token_tiles) {
                    exp["ds-prefill-hc-token-tile"]=enabled?8:4;
                    exp["ds-prefill-router-expert-tile"]=enabled?4:1;
                    exp["ds-prefill-fused-swiglu-quant"]=enabled && fused_tiles;
                } else exp[gate]=enabled;
            };
            const std::vector<int> warm_passes=prefill_metrics || wide_ring?std::vector<int>{1}:once?std::vector<int>{}:no_draft || fast_only?std::vector<int>{1}:std::vector<int>{0,1};
            for(int enabled:warm_passes) {set_gate(enabled);model.reset();model.prefill(warm);}
            if(prefill_metrics)model.reset_metrics();
            std::vector<float> reference;std::vector<int32_t> reference_output;
            const std::vector<int> passes=(topk_merge || local_rows || rotary_table || cache_keys || hc_grouped || router_linear || router_tile8 || compressed_bf16 || window_bf16 || host_dma || batch_dma || tuned_prefill)?std::vector<int>{0,1}:once?std::vector<int>{1}:fast_only?std::vector<int>{1,1}:mode=="attention"?std::vector<int>{0,1,1,0}:no_draft?std::vector<int>{1,1}:prompt.size()>512?std::vector<int>{0,1}:std::vector<int>{0,1,1,0};
            for(int enabled:passes) {
                set_gate(enabled);model.reset();
                const auto before=model.stats();
                std::printf("stream start enabled=%d\n",enabled);std::fflush(stdout);
                const auto started=std::chrono::steady_clock::now();
                const auto result=model.prefill(prompt);
                const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
                std::printf("stream end enabled=%d\n",enabled);std::fflush(stdout);
                const auto after=model.stats();
                if(reference.empty())reference=result.logits;
                double error=0,sa=0,sb=0,kld=0;const double ma=*std::max_element(reference.begin(),reference.end()),mb=*std::max_element(result.logits.begin(),result.logits.end());
                for(size_t j=0;j<reference.size();++j) {error=std::max(error,std::abs(double(reference[j])-result.logits[j]));sa+=std::exp(reference[j]-ma);sb+=std::exp(result.logits[j]-mb);}
                if(mode=="attention" && error!=0) {
                    std::fprintf(stdout,"stream parity failed enabled=%d seconds=%.6f max_abs=%.9g\n",enabled,seconds,error);
                    throw std::runtime_error("32K candidate changed final logits");
                }
                for(size_t j=0;j<reference.size();++j) {const double a=reference[j]-ma-std::log(sa),b=result.logits[j]-mb-std::log(sb);kld+=std::exp(a)*(a-b);}
                int32_t next=pick(result.logits);std::vector<int32_t> output;
                while(output.size()<12) {const auto row=model.decode_greedy(next,int(12-output.size()));output.insert(output.end(),row.tokens.begin(),row.tokens.end());next=row.next;}
                if(reference_output.empty())reference_output=output;
                int agree=0;for(size_t j=0;j<output.size();++j)agree+=output[j]==reference_output[j];
                if(mode=="attention" && agree!=12)throw std::runtime_error("predecoded attention changed greedy output");
                std::printf("stream bench enabled=%d tokens=%zu seconds=%.6f tok_s=%.3f max_abs=%.7g kld=%.7g greedy=%d/12 entries=%llu bytes=%llu peer_bytes=%llu file_reads=%llu\n",
                    enabled,prompt.size(),seconds,prompt.size()/seconds,error,kld,agree,
                    (unsigned long long)(after.streamed_entries-before.streamed_entries),
                    (unsigned long long)(after.streamed_bytes-before.streamed_bytes),
                    (unsigned long long)(after.peer_streamed_bytes-before.peer_streamed_bytes),
                    (unsigned long long)(after.expert_file_reads-before.expert_file_reads));std::fflush(stdout);
            }
            return 0;
        }
        if(mode=="prefill") {
            if(argc!=4)throw std::runtime_error("prefill needs an IDS fixture");
            const auto prompt=read_prompt(argv[3]);
            constexpr int width=128;
            const std::vector<int32_t> prefix(prompt.begin(),prompt.begin()+123),window(prompt.begin()+123,prompt.begin()+123+width);
            model.prefill(prefix,11);const auto wide=model.forward(window,ds::LogitsMode::all);model.commit(7);
            const auto tail=model.forward(std::vector<int32_t>(window.begin()+7,window.end()),ds::LogitsMode::all);model.commit(width-7);
            for(int t=0;t<width-7;++t)for(int j=0;j<129280;++j)
                if(tail.logits[size_t(t)*129280+j]!=wide.logits[size_t(t+7)*129280+j])throw std::runtime_error("prefill partial rollback changed logits");
            model.reset();model.prefill(prefix,11);
            for(int first=0;first<width;first+=11) {
                const int end=std::min(width,first+11);
                const auto row=model.forward(std::vector<int32_t>(window.begin()+first,window.begin()+end),ds::LogitsMode::all);model.commit(end-first);
                for(size_t j=0;j<row.logits.size();++j)
                    if(row.logits[j]!=wide.logits[size_t(first)*129280+j])throw std::runtime_error("prefill/11-token logits differ");
            }
            std::printf("128-token window across ring wrap and partial commit: byte-identical logits\n");std::fflush(stdout);
            model.reset();model.prefill(std::vector<int32_t>(prompt.begin(),prompt.begin()+64),11);
            model.reset();model.prefill(std::vector<int32_t>(prompt.begin(),prompt.begin()+128));
            std::vector<float> reference;std::vector<int32_t> reference_draft,reference_output;
            const std::vector<int> chunks=prompt.size()>512?std::vector<int>{11,128}:std::vector<int>{11,128,128,11};
            for(int chunk:chunks) {
                model.reset();const auto begin=std::chrono::steady_clock::now();
                const auto result=model.prefill(prompt,chunk);
                const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
                ds::DraftResult draft;if(!no_draft)draft=model.draft(pick(result.logits));
                if(reference.empty()){reference=result.logits;reference_draft=draft.tokens;}
                if(result.logits!=reference || draft.tokens!=reference_draft)throw std::runtime_error("prefill chunk changed target logits or DSpark proposals");
                std::printf("prefill bench chunk=%d tokens=%zu seconds=%.6f tok_s=%.3f\n",chunk,prompt.size(),seconds,prompt.size()/seconds);std::fflush(stdout);
                int32_t next=pick(result.logits);std::vector<int32_t> output;
                while(output.size()<12) {
                    const auto row=model.decode_greedy(next,int(12-output.size()));
                    output.insert(output.end(),row.tokens.begin(),row.tokens.end());next=row.next;
                }
                if(reference_output.empty())reference_output=output;
                if(output!=reference_output)throw std::runtime_error("prefill chunk changed subsequent greedy output");
            }
            std::printf("prefill fixed-tier identity: target logits, DSpark proposals and 12 greedy tokens identical\n");return 0;
        }
        if(mode=="boundary") {
            const std::vector<int32_t> cycle={0,128803,8420,304,223,397,16,128804,128822};
            auto prefix=[&] {
                for(int first=0;first<123;first+=11) {
                    std::vector<int32_t> rows;
                    for(int p=first;p<std::min(123,first+11);++p)rows.push_back(cycle[size_t(p)%cycle.size()]);
                    model.forward(rows,ds::LogitsMode::none);model.commit(int(rows.size()));
                }
            };
            std::vector<int32_t> tail;for(int p=123;p<(boundary_overlay?129:134);++p)tail.push_back(cycle[size_t(p)%cycle.size()]);
            prefix();
            const auto batch=model.forward(tail,ds::LogitsMode::all);model.commit(3);
            const std::vector<int32_t> remainder(tail.begin()+3,tail.end());
            const auto replay=model.forward(remainder,ds::LogitsMode::all);model.commit(int(remainder.size()));
            for(int t=0;t<int(remainder.size());++t)if(pick(replay.logits,t)!=pick(batch.logits,t+3))throw std::runtime_error("ring boundary rollback changed greedy output");
            model.reset();prefix();
            for(int t=0;t<int(tail.size());++t) {
                const auto row=model.forward({tail[t]});model.commit(1);
                if(pick(row.logits)!=pick(batch.logits,t))throw std::runtime_error("ring boundary window changed greedy output");
                double error=0;for(int j=0;j<129280;++j)error=std::max(error,std::abs(double(row.logits[j])-batch.logits[size_t(t)*129280+j]));
                std::printf("boundary position %d: max_abs %.7g\n",123+t,error);std::fflush(stdout);
                if(error>0.02)throw std::runtime_error("ring boundary window logits differ");
            }
            const auto reference=model.forward({42});model.commit(0);
            const auto retried=model.forward({42});model.commit(1);
            if(reference.logits!=retried.logits)throw std::runtime_error("zero-length commit changed state");
            std::printf("ring wrap, partial compression rollback, zero commit: 0 failures\n");return 0;
        }
        std::vector<int32_t> input=mode=="draft"?std::vector<int32_t>{0,128803,8420,304,223,397,16,128804,128822}:
            std::vector<int32_t>{0,1234,219,4321,42,128799,87,902,404,19,657};
        if(attention_check)input.resize(2);
        if(all_select)basalt::core::diagnostics().exp["ds-all-select"]=0;
        if(key_attention)basalt::core::diagnostics().exp["ds-decode-split"]=0;
        const auto a=model.forward(input,ds::LogitsMode::all);model.commit(int(input.size()));
        if(all_select)basalt::core::diagnostics().exp["ds-all-select"]=1;
        if(key_attention)basalt::core::diagnostics().exp["ds-decode-split"]=1;
        for(int t=0;t<int(input.size());++t)std::printf("target %d -> %d\n",t,pick(a.logits,t));std::fflush(stdout);
        if(mode=="draft") {
            for(int32_t anchor:{42,128799,1}) {
                basalt::core::diagnostics().exp["ds-draft-device-body"]=0;
                basalt::core::diagnostics().exp["ds-draft-device-picks"]=0;
                const auto legacy=model.draft(anchor);
                for(int variant:{1,2,3}) {
                    basalt::core::diagnostics().exp["ds-draft-device-body"]=(variant&1)!=0;
                    basalt::core::diagnostics().exp["ds-draft-device-picks"]=(variant&2)!=0;
                    const auto gpu=model.draft(anchor);
                    if(gpu.tokens!=legacy.tokens || gpu.confidence!=legacy.confidence)
                        throw std::runtime_error("device DSpark graph changed proposal or confidence bits");
                }
            }
            std::printf("DSpark device body, picks and combined: proposal/confidence bits identical across three anchors\n");
            const auto proposals=model.draft(pick(a.logits,int(input.size())-1));
            for(size_t j=0;j<proposals.tokens.size();++j)std::printf("draft %zu -> %d confidence %.6f\n",j,proposals.tokens[j],proposals.confidence[j]);
            std::vector<int32_t> verify={pick(a.logits,int(input.size())-1)};
            verify.insert(verify.end(),proposals.tokens.begin(),proposals.tokens.end());
            const auto checked=model.forward(verify,ds::LogitsMode::all);
            int accepted=1;while(accepted<int(verify.size()) && verify[accepted]==pick(checked.logits,accepted-1))++accepted;
            model.commit(accepted);
            std::printf("DSpark accepted %d of %zu proposals; target cursor %d\n",accepted-1,proposals.tokens.size(),model.position());
            auto next=pick(checked.logits,accepted-1);
            for(int round=0;round<3 && next!=1;++round) {
                const auto decoded=model.decode_greedy(next,6,1);next=decoded.next;
                std::printf("decode: accepted %d/%d, next %d; output",decoded.accepted,decoded.proposed,next);
                for(int token:decoded.tokens)std::printf(" %d",token);std::printf("\n");std::fflush(stdout);
            }
            model.reset();
            const auto prefix=model.forward(input);model.commit(int(input.size()));
            next=pick(prefix.logits);std::vector<int32_t> expected;
            for(int j=0;j<18;++j) {
                expected.push_back(next);
                const auto row=model.forward({next});model.commit(1);next=pick(row.logits);
            }
            model.reset();const auto prefix_again=model.forward(input);model.commit(int(input.size()));
            next=pick(prefix_again.logits);std::vector<int32_t> speculative;
            while(speculative.size()<expected.size()) {
                const auto row=model.decode_greedy(next,int(expected.size()-speculative.size()));
                speculative.insert(speculative.end(),row.tokens.begin(),row.tokens.end());next=row.next;
            }
            if(speculative!=expected)throw std::runtime_error("DSpark verification changed target greedy sequence");
            std::printf("greedy identity: %zu tokens identical with DSpark verification\n",expected.size());std::fflush(stdout);
            basalt::kernels::SamplerParams sampler;sampler.seed=173;sampler.temperature=.6f;sampler.top_k=20;sampler.top_p=.95f;
            model.reset();model.forward(input);model.commit(int(input.size()));
            next=model.sample(sampler)[0];sampler.counter=1;expected.clear();
            for(int j=0;j<12;++j) {
                const auto row=model.decode(next,sampler,1);expected.push_back(row.tokens[0]);next=row.next;sampler.counter+=1;
            }
            model.reset();model.forward(input);model.commit(int(input.size()));sampler.counter=0;
            next=model.sample(sampler)[0];sampler.counter=1;speculative.clear();
            while(speculative.size()<expected.size()) {
                const auto row=model.decode(next,sampler,int(expected.size()-speculative.size()));
                speculative.insert(speculative.end(),row.tokens.begin(),row.tokens.end());next=row.next;sampler.counter+=row.tokens.size();
            }
            if(speculative!=expected)throw std::runtime_error("DSpark verification changed counter-based sampled sequence");
            std::printf("sampled identity: %zu tokens identical with DSpark verification\n",expected.size());std::fflush(stdout);
            if(basalt::core::exp_flag("ds-adaptive-window")!=0)for(int width:{2,3,4,5}) {
                model.reset();model.forward(input);model.commit(int(input.size()));sampler.counter=0;
                next=model.sample(sampler)[0];sampler.counter=1;speculative.clear();
                while(speculative.size()<expected.size()) {
                    const auto row=model.decode(next,sampler,std::min(width,int(expected.size()-speculative.size())));
                    speculative.insert(speculative.end(),row.tokens.begin(),row.tokens.end());next=row.next;
                    sampler.counter+=row.tokens.size();
                }
                if(speculative!=expected)throw std::runtime_error("adaptive window changed absolute target draws");
                std::printf("adaptive sampled cap %d: target sequence identical\n",width);std::fflush(stdout);
            }
        } else {
            model.reset();
            ds::ForwardResult matrix_rows;matrix_rows.rows=int(input.size());
            for(int t=0;t<int(input.size());++t) {
                const auto row=model.forward({input[t]});model.commit(1);
                if(mode=="matrix")matrix_rows.logits.insert(matrix_rows.logits.end(),row.logits.begin(),row.logits.end());
                double error=0;
                for(size_t j=0;j<129280;++j)error=std::max(error,std::abs(double(row.logits[j])-a.logits[size_t(t)*129280+j]));
                std::printf("window row %d max_abs %.7g greedy %d/%d\n",t,error,pick(row.logits),pick(a.logits,t));std::fflush(stdout);
                if(pick(row.logits)!=pick(a.logits,t) || error>0.02)throw std::runtime_error("window/sequential logits mismatch");
            }
            if(mode=="matrix")quality(a,matrix_rows,input);
            if(mode=="direct")for(int width=2;width<=6;++width) {
                model.reset();double error=0;
                for(int first=0;first<int(input.size());first+=width) {
                    const int count=std::min(width,int(input.size())-first);
                    const auto row=model.forward(std::vector<int32_t>(input.begin()+first,input.begin()+first+count),ds::LogitsMode::all);
                    model.commit(count);
                    for(int t=0;t<count;++t)for(int j=0;j<129280;++j)
                        error=std::max(error,std::abs(double(row.logits[size_t(t)*129280+j])-a.logits[size_t(first+t)*129280+j]));
                }
                std::printf("device route width %d max_abs %.7g\n",width,error);std::fflush(stdout);
                if(error!=0)throw std::runtime_error("device route window changed frozen logits");
            }
            model.reset();model.forward(input);model.commit(3);
            const auto tail=model.forward(std::vector<int32_t>(input.begin()+3,input.end()),ds::LogitsMode::all);model.commit(int(input.size())-3);
            for(int t=0;t<int(input.size())-3;++t)if(pick(tail.logits,t)!=pick(a.logits,t+3))throw std::runtime_error("rollback changed greedy output");
        }
        const auto stats=model.stats();
        std::printf("expert entries: primary %llu, helper %llu, CPU %llu; draft rounds %llu\n",
                    (unsigned long long)stats.primary_entries,(unsigned long long)stats.helper_entries,
                    (unsigned long long)stats.cpu_entries,(unsigned long long)stats.draft_rounds);
        if(mode=="draft" && (!stats.helper_entries || !stats.draft_rounds))throw std::runtime_error("shared GPU roles were not exercised");
        std::printf("deepseek41_model_test: 0 failures\n");return 0;
    } catch(const std::exception& e){std::printf("deepseek41_model_test: %s\n",e.what());return 1;}
}
