#include "deepseek.hpp"
#include "deepseek_session.hpp"
#include "basalt/deepseek41/model.hpp"
#include "basalt/deepseek41/packed_weights.hpp"
#include "basalt/log.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/model_file.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cuda_profiler_api.h>

namespace basalt::program::generate {
int run_deepseek(const Options& o) {
    try {
        namespace ds=basalt::deepseek41;
        if(o.conversation_cache_disk_mib>0) {
            basalt::log::error("basalt deepseek41","disk conversation cache is not implemented");return 2;
        }
        // DeepSeek uses its own execution state and serving scheduler.
        if(!o.greedy || !o.dump_residual.empty() || !o.layer_split.empty() || o.batch_groups!=1 || !core::diagnostics().draft_log.empty()) {
            basalt::log::error("basalt deepseek41","unsupported mode: use token-id serving without layer splitting or draft logs");return 2;
        }
        if(o.resident_budget_given || !o.shared_expert_arena.empty() || o.ple_io!="direct" ||
           o.no_pool || !o.dump_routing.empty() || !o.dump_mixed.empty() || !o.dump_final_r.empty() ||
           !o.rope_scaling.empty() || o.rope_scale!=0 || o.rope_freq_base!=0 || o.rope_freq_scale!=0 ||
           o.yarn_orig_ctx!=0 || o.yarn_ext_factor!=-1 || o.yarn_attn_factor!=1 || o.yarn_beta_fast!=32 || o.yarn_beta_slow!=1 ||
           o.relaxed_drafting>0 || o.route_prior>0 || o.route_tail_skip>0 ||
           std::any_of(o.expert_cache_remote.begin(),o.expert_cache_remote.end(),[](int n){return n!=0;})) {
            basalt::log::error("basalt deepseek41","requested Qwen-only tuning, dump or memory mode is not implemented");return 2;
        }
        if((o.tokens.empty() && !o.serve) || o.max_context<16 || o.max_context>1048576 || o.max_new<1 ||
           o.max_new>o.max_context || o.tokens.size()>uint64_t(o.max_context-o.max_new)) {
            basalt::log::error("basalt deepseek41","prompt and generation must fit --max-context (16..1048576)");return 2;
        }
        for(auto token:o.tokens)if(token<0 || token>=129280) {
            basalt::log::error("basalt deepseek41","token outside vocabulary");return 2;
        }
        if(o.ple_row_cache<0 || o.ple_inflight<1 || o.ple_inflight>4096 || o.full_resident_experts || o.spec<0) {
            basalt::log::error("basalt deepseek41","invalid Engram reader options or unsupported full resident mirror");return 2;
        }
        ds::RunOptions options;options.context=int(o.max_context);options.workers=o.pool_workers;options.slots=std::max(1,o.batch);
        const int window=o.spec_given?std::clamp(o.spec,1,6):6;
        options.dspark=window>1;options.elastic_kv=o.vram_elastic;options.helper_device=o.peer_device>=1?o.peer_device:-2;
        if(options.dspark)options.draft_path=o.dspark_model;
        const bool tuned_prefill=core::exp_flag("ds-prefill-32k-tuned")!=0;
        if(tuned_prefill) {
            if(o.prefill_chunk!=ds::kPromptWindow || o.max_context<ds::kPromptWindow) {
                basalt::log::error("basalt deepseek41","the tuned 32K profile requires --prefill 32768 and enough context");
                return 2;
            }
            auto& exp=core::diagnostics().exp;
            for(const auto* flag:{"ds-prefill-tensor","ds-prefill-hc-grouped","ds-prefill-tile-router-linear",
                                 "ds-prefill-router-tile8","ds-prefill-compressed-bf16","ds-prefill-window-bf16",
                                 "ds-attention-packed-prob","ds-index-cache-keys","ds-prefill-host-dma",
                                 "ds-prefill-batch-dma","ds-prefill-strided-f16","ds-prefill-peer-stream",
                                 "ds-prefill-fused-swiglu-quant","ds-prefill-route-buckets","ds-index-topk-radix"})
                exp.try_emplace(flag,1);
            exp.try_emplace("ds-prefill-hc-token-tile",8);
            exp.try_emplace("ds-prefill-router-expert-tile",4);
            exp.try_emplace("ds-prefill-ring-depth",8);
            if(core::exp_flag("ds-prefill-tensor")==0) {
                basalt::log::error("basalt deepseek41","the tuned 32K profile requires tensor prefill");
                return 2;
            }
        }
        options.draft_device=o.peer_device>=1?o.peer_device:0;
        options.reserve_mib=o.vram_reserve_given?o.vram_reserve_mib:200;
        options.helper_reserve_mib=o.peer_reserve_mib;
        options.stream_prefill_experts=core::exp_flag("ds-prefill-stream")!=0;
        options.tensor_prefill=core::exp_flag("ds-prefill-tensor")!=0;
        if(options.tensor_prefill && o.prefill_chunk>0)options.tensor_prefill_tokens=int(std::clamp<int64_t>(o.prefill_chunk,16,ds::kPromptWindow));
        options.resident_experts=!o.mmap_experts || o.resident_cpu_experts;
        options.freeze_placement=core::exp_flag("ds-bench-freeze-placement")!=0;
        options.engram_cache_rows=uint64_t(o.ple_row_cache);options.engram_inflight=o.ple_inflight;
        if(o.expert_cache>0) {
            core::ModelFile file;std::string error;
            if(!file.open(o.model,error))throw std::runtime_error(error);
            options.cache_limit_bytes=file.range("experts").size()/15360*uint64_t(o.expert_cache);
        }
        std::optional<grammar::PackVocab> vocab;
        std::optional<Tokenizer> tokenizer;
        if(o.serve) {
            core::ModelFile file;std::string error;
            std::vector<uint8_t> tokens,types;
            if(!file.open(o.model,error) ||
               !core::read_range(file.range("tokenizer/vocab.json"),tokens,error) ||
               !core::read_range(file.range("tokenizer/token_type.json"),types,error))
                throw std::runtime_error("DeepSeek grammar vocabulary: "+error);
            const auto text=[](const auto& v){return std::string_view(reinterpret_cast<const char*>(v.data()),v.size());};
            if(o.grammar) {
                vocab.emplace();
                if(!grammar::vocab_from_pack(text(tokens),text(types),129280,{1},*vocab,error,true))
                    throw std::runtime_error("DeepSeek grammar vocabulary: "+error);
            }
            std::vector<uint8_t> meta,merges;
            core::Json cfg;
            if(core::read_range(file.range("tokenizer/tokenizer.json"),meta,error) &&
               core::read_range(file.range("tokenizer/merges.txt"),merges,error) &&
               core::parse_json(reinterpret_cast<const char*>(meta.data()),meta.size(),cfg,error)) {
                const auto* pre=cfg.get("pre");
                tokenizer.emplace();
                if(!pre || !pre->is_string() || !tokenizer->load(text(tokens),text(merges),text(types),error,pre->s))
                    tokenizer.reset();
            }
            if(!tokenizer)basalt::log::warn("basalt deepseek41","native text unavailable: %s",error.c_str());
        }
        ds::Model model(o.model,options);
        if(o.serve)return deepseek_session(model,o,window,tuned_prefill?options.tensor_prefill_tokens:0,vocab,tokenizer?&*tokenizer:nullptr);
        using Clock=std::chrono::steady_clock;
        const int warmups=int(core::exp_flag("ds-metrics-warmups"));
        if(warmups<0 || warmups>16)throw std::runtime_error("invalid profiling warm-up count");
        const int trace_tokens=int(core::exp_flag("ds-trace-tokens"));
        if(trace_tokens<0 || trace_tokens>128 || (trace_tokens && core::diagnostics().profile))
            throw std::runtime_error("invalid bounded decode trace options");
        for(int run=0;run<warmups;++run) {
            model.reset();const auto row=model.prefill(std::vector<int32_t>(o.tokens.begin(),o.tokens.end()));
            int32_t next=int32_t(std::max_element(row.logits.begin(),row.logits.end())-row.logits.begin());
            int generated=0;bool ended=false;
            while(generated<o.max_new && !ended) {
                const auto step=model.decode_greedy(next,int(std::min<int64_t>(window,o.max_new-generated)),o.stop_eos?1:-1);
                generated+=int(step.tokens.size());next=step.next;ended=o.stop_eos && step.tokens.back()==1;
            }
        }
        if(warmups)model.reset();
        const bool profile_prompt=core::diagnostics().profile && core::exp_flag("ds-profile-prefill")!=0;
        if(profile_prompt)cudaProfilerStart();
        const auto start=Clock::now();
        const auto logits=model.prefill(std::vector<int32_t>(o.tokens.begin(),o.tokens.end()));
        const auto prefetched=Clock::now();
        const bool metrics=core::exp_flag("ds-metrics")!=0 || core::exp_flag("ds-front-metrics")!=0;
        const auto metric_base=metrics?model.stats():ds::ExecutionStats{};
        if(metrics)model.reset_metrics();
        if(core::diagnostics().profile && !profile_prompt)cudaProfilerStart();
        int32_t next=int32_t(std::max_element(logits.logits.begin(),logits.logits.end())-logits.logits.begin());
        std::vector<int32_t> generated;
        int proposed=0,accepted=0;bool ended=false;
        bool tracing=trace_tokens>0;
        if(tracing && cudaProfilerStart()!=cudaSuccess)throw std::runtime_error("decode trace start failed");
        while(generated.size()<size_t(o.max_new) && !ended) {
            const auto row=model.decode_greedy(next,int(std::min<int64_t>(window,o.max_new-int64_t(generated.size()))),o.stop_eos?1:-1);
            generated.insert(generated.end(),row.tokens.begin(),row.tokens.end());
            proposed+=row.proposed;accepted+=row.accepted;next=row.next;
            ended=o.stop_eos && generated.back()==1;
            if(tracing && generated.size()>=size_t(trace_tokens)) {
                if(cudaProfilerStop()!=cudaSuccess)throw std::runtime_error("decode trace stop failed");
                tracing=false;
            }
        }
        if(tracing && cudaProfilerStop()!=cudaSuccess)throw std::runtime_error("decode trace stop failed");
        const auto finished=Clock::now();
        if(core::diagnostics().profile)cudaProfilerStop();
        std::printf("prompt  :");for(auto token:o.tokens)std::printf(" %lld",(long long)token);
        std::printf("\noutput  :");for(auto token:generated)std::printf(" %d",token);std::printf("\n");
        const double prefill=std::chrono::duration<double>(prefetched-start).count();
        const double decode=std::chrono::duration<double>(finished-prefetched).count();
        std::printf("DeepSeek prefill: %zu tokens in %.3f s; decode: %zu tokens in %.3f s (%.2f tok/s)\n",
                    o.tokens.size(),prefill,generated.size(),decode,generated.size()/std::max(decode,1e-9));
        const auto stats=model.stats();
        std::printf("DSpark: %d/%d accepted; experts: primary %llu, helper %llu, CPU %llu\n",accepted,proposed,
                    (unsigned long long)(stats.primary_entries-metric_base.primary_entries),
                    (unsigned long long)(stats.helper_entries-metric_base.helper_entries),
                    (unsigned long long)(stats.cpu_entries-metric_base.cpu_entries));
        return 0;
    } catch(const std::exception& error) {basalt::log::error("basalt deepseek41","%s",error.what());return 1;}
}
} // namespace basalt::program::generate
