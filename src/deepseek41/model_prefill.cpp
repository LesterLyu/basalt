#include "model_state.hpp"
#include "expert_tiers.hpp"
#include "basalt/platform/memory.hpp"
#include <algorithm>
#include <chrono>
#include <future>
#include <atomic>
#include <cstring>

namespace basalt::deepseek41 {
namespace {
struct LoanStaging {
    void* data=nullptr;
    ~LoanStaging(){if(data)(void)cudaFreeHost(data);}
};
struct TileViews {
    Work& w;
    float *h=w.h,*x=w.x,*post=w.post,*comb=w.comb,*weights=w.weights;
    int32_t* ids=w.ids;
    ~TileViews() {w.h=h;w.x=x;w.post=post;w.comb=comb;w.weights=weights;w.ids=ids;}
};
struct PromptLoan {
    core::ExpertCache& cache;
    core::FileExpertSource& source;
    DeviceModel& device;
    std::vector<std::pair<int32_t,int32_t>> homes;
    std::vector<int32_t> slots;
    int64_t first=cache.slots();
    LoanStaging staging;
    uint64_t ram_min=UINT64_MAX;
    uint64_t copied_bytes=0;

    uint64_t sample_ram() {
        const auto available=platform::available_memory();
        if(available)ram_min=std::min(ram_min,available);
        return available;
    }
    double min_ram_mib() const {return ram_min==UINT64_MAX?-1.:ram_min/1048576.;}
    void abort_copies() {
        const auto before=sample_ram();
        const auto released=source.homes_abort();
        const auto after=sample_ram();
        basalt::log::info("basalt deepseek41","prompt loan rollback: copied %.1f MiB, released %.1f MiB, RAM available %.1f -> %.1f MiB (min %.1f MiB)",
                          copied_bytes/1048576.,released/1048576.,before/1048576.,after/1048576.,min_ram_mib());
    }
    uint8_t* take(size_t bytes,int tokens,std::string& error,const std::function<void()>& between_copies) {
        if(bytes+255>=size_t(cache.bytes())) {error="workspace exceeds expert cache";return nullptr;}
        first=cache.slots_within(cache.bytes()-int64_t(bytes+255));
        if(first<=0 || first>=cache.slots()) {error="no lendable expert cache tail";return nullptr;}
        const auto& layout=kernels::cpu::expert_layout();
        size_t staging_bytes=0;
        for(int l=0;l<40;++l)staging_bytes=std::max(staging_bytes,size_t(layout.blob_bytes(l)));
        uint64_t temporary_bytes=0;
        for(int l=0;l<40;++l)for(int e=0;e<384;++e) {
            const int slot=cache.slot_of(l,e);
            if(slot>=first && slot<cache.slots() && !source.has_resident(l,e)) {
                homes.emplace_back(l,e);slots.push_back(slot);
                temporary_bytes+=layout.blob_bytes(l);
            }
        }
        basalt::log::info("basalt deepseek41","prompt loan plan: %d tokens, workspace %.1f MiB, temporary RAM %.1f MiB (%zu experts), pinned buffer %.1f MiB, RAM available %.1f MiB (floor %.0f MiB)",
                          tokens,bytes/1048576.,temporary_bytes/1048576.,homes.size(),staging_bytes/1048576.,
                          sample_ram()/1048576.,kExpertRamFloor/1048576.);
        device.synchronize();
        void* transfer=nullptr;
        const auto allocation=cudaMallocHost(&transfer,staging_bytes);
        if(allocation==cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            error="CUDA expert transfer buffer: out of memory";
            return nullptr;
        }
        cuda_check(allocation);
        staging.data=transfer;
        std::vector<uint8_t*> dst;
        try {
            // The large temporary copies stay pageable. DMA only uses one blocking, pinned transfer buffer.
            if(!homes.empty() && !source.homes_take(homes,error,dst,false,kExpertRamFloor))return nullptr;
            for(size_t i=0;i<homes.size();++i) {
                const size_t n=layout.blob_bytes(homes[i].first);
                const auto status=cudaMemcpy(staging.data,cache.device_slot(slots[i]),n,cudaMemcpyDeviceToHost);
                if(status==cudaErrorMemoryAllocation) {
                    (void)cudaGetLastError();
                    abort_copies();
                    error="CUDA expert copy: out of memory";
                    return nullptr;
                }
                cuda_check(status);
                std::memcpy(dst[i],staging.data,n);
                copied_bytes+=n;
                if((i+1)%8==0 || i+1==homes.size()) {
                    const auto free=sample_ram();
                    if(free && free<kExpertRamFloor) {
                        abort_copies();
                        error="expert copies reached the RAM floor";
                        return nullptr;
                    }
                    if(between_copies && (i+1)%8==0)between_copies();
                }
            }
        } catch(...) {
            abort_copies();
            throw;
        }
        if(!homes.empty())source.homes_commit();
        const auto base=reinterpret_cast<uintptr_t>(cache.device_slot(int32_t(first)));
        return reinterpret_cast<uint8_t*>((base+255)&~uintptr_t(255));
    }

    void restore(const std::function<void()>& between_copies) {
        std::string error;
        unsigned copies=0;
        for(int l=0;l<40;++l)for(int e=0;e<384;++e) {
            const int slot=cache.slot_of(l,e);
            if(slot<first || slot>=cache.slots())continue;
            const auto* blob=source.resident_blob(l,e);
            expect(blob,"missing RAM copy for prompt loan");
            const auto n=kernels::cpu::expert_layout().blob_bytes(l);
            std::memcpy(staging.data,blob,n);
            expect(cache.fill_slot_blocking(slot,static_cast<const uint8_t*>(staging.data),error,n),error);
            if(between_copies && ++copies%8==0)between_copies();
        }
        device.synchronize();
        if(core::exp_flag("ds-check-prefill-loan")!=0 && !homes.empty()) {
            for(size_t i:{size_t(0),homes.size()-1}) {
                const auto [l,e]=homes[i];
                expect(cache.verify_slot(slots[i],source.resident_blob(l,e),error,
                                         kernels::cpu::expert_layout().blob_bytes(l)),error);
            }
        }
        for(const auto [l,e]:homes)expect(source.home_release(l,e),"cannot release prompt loan RAM");
        sample_ram();
        const auto released=homes.empty()?0:source.homes_reclaim();
        const auto available=sample_ram();
        basalt::log::info("basalt deepseek41","prompt loan returned, %lld expert slots",
                          (long long)(cache.slots()-first));
        basalt::log::info("basalt deepseek41","prompt loan RAM: released %.1f MiB, available %.1f MiB (min %.1f MiB)",
                          released/1048576.,available/1048576.,min_ram_mib());
    }
};
}
LongWork::LongWork(DeviceModel& d,int context,int n):borrowed(core::exp_flag("ds-prefill-borrow")!=0) {
    if(!borrowed) {
        const auto bytes=layout(nullptr,n,context);
        layout(d.allocate(bytes,MemoryKind::long_prefill),n,context);
    }
}
size_t LongWork::layout(uint8_t* base,int n,int end_position) {
    capacity=n;
    candidate_stride=(end_position+7)/8;
    const size_t t=size_t(n);
    size_t bytes=0;
    const auto place=[&]<typename T>(T*& p,size_t count) {
        p=base && count?reinterpret_cast<T*>(base+bytes):nullptr;
        bytes+=(count*sizeof(T)+255)&~size_t(255);
    };
    place(compressed_bf16,core::exp_flag("ds-prefill-compressed-bf16")!=0?t*512:0);
    place(window_bf16,core::exp_flag("ds-prefill-window-bf16")!=0?size_t(kPrefillWindow+128)*512:0);
    place(rotary,core::exp_flag("ds-prefill-rotary-table")!=0?t*4*64:0);
    place(h,t*4*Shape::hidden);place(x,t*Shape::hidden);
    place(pre,t*4);place(post,t*4);place(comb,t*16);
    place(accumulator,t*Shape::hidden);place(weights,t*6);
    place(ids,t*6);place(selected,t*512);place(candidates,t*candidate_stride);
    if(base)host_ids.resize(t*6);
    return bytes;
}

std::optional<ForwardResult> Model::Impl::prefill_long(const std::vector<int32_t>& tokens,LogitsMode mode,
                                                     const std::function<void()>& between_steps,const std::function<void()>& check_stop) {
    expect(!broken && !pending && long_work && prefill_tensor && prompt_experts,"long prompt path unavailable");
    const int count=int(tokens.size()),start=pos,H=Shape::hidden;
    expect(count>0 && count<=options.tensor_prefill_tokens && count<=options.context-pos,"invalid long prompt length");
    expect(mode!=LogitsMode::all || count<=window_capacity,"all-logits long prompts exceed the bounded readback path");
    try {
        const core::OnDevice on(0);auto& d=*primary;auto& w=*main_work;auto& large=*long_work;
        using Clock=std::chrono::steady_clock;
        auto last_yield=Clock::now();
        cooperative_prefill=bool(between_steps);
        struct ClearCooperation {bool& flag;~ClearCooperation(){flag=false;}} clear_cooperation{cooperative_prefill};
        std::function<void()> yield;
        if(between_steps || check_stop)yield=[&] {
            if(Clock::now()-last_yield<std::chrono::milliseconds(35))return;
            // Tile views and any submitted expert copies have completed. No decode
            // work can mutate expert placement until the prompt loan is returned.
            if(check_stop)check_stop();
            if(between_steps)pause_prefill(between_steps);
            last_yield=Clock::now();
        };
        struct BackgroundStream {
            cudaStream_t stream=nullptr;
            ~BackgroundStream(){if(stream)cudaStreamDestroy(stream);}
        } background;
        const bool background_requested=between_steps && core::exp_flag("ds-prefill-background")!=0 && !profile.prefill_enabled;
        // Draft attention only reads its sliding-window KV after the prompt.
        const int draft_first=drafter?std::max(0,count-Shape::window):count;
        if(helper.valid())helper.apply_pending(true);
        PromptLoan loan{cache,source,d};
        bool loan_active=false;
        const auto return_loan=[&](const std::function<void()>& step) {
            if(!loan_active)return;
            loan.restore(step);
            loan_active=false;
            borrowed_prefill_bytes=0;
            if(cooperative_prefill) {
                prompt_loan_first=INT64_MAX;
                refresh_primary_lookup();
            }
            if(memory_changed)memory_changed();
        };
        try {
        if(large.borrowed) {
            const auto bytes=large.layout(nullptr,count,start+count);
            std::string error;
            if(cooperative_prefill && bytes>uint64_t(cache.bytes())/4) {
                basalt::log::info("basalt deepseek41","prompt loan %d tokens skipped: workspace %.1f MiB exceeds concurrent limit %.1f MiB",
                                  count,bytes/1048576.,cache.bytes()/4194304.);
                return std::nullopt;
            }
            if(!platform::available_memory()) {
                basalt::log::warn("basalt deepseek41","prompt loan %d tokens unavailable: no available RAM reading",count);
                return std::nullopt;
            }
            auto* base=loan.take(bytes,count,error,yield);
            if(!base) {
                basalt::log::warn("basalt deepseek41","prompt loan %d tokens unavailable: %s",count,error.c_str());
                return std::nullopt;
            }
            loan_active=true;
            large.layout(base,count,start+count);
            borrowed_prefill_bytes=uint64_t(cache.bytes()-cache.bytes_of(loan.first));
            if(cooperative_prefill) {
                prompt_loan_first=loan.first;
                refresh_primary_lookup();
            }
            if(memory_changed)memory_changed();
            basalt::log::info("basalt deepseek41","prompt loan %.2f MiB, %lld expert slots",
                              bytes/1048576.,(long long)(cache.slots()-loan.first));
        }
        const auto stream=static_cast<cudaStream_t>(d.stream());
        const size_t candidate_width=(start+count+7)/8;
        saved_history=history;base_pos=pos;
        large.rotary_count=0;
        large.compressed_source=nullptr;large.compressed_count=0;
        large.compressed_active=large.compressed_bf16 && core::exp_flag("ds-prefill-compressed-bf16")!=0;
        struct ResetScratch {LongWork& w;~ResetScratch(){w.compressed_active=false;w.rotary_count=0;}} reset_scratch{large};
        if(large.rotary && core::exp_flag("ds-prefill-rotary-table")!=0) {
            large.rotary_first=start;large.rotary_count=count;
            for(int kind=0;kind<4;++kind)
                make_rotary_pairs(large.rotary+size_t(kind)*large.capacity*64,
                                  kind>=2?w.freq_compressed:w.freq_window,start,count,(kind&1)!=0,d.stream());
        }
        host_engram.resize(size_t(count)*48*144);row_ids.resize(size_t(count)*48);
        for(int t=0;t<count;++t) {
            const auto ids=hasher.advance(tokens[t],history);
            std::copy(ids.begin(),ids.end(),row_ids.begin()+size_t(t)*48);
        }
        engram_ticket=engram_reader.issue(row_ids.data(),count*48,host_engram.data());engram_pending=true;
        for(int first=0;first<count;first+=window_capacity) {
            const int n=std::min(window_capacity,count-first);
            embeddings(std::vector<int32_t>(tokens.begin()+first,tokens.begin()+first+n),d,w);
            d.copy(large.h+size_t(first)*4*H,w.h,size_t(n)*4*H*4);
            d.copy(large.pre+size_t(first)*4,w.pre,size_t(n)*4*4);
            if(yield)yield();
        }
        for(auto& l:layers) {
            using Clock=std::chrono::steady_clock;const auto layer_start=Clock::now();
            for(int first=0;first<count;first+=window_capacity) {
                const int n=std::min(window_capacity,count-first);
                {
                    TileViews saved{w};
                    w.h=large.h+size_t(first)*4*H;w.x=large.x+size_t(first)*H;
                    w.post=large.post+size_t(first)*4;w.comb=large.comb+size_t(first)*16;
                    w.ids=large.ids+size_t(first)*6;w.weights=large.weights+size_t(first)*6;
                    d.copy(w.pre,large.pre+size_t(first)*4,size_t(n)*4*4);
                    if(!l.index_source && l.ratio)d.copy(w.selected,large.selected+size_t(first)*512,size_t(n)*512*4);
                    if(l.index_source && l.id>20)
                        cuda_check(cudaMemcpy2DAsync(w.candidates,w.candidate_stride,large.candidates+size_t(first)*large.candidate_stride,
                                                    large.candidate_stride,candidate_width,n,cudaMemcpyDeviceToDevice,stream));
                    if(l.engram)engram(l,n,first);
                    if(l.id>=37 && first+n>draft_first) {
                        const int anchor_first=std::max(first,draft_first);
                        stream_mean(w.h+size_t(anchor_first-first)*4*H,w.anchors+size_t(anchor_first-draft_first)*3*H,
                                    H,first+n-anchor_first,3*H,size_t(l.id-37)*H,d.stream());
                    }
                    block_prefix(d,w,l,start+first,n,false);
                    if(l.index_source)d.copy(large.selected+size_t(first)*512,w.selected,size_t(n)*512*4);
                    if(l.id==20)
                        cuda_check(cudaMemcpy2DAsync(large.candidates+size_t(first)*large.candidate_stride,large.candidate_stride,
                                                    w.candidates,w.candidate_stride,candidate_width,n,cudaMemcpyDeviceToDevice,stream));
                    d.copy(large.pre+size_t(first)*4,w.f_pre,size_t(n)*4*4);
                    AsyncGpuScope shared(profile.prefill_enabled?&profile:nullptr,"long_shared",d.stream());
                    d.linear(*l.shared_gate,w.x,w.gate,n);d.linear(*l.shared_up,w.x,w.up,n);
                    kernels::native_swiglu_f32(w.gate,w.up,w.activated,n*2304,10,d.stream());
                    d.linear(*l.shared_down,w.activated,w.shared,n);
                    seed_experts_f64(w.shared,large.accumulator+size_t(first)*H,size_t(n)*H,d.stream());
                }
                if(yield)yield();
            }
            d.read(large.host_ids.data(),large.ids,size_t(count)*6*4);
            for(int j=0;j<count*6;++j) {
                const int e=large.host_ids[j];expect(e>=0 && e<384,"invalid long-prompt expert");
                usage[size_t(l.id)*384+e]+=1;
                const auto slot=cache.slot_of(l.id,e);
                if(slot>=0 && slot<loan.first)++execution.primary_entries;
                else if(!(helper.valid() && helper.has(l.id,e)))++execution.streamed_entries;
            }
            const auto prefix_done=Clock::now();
            if(background_requested) {
                if(!background.stream) {
                    int low=0,high=0;
                    cuda_check(cudaDeviceGetStreamPriorityRange(&low,&high));
                    cuda_check(cudaStreamCreateWithPriority(&background.stream,cudaStreamNonBlocking,low));
                    basalt::log::info("basalt deepseek41","background prefill experts: low-priority stream");
                }
                ExecutionStats prefill_stats;
                // The read above has drained the prefix stream. Cache and helper ownership stay fixed while
                // both slots read weights; the worker only writes this slot's MMQ and accumulation buffers.
                std::atomic<bool> stop_worker=false;
                auto work=std::async(std::launch::async,[&] {
                    prompt_experts->run_long(l.id,large.x,large.host_ids,large.weights,count,6,cache,source,helper,
                                             w.gpu_parts,large.accumulator,prefill_stats,loan.first,
                                             nullptr,[&]{if(stop_worker.load())throw PrefillCancelled{};},background.stream);
                });
                try {
                    while(work.wait_for(std::chrono::milliseconds(2))!=std::future_status::ready) {
                        if(check_stop)check_stop();
                        pause_prefill(between_steps);
                    }
                } catch(...) {
                    stop_worker.store(true);
                    throw;
                }
                work.get();
                execution.streamed_bytes+=prefill_stats.streamed_bytes;
                execution.peer_streamed_bytes+=prefill_stats.peer_streamed_bytes;
            } else {
                AsyncGpuScope experts(profile.prefill_enabled?&profile:nullptr,"long_experts",d.stream());
                prompt_experts->run_long(l.id,large.x,large.host_ids,large.weights,count,6,cache,source,helper,
                                         w.gpu_parts,large.accumulator,execution,loan.first,
                                         profile.prefill_enabled && core::exp_flag("ds-prefill-expert-breakdown")!=0?&profile:nullptr,
                                         yield);
            }
            const auto experts_done=Clock::now();
            for(int first=0;first<count;first+=window_capacity) {
                const int n=std::min(window_capacity,count-first);
                {
                    TileViews saved{w};
                    w.h=large.h+size_t(first)*4*H;
                    w.post=large.post+size_t(first)*4;w.comb=large.comb+size_t(first)*16;
                    finish_experts_f64(large.accumulator+size_t(first)*H,w.branch,size_t(n)*H,d.stream());
                    hc_expand(w.branch,w.h,w.post,w.comb,w.h,H,n,d.stream());
                }
                if(yield)yield();
            }
            d.synchronize();
            if(loan_active)loan.sample_ram();
            if(profile.prefill_enabled)profile.collect_async();
            const auto done=Clock::now();const auto ms=[](auto a,auto b){return std::chrono::duration<double,std::milli>(b-a).count();};
            basalt::log::info("basalt deepseek41","long prefill layer %d/40 tokens=%d prefix_ms=%.1f experts_ms=%.1f merge_ms=%.1f",
                              l.id+1,count,ms(layer_start,prefix_done),ms(prefix_done,experts_done),ms(experts_done,done));
        }
        ForwardResult result;
        if(mode!=LogitsMode::none) {
            const int n=mode==LogitsMode::all?count:1,first=mode==LogitsMode::all?0:count-1;
            d.copy(w.h,large.h+size_t(first)*4*H,size_t(n)*4*H*4);d.copy(w.pre,large.pre+size_t(first)*4,size_t(n)*4*4);
            hc_reduce(w.h,w.pre,d.f32("output_norm.weight",{H}),w.head_x,H,n,1e-20f,d.stream());
            d.set_prefill_tensor(false);result.rows=n;result.logits.resize(size_t(n)*Shape::vocab);
            for(int row=0;row<n;row+=w.logit_capacity) {
                const int nr=std::min(w.logit_capacity,n-row);
                d.linear("output.weight",w.head_x+size_t(row)*H,w.logits,nr);
                d.read(result.logits.data()+size_t(row)*Shape::vocab,w.logits,size_t(nr)*Shape::vocab*4);
            }
            d.set_prefill_tensor(true);
            expect(std::all_of(result.logits.begin(),result.logits.end(),[](float x){return std::isfinite(x);}),"nonfinite long prompt logits");
        }
        d.synchronize();pos=start+count;logit_rows=result.rows<=w.logit_capacity?result.rows:0;
        return_loan(yield);
        if(drafter)seed_draft(start+draft_first,count-draft_first);
        if(!options.freeze_placement && !cooperative_prefill) {
            std::string error;
            if(core::exp_flag("ds-primary-adapt",1)!=0) {
                const auto before=execution.primary_swaps;
                expect(adapt_primary(cache,source,helper,usage.data(),40,384,256,primary_residency.data(),
                                     static_cast<cudaStream_t>(d.stream()),execution.primary_swaps,execution.peer_promotions,error),error);
                if(execution.primary_swaps!=before)refresh_primary_lookup();
            }
            if(helper.valid())expect(helper.adapt(usage.data(),primary_residency.data(),8,error),error);
        }
        return result;
        } catch(const PrefillCancelled&) {
            d.synchronize();
            return_loan({});
            if(engram_pending) {
                std::string error;
                const bool ok=engram_reader.collect(engram_ticket,error);
                engram_pending=false;
                expect(ok,error);
            }
            throw;
        }
    }catch(const PrefillCancelled&) {throw;}
     catch(...) {broken=true;throw;}
}
}
