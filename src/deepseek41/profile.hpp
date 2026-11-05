#pragma once
#include "basalt/core/diag.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <memory>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>
#include "trace.hpp"

namespace basalt::deepseek41 {
// Separate events for the draft device. External record nodes preserve phase
// boundaries when the existing DSpark body is captured as a CUDA graph.
class DraftGpuStages {
    static constexpr int layers=3,stages=4,event_count=layers*stages+1;
    int device_;
    std::array<cudaEvent_t,event_count> events_{};
    std::array<double,stages+1> times_{};
    uint64_t rounds_=0;
    static void check(cudaError_t e) {
        if(e!=cudaSuccess)throw std::runtime_error(std::string("DSpark timing: ")+cudaGetErrorString(e));
    }
    void close() noexcept {
        const core::OnDevice on(device_);
        for(auto event:events_)if(event)cudaEventDestroy(event);
    }
public:
    explicit DraftGpuStages(int device):device_(device) {
        const core::OnDevice on(device_);
        try {for(auto& event:events_)check(cudaEventCreate(&event));}
        catch(...) {close();throw;}
    }
    ~DraftGpuStages(){close();}
    DraftGpuStages(const DraftGpuStages&)=delete;
    void record(int layer,int stage,void* stream) {
        const auto s=static_cast<cudaStream_t>(stream);
        cudaStreamCaptureStatus capture;
        check(cudaStreamIsCapturing(s,&capture));
        const auto ev=events_[size_t(layer)*stages+stage];
        check(capture==cudaStreamCaptureStatusActive?cudaEventRecordWithFlags(ev,s,cudaEventRecordExternal):cudaEventRecord(ev,s));
    }
    void collect() {
        for(int i=0;i+1<event_count;++i) {
            float ms=0;check(cudaEventElapsedTime(&ms,events_[i],events_[i+1]));
            times_[i%stages]+=ms;
        }
        float ms=0;check(cudaEventElapsedTime(&ms,events_.front(),events_.back()));
        times_.back()+=ms;++rounds_;
    }
    void reset() {times_={};rounds_=0;}
    void report() const {
        if(!rounds_)return;
        basalt::log::info("basalt deepseek41","DSpark GPU body rounds=%llu pre_HC_ms=%.3f attention_ms=%.3f post_HC_router_ms=%.3f experts_tail_ms=%.3f layer_total_ms=%.3f",
                         (unsigned long long)rounds_,times_[0],times_[1],times_[2],times_[3],times_[4]);
    }
};
struct Profile {
    bool front_enabled=core::exp_flag("ds-front-metrics")!=0;
    bool nvtx_enabled=core::exp_flag("ds-trace-tokens")>0;
    bool enabled=front_enabled || core::diagnostics().profile || core::exp_flag("ds-cpu-timing")!=0 || core::exp_flag("ds-metrics")!=0;
    struct Entry { double ms=0; uint64_t calls=0; };
    std::map<std::string,Entry> entries;
    bool gpu_enabled=enabled && core::exp_flag("ds-gpu-timing")!=0;
    std::map<std::string,Entry> gpu_entries;
    uint64_t gpu_failures=0;
    bool prefill_enabled=core::exp_flag("ds-prefill-metrics")!=0;
    bool async_enabled=front_enabled || prefill_enabled || core::exp_flag("ds-metrics")!=0;
    int64_t event_first=core::exp_flag("ds-metrics-from",-1),event_count=core::exp_flag("ds-metrics-count",16);
    bool event_sample=event_first<0;
    Profile() {
        if(event_first<-1 || event_first>1048576 || event_count<1 || event_count>128)
            throw std::invalid_argument("invalid bounded GPU metric range");
        if(event_first>=0 && gpu_enabled)throw std::invalid_argument("bounded GPU metrics require asynchronous timing");
    }
    // Decode-position bounds keep both host and GPU samples out of prompt construction.
    void select_position(int position,bool decode) {
        if(event_first>=0)event_sample=decode && position>=event_first && position-event_first<event_count;
    }
    struct Stamp { cudaEvent_t begin=nullptr,end=nullptr;const char* name=nullptr; };
    std::vector<std::unique_ptr<Stamp>> stamps;
    size_t used=0;
    std::map<std::string,Entry> async_entries;
    cudaError_t record(cudaEvent_t event,void* stream) {
        const auto s=static_cast<cudaStream_t>(stream);
        if(front_enabled) {
            cudaStreamCaptureStatus status;
            const auto error=cudaStreamIsCapturing(s,&status);if(error!=cudaSuccess)return error;
            // External event nodes retain timestamps inside a captured graph.
            if(status==cudaStreamCaptureStatusActive)return cudaEventRecordWithFlags(event,s,cudaEventRecordExternal);
        }
        return cudaEventRecord(event,s);
    }
    ~Profile() {
        if(stamps.empty())return;
        const core::OnDevice on(0);
        for(const auto& s:stamps) {if(s->begin)cudaEventDestroy(s->begin);if(s->end)cudaEventDestroy(s->end);}
    }
    size_t begin_async(const char* name,void* stream) {
        if(!async_enabled || !event_sample)return size_t(-1);
        if(used==stamps.size()) {
            auto s=std::make_unique<Stamp>();
            if(cudaEventCreate(&s->begin)!=cudaSuccess || cudaEventCreate(&s->end)!=cudaSuccess) {
                if(s->begin)cudaEventDestroy(s->begin);if(s->end)cudaEventDestroy(s->end);
                ++gpu_failures;return size_t(-1);
            }
            stamps.push_back(std::move(s));
        }
        const size_t index=used++;auto& s=*stamps[index];s.name=name;
        const auto recorded=record(s.begin,stream);
        if(recorded!=cudaSuccess) {++gpu_failures;if(front_enabled)throw std::runtime_error(std::string("front event record: ")+cudaGetErrorString(recorded));}
        return index;
    }
    void end_async(size_t index,void* stream) {
        if(index!=size_t(-1) && record(stamps[index]->end,stream)!=cudaSuccess)++gpu_failures;
    }
    // The caller has already completed its forward readback; event queries add no stream waits.
    void collect_async() {
        for(size_t i=0;i<used;++i) {
            const auto& s=*stamps[i];float ms=0;
            if(cudaEventQuery(s.end)==cudaSuccess && cudaEventElapsedTime(&ms,s.begin,s.end)==cudaSuccess) {
                auto& e=async_entries[s.name];e.ms+=ms;++e.calls;
            } else ++gpu_failures;
        }
        used=0;
    }
    void reset_metrics() {
        collect_async();
        for(auto& [name,e]:entries)e={};
        for(auto& [name,e]:gpu_entries)e={};
        for(auto& [name,e]:async_entries)e={};
        gpu_failures=0;
    }
    void report()const {
        if(enabled)for(const auto& [name,entry]:entries)
            basalt::log::info("basalt deepseek41 profile","%s wall_ms=%.3f calls=%llu mean_ms=%.3f",name.c_str(),entry.ms,
                              (unsigned long long)entry.calls,entry.ms/std::max<uint64_t>(entry.calls,1));
        if(gpu_enabled)for(const auto& [name,entry]:gpu_entries)
            basalt::log::info("basalt deepseek41 profile","%s gpu_elapsed_ms=%.3f calls=%llu mean_ms=%.3f",name.c_str(),entry.ms,
                              (unsigned long long)entry.calls,entry.ms/std::max<uint64_t>(entry.calls,1));
        if(gpu_failures)basalt::log::warn("basalt deepseek41 profile","GPU timing failures %llu",(unsigned long long)gpu_failures);
        if(async_enabled)for(const auto& [name,e]:async_entries)
            basalt::log::info("basalt deepseek41 metrics","%s stream_ms=%.3f calls=%llu mean_us=%.3f",name.c_str(),e.ms,
                             (unsigned long long)e.calls,1000*e.ms/std::max<uint64_t>(e.calls,1));
    }
};
class AsyncGpuScope {
    Profile* profile_;void* stream_;size_t index_=size_t(-1);
public:
    AsyncGpuScope(Profile* p,const char* name,void* stream):profile_(p),stream_(stream) {
        if(p)index_=p->begin_async(name,stream);
    }
    ~AsyncGpuScope(){if(profile_)profile_->end_async(index_,stream_);}
};
class ProfileScope {
    using Clock=std::chrono::steady_clock;
    Profile& profile_;
    const char* name_;
    Clock::time_point started_;
    bool active_;
public:
    ProfileScope(Profile& p,const char* name):profile_(p),name_(name),active_(p.enabled && p.event_sample) {
        if(p.nvtx_enabled)trace_push(name);
        if(active_)started_=Clock::now();
    }
    ~ProfileScope() {
        if(profile_.nvtx_enabled)trace_pop();
        if(active_) {auto& e=profile_.entries[name_];e.ms+=std::chrono::duration<double,std::milli>(Clock::now()-started_).count();++e.calls;}
    }
};
// Stream elapsed time includes dependencies and host submission gaps; nested intervals must not be summed.
class GpuProfileScope {
    Profile& profile_;
    const char* name_;
    cudaStream_t stream_;
    cudaEvent_t start_=nullptr,stop_=nullptr;
    bool active_=false;
public:
    GpuProfileScope(Profile& p,const char* name,void* stream):profile_(p),name_(name),stream_(static_cast<cudaStream_t>(stream)) {
        if(!p.gpu_enabled)return;
        active_=cudaEventCreate(&start_)==cudaSuccess && cudaEventCreate(&stop_)==cudaSuccess && cudaEventRecord(start_,stream_)==cudaSuccess;
        if(!active_)++p.gpu_failures;
    }
    ~GpuProfileScope() {
        if(active_) {
            float ms=0;
            if(cudaEventRecord(stop_,stream_)==cudaSuccess && cudaEventSynchronize(stop_)==cudaSuccess && cudaEventElapsedTime(&ms,start_,stop_)==cudaSuccess) {
                auto& e=profile_.gpu_entries[name_];e.ms+=ms;++e.calls;
            }else ++profile_.gpu_failures;
        }
        if(start_)cudaEventDestroy(start_);if(stop_)cudaEventDestroy(stop_);
    }
};
} // namespace basalt::deepseek41
