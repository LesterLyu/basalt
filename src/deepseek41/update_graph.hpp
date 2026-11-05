#pragma once
#include "basalt/core/on_device.hpp"
#include "profile.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace basalt::deepseek41 {
// Recapture dynamic positions/plans while reusing an executable when its topology permits an update.
class UpdateGraph {
    cudaGraphExec_t exec_=nullptr;
    int device_=0;
    static void check(cudaError_t e) {
        if(e!=cudaSuccess)throw std::runtime_error(std::string("deepseek41 graph: ")+cudaGetErrorString(e));
    }
public:
    uint64_t updates=0,rebuilds=0,cached_launches=0;
    UpdateGraph(){check(cudaGetDevice(&device_));}
    ~UpdateGraph(){const core::OnDevice on(device_);if(exec_)cudaGraphExecDestroy(exec_);}
    UpdateGraph(const UpdateGraph&)=delete;
    template<class Submit> void run_static(void* stream,Submit&& submit,Profile* metrics,const char* name) {
        if(!exec_) {run(stream,std::forward<Submit>(submit),metrics,name);return;}
        AsyncGpuScope timing(metrics,name,stream);
        check(cudaGraphLaunch(exec_,static_cast<cudaStream_t>(stream)));++cached_launches;
    }
    template<class Submit> void run(void* stream,Submit&& submit,Profile* metrics=nullptr,const char* name=nullptr) {
        const bool sampled=metrics && metrics->enabled && metrics->event_sample;
        const auto started=sampled?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        const auto s=static_cast<cudaStream_t>(stream);
        check(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));
        cudaGraph_t graph=nullptr;
        try {submit();}
        catch(...) {cudaStreamEndCapture(s,&graph);if(graph)cudaGraphDestroy(graph);throw;}
        check(cudaStreamEndCapture(s,&graph));
        try {
            bool replaced=false;
            if(exec_) {
                cudaGraphExecUpdateResultInfo info{};
                const auto status=cudaGraphExecUpdate(exec_,graph,&info);
                if(status!=cudaSuccess && status!=cudaErrorGraphExecUpdateFailure)check(status);
                if(status==cudaErrorGraphExecUpdateFailure)cudaGetLastError();
                replaced=status==cudaSuccess && info.result==cudaGraphExecUpdateSuccess;
                if(replaced)++updates;
            }
            if(!replaced) {
                cudaGraphExec_t next=nullptr;check(cudaGraphInstantiate(&next,graph,0));
                if(exec_)cudaGraphExecDestroy(exec_);exec_=next;++rebuilds;
            }
            if(sampled) {
                auto& e=metrics->entries["graph_capture_update"];
                e.ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();++e.calls;
            }
            AsyncGpuScope timing(metrics,name,stream);
            check(cudaGraphLaunch(exec_,s));
        }catch(...) {cudaGraphDestroy(graph);throw;}
        check(cudaGraphDestroy(graph));
    }
};
}
