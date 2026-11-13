#pragma once

#include "basalt/core/memory_audit.hpp"
#include "basalt/log.hpp"

#include <cuda_runtime.h>
#include <string>

namespace basalt::models::deepseek41 {

// A failed capture has not run the body. After an allocation failure, submit the same work directly
// and stop trying new graphs in this family; previously instantiated shapes still replay.
template <class F>
bool launch_decode_graph(cudaGraphExec_t& exec, bool& direct, cudaStream_t stream, unsigned long long flags,
                         const char* phase, F&& body, std::string& err) {
    if (!exec && direct) return body();
    if (!exec) {
        core::MemoryAuditScope audit(phase, stream);
        cudaError_t status = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
        const char* call = "begin capture";
        if (status == cudaSuccess) {
            cudaGraph_t graph = nullptr;
            bool ok;
            try {
                ok = body();
            } catch (...) {
                cudaStreamEndCapture(stream, &graph);
                if (graph) cudaGraphDestroy(graph);
                throw;
            }
            status = cudaStreamEndCapture(stream, &graph);
            call = "end capture";
            if (ok && status == cudaSuccess) {
                status = cudaGraphInstantiateWithFlags(&exec, graph, flags);
                call = "instantiate";
            }
            if (graph) cudaGraphDestroy(graph);
            if (!ok) return false;
        }
        if (status != cudaSuccess) {
            if (status != cudaErrorMemoryAllocation) {
                err = std::string(phase) + ": " + call + ": " + cudaGetErrorString(status);
                return false;
            }
            (void) cudaGetLastError();
            direct = true;
            audit.finish();
            basalt::log::warn("basalt deepseek41", "%s: CUDA out of memory; submitting new shapes without graphs", phase);
            return body();
        }
    }
    const cudaError_t status = cudaGraphLaunch(exec, stream);
    if (status == cudaSuccess) return true;
    err = std::string(phase) + ": launch: " + cudaGetErrorString(status);
    return false;
}

}  // namespace basalt::models::deepseek41
