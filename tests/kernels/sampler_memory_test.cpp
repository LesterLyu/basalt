#include "basalt/kernels/sampler.hpp"
#include "basalt/core/memory_audit.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    try {
        require(cudaSetDevice(0) == cudaSuccess, "select device");
        cudaStream_t stream{};
        require(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess, "create stream");
        require(basalt::kernels::sampler_cache_bytes(0) == 0, "unallocated sampler cache is nonzero");
        require(basalt::kernels::sampler_wide_reserve(129280, stream), "reserve model vocabulary");
        const uint64_t first = basalt::kernels::sampler_cache_bytes(0);
        require(first >= 4ull * 8 * 129280 * sizeof(float), "owned buffers are missing");
        require(basalt::kernels::sampler_wide_reserve(129280, stream), "reuse cached buffers");
        require(basalt::kernels::sampler_cache_bytes(0) == first, "reuse double-counted a buffer");
        require(basalt::kernels::sampler_wide_reserve(248320, stream), "grow vocabulary");
        const uint64_t grown = basalt::kernels::sampler_cache_bytes(0);
        require(grown >= first + 4ull * 8 * 248320 * sizeof(float), "growth lost retired buffer ownership");
        require(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed) == cudaSuccess, "begin capture");
        basalt::core::enable_memory_audit(true);
        { basalt::core::MemoryAuditScope audit("already-capturing", stream); }
        basalt::core::enable_memory_audit(false);
        require(cudaGetLastError() == cudaSuccess, "memory audit disturbed capture");
        require(!basalt::kernels::sampler_wide_reserve(248321, stream), "allocated while capturing");
        require(basalt::kernels::sampler_cache_bytes(0) == grown, "refused grow changed ownership");
        cudaGraph_t graph{};
        require(cudaStreamEndCapture(stream, &graph) == cudaSuccess, "capture was invalidated");
        require(cudaGraphDestroy(graph) == cudaSuccess, "destroy graph");
        require(cudaStreamDestroy(stream) == cudaSuccess, "destroy stream");
        std::puts("sampler_memory_test: ok");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
