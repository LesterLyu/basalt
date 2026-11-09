#include "basalt/core/memory_audit.hpp"
#include "basalt/platform/system.hpp"
#include "basalt/log.hpp"

#include <cuda_runtime.h>
#include <atomic>

namespace basalt::core {
namespace {
std::atomic<bool> enabled{false};
std::atomic<uint64_t> sequence{0};
thread_local uint64_t parent = 0;

void snapshot(int64_t& cuda_free, int64_t& physical_free) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) cuda_free = (int64_t) free_b;
    int device = 0;
    char bus[32]{};
    uint64_t physical = 0;
    if (cudaGetDevice(&device) == cudaSuccess && cudaDeviceGetPCIBusId(bus, sizeof bus, device) == cudaSuccess &&
        platform::gpu_physical_free(bus, physical)) physical_free = (int64_t) physical;
}
}

void enable_memory_audit(bool on) { enabled.store(on, std::memory_order_relaxed); }
bool memory_audit_enabled() { return enabled.load(std::memory_order_relaxed); }

MemoryAuditScope::MemoryAuditScope(const char* phase, void* stream) {
    if (!phase || !memory_audit_enabled()) return;
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaGetDevice(&device_) != cudaSuccess ||
        cudaStreamIsCapturing((cudaStream_t) stream, &status) != cudaSuccess || status != cudaStreamCaptureStatusNone)
        return;
    phase_ = phase;
    id_ = sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    parent_ = parent;
    parent = id_;
    snapshot(cuda_free_, physical_free_);
    log::info("basalt memory", "audit begin id=%llu parent=%llu gpu=%d phase=%s cuda_free=%lld physical_free=%lld",
              (unsigned long long) id_, (unsigned long long) parent_, device_, phase_,
              (long long) cuda_free_, (long long) physical_free_);
}

MemoryAuditScope::~MemoryAuditScope() { finish(); }

void MemoryAuditScope::finish() {
    if (!phase_) return;
    int current = -1;
    int64_t after = -1, physical = -1;
    if (cudaGetDevice(&current) == cudaSuccess && current == device_) snapshot(after, physical);
    log::info("basalt memory", "audit end id=%llu parent=%llu gpu=%d phase=%s requested=%llu cuda_before=%lld cuda_after=%lld physical_before=%lld physical_after=%lld",
              (unsigned long long) id_, (unsigned long long) parent_, device_, phase_, (unsigned long long) requested_,
              (long long) cuda_free_, (long long) after, (long long) physical_free_, (long long) physical);
    parent = parent_;
    phase_ = nullptr;
}

}  // namespace basalt::core
