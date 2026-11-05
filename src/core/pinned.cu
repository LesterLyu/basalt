// src/core/pinned.cu - the pinned host arena.
#include "basalt/core/pinned.hpp"
#include "basalt/core/ram_load.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>
#include "basalt/log.hpp"

namespace basalt::core {

bool host_pin_budget(uint64_t& limit, std::string& why) {
    limit = 0;
    why.clear();
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) { (void) cudaGetLastError(); return false; }
    std::vector<cudaDeviceProp> props((size_t) count);
    for (int d = 0; d < count; ++d) {
        if (cudaGetDeviceProperties(&props[(size_t) d], d) != cudaSuccess) { (void) cudaGetLastError(); continue; }
        uint64_t unused = 0;
        std::string note;
        const auto& prop = props[(size_t) d];
        if (!platform::pin_budget(unused, note, prop.luid, prop.luidDeviceNodeMask)) continue;
        // Create mapping consumers before registering portable pages, without changing the caller's device.
        const cudaError_t initialized = cudaInitDevice(d, 0, 0);
        if (initialized != cudaSuccess) {
            why = "CUDA" + std::to_string(d) + " context initialization refused: " + cudaGetErrorString(initialized);
            (void) cudaGetLastError();
            return true;
        }
    }
    bool known = false;
    for (int d = 0; d < count; ++d) {
        const auto& prop = props[(size_t) d];
        uint64_t allowance = 0;
        std::string note;
        if (!platform::pin_budget(allowance, note, prop.luid, prop.luidDeviceNodeMask, count > 1)) continue;
        if (!known || allowance < limit) {
            limit = allowance;
            why = "CUDA" + std::to_string(d) + " " + note;
        }
        known = true;
    }
    return known;
}

uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed) {
    uint64_t h = seed;
    for (uint64_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

namespace {
bool clear_error() { (void) cudaGetLastError(); return true; }

std::vector<uint64_t> uniform_bounds(uint64_t bytes, uint64_t slice) {
    std::vector<uint64_t> b;
    if (slice == 0) return b;
    for (uint64_t off = 0; off + slice <= bytes; off += slice) b.push_back(off);
    if (!b.empty()) b.push_back(b.back() + slice);
    return b;
}
}  // namespace

PinnedArena::PinnedArena(uint64_t bytes, uint64_t slice) : PinnedArena(bytes, uniform_bounds(bytes, slice)) {
    if (slice_bytes) slice_bytes = slice;   // sliced registration: record the uniform size
}

PinnedArena::PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds,
                         const std::string& shared_file, uint64_t shared_hash, uint64_t max_pinned_bytes) : capacity(bytes) {
    if (bytes == 0) return;
    basalt::platform::RegionBacking region_backing = basalt::platform::RegionBacking::NormalPages;
    base = basalt::platform::alloc_region(bytes, region_backing, note, shared_file, shared_hash, mapping_base,
                                          mapping_bytes);
    backing = region_backing == basalt::platform::RegionBacking::LargePages ? PageBacking::LargePages
                                                                             : PageBacking::NormalPages;

    // Register with CUDA BEFORE any page is touched: cudaHostRegister pins what is resident now, and a region
    // that has already been faulted in page by page is far more expensive to register and may fail outright.
    if (base) {
        uint64_t cap = max_pinned_bytes;
        std::string cap_why = "the engine registration cap";
        bool capped = cap > 0 && cap < bytes;
        uint64_t allowance = 0;
        std::string budget_note;
        if (host_pin_budget(allowance, budget_note) && allowance < bytes && (!capped || allowance < cap)) {
            cap = allowance;
            cap_why = std::move(budget_note);
            capped = true;
        }
        const cudaError_t e = capped ? cudaSuccess :
            cudaHostRegister(base, (size_t) bytes, cudaHostRegisterPortable | cudaHostRegisterMapped);
        if (!capped && e == cudaSuccess) {
            note = "cudaHostRegister PORTABLE ok; " + note;
            registered_bytes = bytes;
        } else if (bounds.size() >= 2 && (capped || clear_error())) {
            // the whole range is refused, so pin it slice by slice from the start.  The rest stays resident through the
            // working-set lock below.  (Slices may differ in size, one per layer.)
            slice_bytes = 1;   // sliced; the uniform constructor records the size
            const bool limited = capped;
            const uint64_t limit = capped ? cap : bytes;
            std::string refusal;
            for (size_t i = 0; i + 1 < bounds.size(); ++i) {
                const uint64_t off = bounds[i], n = bounds[i + 1] - bounds[i];
                if (limited && (off > limit || n > limit - off)) break;
                const cudaError_t registered = cudaHostRegister((uint8_t*) base + off, (size_t) n,
                                                               cudaHostRegisterPortable | cudaHostRegisterMapped);
                if (registered != cudaSuccess) {
                    refusal = "; slice registration refused: " + std::string(cudaGetErrorString(registered));
                    (void) cudaGetLastError();
                    break;
                }
                slice_starts.push_back(off);
                registered_bytes = off + n;
                ++registered_slices;
            }
            char gib[32];
            std::snprintf(gib, sizeof gib, "%.1f", (double) limit / (double) (1ull << 30));
            note = (capped ? "cudaHostRegister slices capped at " + std::string(gib) + " GiB (" + cap_why + "); " :
                             "cudaHostRegister of the whole arena FAILED (" + std::string(cudaGetErrorString(e)) + "); ") +
                   std::to_string(registered_slices) + " slices pinned (" + std::to_string(registered_bytes >> 30) +
                   " GiB)" + refusal + "; " + note;
            if (registered_bytes < bytes) {
                const basalt::platform::LockResult lr =
                    basalt::platform::lock_resident((uint8_t*) base + registered_bytes, bytes - registered_bytes);
                locked_bytes = lr.locked_bytes;
                note = lr.note + "; " + note;
            }
        } else {
            note = (capped ? "no CUDA registration within " + cap_why :
                            std::string("cudaHostRegister FAILED (") + cudaGetErrorString(e) + ")") +
                   " - the arena is NOT pinned, so copies will be slow; " + note;
            // **CONSUME THE ERROR, OR IT LIES ABOUT SOMETHING ELSE LATER.**
            //
            // `cudaGetLastError()` returns the last error and CLEARS it; until something reads it, the error
            // state is sticky.  This failure is caught and handled right here - the arena is simply not pinned -
            // but leaving it set meant the next `cudaGetLastError()` in the engine, which is `gr_read`'s launch
            // check, reported "out of memory" for kernels that allocate nothing.  That cost a round: the arena
            // was written off as not fitting the machine when in fact the only thing wrong was a stale error
            // from this line.
            //
            // It is the same trap `gr.cu` warns about for ASYNC faults, in the other direction: a synchronous
            // failure is sticky too, and it lies about where it happened just as convincingly.
            (void) cudaGetLastError();
            // Keep it resident instead of pinned, so the OS does not page the arena out.
            const basalt::platform::LockResult lr = basalt::platform::lock_resident(base, bytes);
            locked_bytes = lr.locked_bytes;
            note = lr.note + "; " + note;
        }
    }
}

PinnedArena::~PinnedArena() {
    if (base) {
        if (locked_bytes) basalt::platform::unlock_resident((uint8_t*) base + (slice_bytes ? registered_bytes : 0), locked_bytes);
        if (slice_bytes) {
            for (uint64_t off : slice_starts) cudaHostUnregister((uint8_t*) base + off);
        } else if (registered_bytes) {
            cudaHostUnregister(base);
        }
        basalt::platform::free_region(mapping_base ? mapping_base : base, mapping_bytes ? mapping_bytes : capacity);
        base = nullptr;
        mapping_base = nullptr;
        mapping_bytes = 0;
    }
}

StreamStats stream_bandwidth(const uint8_t* src, uint64_t bytes, uint64_t chunk, int iters) {
    StreamStats st;
    st.bytes = bytes * (uint64_t) iters;
    st.chunk = chunk;
    uint8_t* dst = nullptr;
    cudaStream_t s{};
    if (cudaMalloc(&dst, (size_t) chunk) != cudaSuccess) {
        basalt::log::error("basalt", "stream_bandwidth: cudaMalloc failed for %llu B", (unsigned long long) chunk);
        st.seconds = -1.0;
        return st;
    }
    cudaStreamCreate(&s);

    // one untimed pass so the first transfer's page-fault and setup cost is not in the measurement
    for (uint64_t off = 0; off + chunk <= bytes; off += chunk) {
        cudaMemcpyAsync(dst, src + off, (size_t) chunk, cudaMemcpyHostToDevice, s);
    }
    cudaStreamSynchronize(s);

    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        for (uint64_t off = 0; off + chunk <= bytes; off += chunk) {
            cudaMemcpyAsync(dst, src + off, (size_t) chunk, cudaMemcpyHostToDevice, s);
        }
    }
    cudaStreamSynchronize(s);
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    cudaStreamDestroy(s);
    cudaFree(dst);
    return st;
}

}  // namespace basalt::core
