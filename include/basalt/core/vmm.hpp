// include/basalt/core/vmm.hpp - the driver's virtual memory management, looked up through the runtime.
//
// One reserved address range, backed piece by piece by physical allocations that can be unmapped and mapped again
// while every address in the range stays the same.  The expert cache's segments and the elastic KV pools both use
// it: captured graphs and kernel arguments hold addresses into these ranges, so only the backing moves.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::core::vmm {

/// Whether the current device supports virtual memory management (and the driver entry points were found).
bool supported();
/// Whether the current CUDA device runs in TCC driver mode.
bool tcc_mode();
#ifdef _WIN32
/// TCC can advertise VMM even when the driver rejects device backing.
bool can_map_device_memory();
#endif
/// The allocation granularity of the current device (every size and offset below is a multiple of it); 0 on failure.
uint64_t granularity();
/// The minimum host allocation granularity on the current device, or 0 without host VMM.
uint64_t host_granularity();
/// Bounded backing lets a large arena exceed the driver's single-allocation allowance.
enum class HostBacking { Whole, Bounded };
struct HostMapping {
    uint8_t* host = nullptr;    ///< CPU alias, contiguous across all pieces
    uint8_t* device = nullptr;  ///< reserved CUDA VA, also used to unmap the pieces
    uint64_t bytes = 0;
    uint64_t stride = 0;        ///< size of one physical backing allocation
};
/// Map CUDA-pinned system RAM into the current GPU and, by default, the other visible TCC GPUs.
/// An empty mapping leaves the caller free to use ordinary RAM registration.
HostMapping alloc_host(uint64_t bytes, HostBacking backing = HostBacking::Whole, bool map_peers = true);
void free_host(HostMapping mapping);
/// Reserves `bytes` of address space (a multiple of the granularity); null on failure.
uint8_t* reserve(uint64_t bytes);
void release_range(uint8_t* va, uint64_t bytes);
/// Backs `[va, va + bytes)` with a new physical allocation on the current device, readable and writable by it.
/// Returns the handle (0 on failure: nothing is left mapped).
unsigned long long map(uint8_t* va, uint64_t bytes);
/// Unmaps what `map` mapped there and frees its physical memory.  The caller has made sure no work reads it.
bool unmap(uint8_t* va, uint64_t bytes, unsigned long long handle);
/// A physical allocation of `bytes` on the current device, mapped nowhere yet (0 on failure): one backing that
/// `map_handle` can put under several ranges at once, so they share its bytes.  `release` drops the caller's reference;
/// the memory is freed once every mapping of it is gone too.
unsigned long long create(uint64_t bytes);
/// Maps the first `bytes` of `handle` at `va`, readable and writable by the current device.
bool map_handle(uint8_t* va, uint64_t bytes, unsigned long long handle);
/// Unmaps `[va, va + bytes)` without releasing any handle (the mapping's reference goes with it).
void unmap_only(uint8_t* va, uint64_t bytes);
void release(unsigned long long handle);

}  // namespace basalt::core::vmm
