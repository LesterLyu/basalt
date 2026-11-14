# Windows runtime details

See [platforms](platforms.md) for requirements, platform seams and limitations.

## Building with newer CUDA headers

From a Visual Studio Developer command prompt:

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCMAKE_CXX_FLAGS="/EHsc /arch:AVX2 /Zc:preprocessor" -DCMAKE_CUDA_FLAGS="-Xcompiler=/Zc:preprocessor"
cmake --build build -j
```

The standard preprocessor flag is required by newer CCCL headers, including CUDA 13.3.
The AVX2 flag supplies the CPU router warm-up's intrinsics on MSVC, where GCC's
per-function target attribute is unavailable. Scalar half conversion uses the portable
vector spelling of the same F16C conversion.

## Qwen elastic KV in TCC

When a TCC device does not expose CUDA virtual memory management, Qwen resizes
its KV pools with ordinary CUDA allocations between inference windows. The
expert cache can return memory in small blocks, while its contiguous region
continues to lend workspace to prefill. Devices with VMM keep the original path.

## DeepSeek resident expert memory

DeepSeek uses ordinary CUDA host registration on TCC: its expert transfers need a host
address that CUDA can copy from without blocking the submission thread. WDDM retains
its existing host NUMA VMM path. CPU reads use the host alias; kernels use the device
address. If any layer cannot be mapped, the engine releases the mappings before
allocating ordinary RAM and registering what it can. Linux and WSL use ordinary RAM registration.
For diagnostics, set `BASALT_DS_PROFILE_PEER_EXCHANGE=1` to time helper exchanges.
By default the engine neither times these exchanges nor prints their timing summary.

## Direct reads and resident memory

GPU-cache fills read expert blobs concurrently into pinned staging buffers and upload them
on independent CUDA streams. The resident RAM complement uses unbuffered reads into its
arena. Descriptors use `NO_BUFFERING | OVERLAPPED`; each positional read has its own offset
and completion event. Buffers, offsets and byte counts retain a 4 KiB alignment contract.
A file system that refuses unbuffered reads, or a short read, retains the mapped-copy fallback.

The compact and full resident arenas use ordinary CUDA host registration on TCC.
Elsewhere they try host VMM and check that the CPU aliases are contiguous across the
backing allocations. The compact arena registers ordinary host memory after its reads
complete when it cannot use VMM. Registration queries each visible
adapter's current DXGI non-local budget and usage, leaving the larger of 1 GiB and 5% of its
budget for later allocations. The adapter contexts are initialized before registration so
later context creation does not inherit a large mapping. The smallest remaining allowance bounds the portable mapping;
it is not divided by the number of devices. With multiple WDDM devices, or unknown driver modes,
the target is also bounded by DXGI's available plus existing reservation allowance. A single WDDM
card alongside TCC uses the non-local budget without that additional cap. This is a conservative working-set
target, not a promise that the driver accepts every registration up to it. After loading, the complement is registered
in chunks of about 1 GiB within that allowance. Cuts lie at aligned expert starts, so no expert spans registrations;
each chunk keeps its own device alias. A refusal stops at the last registered chunk. The registered prefix supports DMA and mapped
GPU reads; only the remaining tail is locked with a working-set minimum plus chunked
`VirtualLock`. CUDA-registered pages are not locked again. Startup reports CUDA-mapped and
resident sizes separately and reports any working-set quota failure.

The full expert arena registers ordinary host memory before loading its bytes,
using the same budget query and whole-layer registration slices. A driver refusal stops the slice probe; unregistered bytes
remain in the resident tail. The registration target leaves the reserve before probing;
it does not rely on recovering a context after driver exhaustion. Startup reports the budget and prefix.
The remaining bytes stay resident through the working-set lock; the complete RAM copy is retained.
If the OS cannot report a budget, registration falls back to the driver's result.

Automatic peer sizing takes the smaller of CUDA's allocation allowance and the management
driver's physical free memory, after accounting for its scratch allocations. The query uses
the device's PCI bus ID; an unavailable management driver retains CUDA's budget. This startup
snapshot bounds the expert tier but does not reserve memory against later allocations by other apps.

Native session RAM spans use the same device-aware portable registration allowance as the full arena.
Whole spans beyond that allowance stay resident through the working-set lock. The native peer's capacity query
also uses physical headroom, so its reserve is applied after other applications' local VRAM usage.

Before sizing the resident copy, the loader drops the clean expert view left by the GPU fill.
Each copied layer drops its view again. `VirtualUnlock` trims unlocked file pages without
unmapping the view; `ERROR_NOT_LOCKED` is the documented success condition for that trim.
This keeps the source view and its RAM copy from remaining resident together. Per-descriptor
file-cache advice remains a no-op.

The platform memory test queries page residency around an unaligned one-byte trim, then
reads the original bytes again. See the
[VirtualUnlock contract](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualunlock)
and the [direct-read API](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-reopenfile).

## Peer window launch

A persistent host thread keeps the expert peer's CUDA context current and enqueues its windows.
The main thread remains on the primary device while the peer waits for its first doorbell. This
avoids switching contexts after starting a kernel that depends on work the caller has not launched.
Stall diagnostics and timeout messages read mapped host words without querying the peer stream,
so reporting a stall cannot block on the same driver context lock before releasing GPU waits.

## Tensor-map transport

The build passes `/Zc:__cplusplus` to MSVC and nvcc's host pass so CUDA headers select the
correct tensor-map alignment. When those headers declare `CUtensorMap` with 128-byte alignment,
the kernel launch packet transports its opaque bytes in a 64-byte-aligned wrapper, within
MSVC's by-value parameter alignment limit. Driver API encoding still uses the original type;
the bytes are copied unchanged. Both platforms use the same packet layout, with compile-time
checks of the packet and each descriptor offset against the GPU's 64-byte requirement.

Transport identity measurements are recorded in [precision/windows-transport.md](precision/windows-transport.md).
