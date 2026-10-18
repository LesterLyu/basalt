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

## Direct reads and resident memory

GPU-cache fills read expert blobs concurrently into pinned staging buffers and upload them
on independent CUDA streams. The resident RAM complement uses unbuffered reads into its
arena. Descriptors use `NO_BUFFERING | OVERLAPPED`; each positional read has its own offset
and completion event. Buffers, offsets and byte counts retain a 4 KiB alignment contract.
A file system that refuses unbuffered reads, or a short read, retains the mapped-copy fallback.

The resident arena is registered after its reads complete. Registration queries each visible
adapter's current DXGI non-local budget and usage, leaving the larger of 1 GiB and 5% of its
budget for later allocations. The adapter contexts are initialized before registration so
later context creation does not inherit a large mapping. The smallest remaining allowance bounds the portable mapping;
it is not divided by the number of devices. With several visible devices, the target is also
bounded by DXGI's available plus existing reservation allowance. This is a conservative working-set
target, not a promise that the driver accepts every registration up to it. After loading, the complement is registered
in chunks of about 1 GiB within that allowance. Cuts lie at aligned expert starts, so no expert spans registrations;
each chunk keeps its own device alias. A refusal stops at the last registered chunk. The registered prefix supports DMA and mapped
GPU reads; only the remaining tail is locked with a working-set minimum plus chunked
`VirtualLock`. CUDA-registered pages are not locked again. Startup reports CUDA-mapped and
resident sizes separately and reports any working-set quota failure.

The full expert arena registers before loading its bytes, using the same budget query and
whole-layer registration slices. A driver refusal stops the slice probe; unregistered bytes
remain in the resident tail. The registration target leaves the reserve before probing;
it does not rely on recovering a context after driver exhaustion. Startup reports the budget and prefix.
The remaining bytes stay resident through the working-set lock; the complete RAM copy is retained.
If the OS cannot report a budget, registration falls back to the driver's result.

Before sizing the resident copy, the loader drops the clean expert view left by the GPU fill.
Each copied layer drops its view again. `VirtualUnlock` trims unlocked file pages without
unmapping the view; `ERROR_NOT_LOCKED` is the documented success condition for that trim.
This keeps the source view and its RAM copy from remaining resident together. Per-descriptor
file-cache advice remains a no-op.

The platform memory test queries page residency around an unaligned one-byte trim, then
reads the original bytes again. See the
[VirtualUnlock contract](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualunlock)
and the [direct-read API](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-reopenfile).

## Tensor-map transport

The build passes `/Zc:__cplusplus` to MSVC and nvcc's host pass so CUDA headers select the
correct tensor-map alignment. When those headers declare `CUtensorMap` with 128-byte alignment,
the kernel launch packet transports its opaque bytes in a 64-byte-aligned wrapper, within
MSVC's by-value parameter alignment limit. Driver API encoding still uses the original type;
the bytes are copied unchanged. Both platforms use the same packet layout, with compile-time
checks of the packet and each descriptor offset against the GPU's 64-byte requirement.

Transport identity measurements are recorded in [precision/builds.md](precision/builds.md).
