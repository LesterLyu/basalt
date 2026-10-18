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

The resident arena is registered after its reads complete. Registration is capped by the
shared-memory budget. If the driver refuses that range, the loader retries smaller prefixes
ending at both a whole expert and an OS page. The registered prefix supports DMA and mapped
GPU reads; only the remaining tail is locked with a working-set minimum plus chunked
`VirtualLock`. CUDA-registered pages are not locked again. Startup reports CUDA-mapped and
resident sizes separately and reports any working-set quota failure.

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
