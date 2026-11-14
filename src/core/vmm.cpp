// src/core/vmm.cpp - the driver's virtual memory functions through the runtime's entry points (no link against the
// driver library).  Read the header first.
#include "basalt/core/vmm.hpp"
#include "basalt/log.hpp"

#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>
#include <vector>

namespace basalt::core::vmm {
namespace {
struct Fns {
    CUresult (CUDAAPI* device_get)(CUdevice*, int) = nullptr;
    CUresult (CUDAAPI* attribute)(int*, CUdevice_attribute, CUdevice) = nullptr;
    CUresult (CUDAAPI* granularity)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags) = nullptr;
    CUresult (CUDAAPI* reserve)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long) = nullptr;
    CUresult (CUDAAPI* address_free)(CUdeviceptr, size_t) = nullptr;
    CUresult (CUDAAPI* create)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*,
                               unsigned long long) = nullptr;
    CUresult (CUDAAPI* release)(CUmemGenericAllocationHandle) = nullptr;
    CUresult (CUDAAPI* map)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long) = nullptr;
    CUresult (CUDAAPI* unmap)(CUdeviceptr, size_t) = nullptr;
    CUresult (CUDAAPI* set_access)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) = nullptr;
    CUresult (CUDAAPI* pointer_attribute)(void*, CUpointer_attribute, CUdeviceptr) = nullptr;
    bool ok = false;
};

template <class F> bool entry(const char* name, F& f) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
#if CUDART_VERSION >= 12050
    const cudaError_t e = cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q);
#else
    const cudaError_t e = cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &q);
#endif
    if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || p == nullptr) {
        (void) cudaGetLastError();
        return false;
    }
    f = reinterpret_cast<F>(p);
    return true;
}

const Fns& fns() {
    static const Fns v = [] {
        Fns x;
        x.ok = entry("cuDeviceGet", x.device_get) && entry("cuDeviceGetAttribute", x.attribute) &&
               entry("cuMemGetAllocationGranularity", x.granularity) && entry("cuMemAddressReserve", x.reserve) &&
               entry("cuMemAddressFree", x.address_free) && entry("cuMemCreate", x.create) &&
               entry("cuMemRelease", x.release) && entry("cuMemMap", x.map) && entry("cuMemUnmap", x.unmap) &&
               entry("cuMemSetAccess", x.set_access);
#ifdef _WIN32
        entry("cuPointerGetAttribute", x.pointer_attribute);
#endif
        return x;
    }();
    return v;
}

int current_device() {
    int dev = 0;
    cudaGetDevice(&dev);
    return dev;
}

CUmemAllocationProp device_prop(int dev) {
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    return prop;
}
}  // namespace

bool supported() {
    const Fns& v = fns();
    int supported = 0;
    CUdevice cu = 0;
    return v.ok && v.device_get(&cu, current_device()) == CUDA_SUCCESS &&
           v.attribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cu) == CUDA_SUCCESS &&
           supported != 0;
}

bool tcc_mode() {
    int dev = 0, tcc = 0;
    return cudaGetDevice(&dev) == cudaSuccess &&
           cudaDeviceGetAttribute(&tcc, cudaDevAttrTccDriver, dev) == cudaSuccess && tcc != 0;
}

uint64_t granularity() {
    const Fns& v = fns();
    if (!v.ok) return 0;
    const CUmemAllocationProp prop = device_prop(current_device());
    size_t g = 0;
    if (v.granularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS) return 0;
    return (uint64_t) g;
}

namespace {
// Separate backing allocations avoid the driver's single-allocation limit while preserving one address range.
constexpr uint64_t kHostPieceBytes = 2ull << 30;

bool host_prop(CUmemAllocationProp& prop, size_t& g) {
    const Fns& v = fns();
    int dev = 0, supported = 0, node = -1;
    CUdevice cu;
    if (!v.ok || cudaGetDevice(&dev) != cudaSuccess || v.device_get(&cu, dev) != CUDA_SUCCESS ||
        v.attribute(&supported, CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cu) != CUDA_SUCCESS ||
        !supported || v.attribute(&node, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, cu) != CUDA_SUCCESS) return false;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location = {CU_MEM_LOCATION_TYPE_HOST_NUMA, node < 0 ? 0 : node};
    return v.granularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_SUCCESS && g;
}
}

uint64_t host_granularity() {
    CUmemAllocationProp prop{};
    size_t g = 0;
    return host_prop(prop, g) ? g : 0;
}

HostMapping alloc_host(uint64_t bytes, HostBacking backing, bool map_peers) {
    const Fns& v = fns();
    CUmemAllocationProp prop{};
    size_t g = 0;
    if (!bytes || !host_prop(prop, g)) return {};
    if (backing == HostBacking::Bounded && kHostPieceBytes % g != 0) return {};
    const int dev = current_device();
    if (bytes > SIZE_MAX - (g - 1)) return {};
    size_t n = (bytes + g - 1) / g * g;
    uint64_t stride = backing == HostBacking::Bounded ? kHostPieceBytes : n;
    std::vector<CUmemAccessDesc> access = {{prop.location, CU_MEM_ACCESS_FLAGS_PROT_READWRITE},
        {{CU_MEM_LOCATION_TYPE_DEVICE, dev}, CU_MEM_ACCESS_FLAGS_PROT_READWRITE}};
#ifdef _WIN32
    if (!v.pointer_attribute) return {};
    if (backing == HostBacking::Bounded && n > kHostPieceBytes) {
        // Use equal pieces so the driver's separate CPU aliases can remain contiguous.
        const uint64_t count = 1 + (n - 1) / kHostPieceBytes;
        stride = (1 + (n / g - 1) / count) * g;
        if (count > SIZE_MAX / stride) return {};
        n = count * stride;
    }
    int count = 0;
    if (map_peers && tcc_mode() && cudaGetDeviceCount(&count) == cudaSuccess) {
        for (int peer = 0; peer < count; ++peer) {
            int tcc = 0, host_numa = 0;
            CUdevice cu = 0;
            if (peer != dev && cudaDeviceGetAttribute(&tcc, cudaDevAttrTccDriver, peer) == cudaSuccess && tcc &&
                v.device_get(&cu, peer) == CUDA_SUCCESS &&
                v.attribute(&host_numa, CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cu) ==
                    CUDA_SUCCESS && host_numa)
                access.push_back({{CU_MEM_LOCATION_TYPE_DEVICE, peer}, CU_MEM_ACCESS_FLAGS_PROT_READWRITE});
        }
    }
#else
    (void) map_peers;
#endif
    CUdeviceptr p = 0;
    uint8_t* host = nullptr;
    const auto refused = [&](const char* op, CUresult error, uint64_t off) {
        basalt::log::warn("basalt memory", "host VMM %s at %.2f of %.2f GiB on CUDA%d failed (driver code %d)",
                          op, off / 1073741824.0, n / 1073741824.0, dev, (int) error);
    };
    CUresult error = v.reserve(&p, n, 0, 0, 0);
    if (error != CUDA_SUCCESS) { refused("reserve", error, 0); return {}; }
    uint64_t off = 0;
    for (; off < n;) {
        const uint64_t piece = std::min<uint64_t>(stride, n - off);
        CUmemGenericAllocationHandle h;
        const char* op = "create";
        error = v.create(&h, piece, &prop, 0);
        if (error == CUDA_SUCCESS) {
            op = "map";
            error = v.map(p + off, piece, 0, h, 0);
            v.release(h);   // a successful mapping owns the physical allocation
            if (error == CUDA_SUCCESS) {
                op = "access";
                error = v.set_access(p + off, piece, access.data(), access.size());
#ifdef _WIN32
                if (error == CUDA_SUCCESS) {
                    // A device VA is not necessarily CPU-accessible, even for system memory.
                    void* alias = nullptr;
                    op = "CPU alias";
                    error = v.pointer_attribute(&alias, CU_POINTER_ATTRIBUTE_HOST_POINTER, p + off);
                    if (error == CUDA_SUCCESS && !alias) error = CUDA_ERROR_INVALID_VALUE;
                    if (error == CUDA_SUCCESS && off == 0) host = static_cast<uint8_t*>(alias);
                    if (error == CUDA_SUCCESS && reinterpret_cast<uintptr_t>(alias) !=
                            reinterpret_cast<uintptr_t>(host) + off) {
                        op = "noncontiguous CPU aliases";
                        error = CUDA_ERROR_NOT_SUPPORTED;
                    }
                }
#endif
                if (error == CUDA_SUCCESS) { off += piece; continue; }
                v.unmap(p + off, piece);
            }
        }
        refused(op, error, off);
        for (uint64_t at = 0; at < off; at += stride)
            v.unmap(p + at, std::min<uint64_t>(stride, off - at));
        v.address_free(p, n);
        return {};
    }
    auto* device = reinterpret_cast<uint8_t*>(p);
    return {host ? host : device, device, n, stride};
}

void free_host(HostMapping mapping) {
    if (!mapping.device) return;
    const auto p = reinterpret_cast<CUdeviceptr>(mapping.device);
    const uint64_t mapped = mapping.bytes;
    const uint64_t stride = mapping.stride;
    for (uint64_t off = 0; off < mapped; off += stride)
        fns().unmap(p + off, std::min<uint64_t>(stride, mapped - off));
    fns().address_free(p, mapped);
}

uint8_t* reserve(uint64_t bytes) {
    const Fns& v = fns();
    CUdeviceptr va = 0;
    if (!v.ok || v.reserve(&va, (size_t) bytes, 0, 0, 0) != CUDA_SUCCESS) return nullptr;
    return reinterpret_cast<uint8_t*>(va);
}

void release_range(uint8_t* va, uint64_t bytes) {
    const Fns& v = fns();
    if (v.ok && va != nullptr && bytes > 0) v.address_free(reinterpret_cast<CUdeviceptr>(va), (size_t) bytes);
}

unsigned long long map(uint8_t* va, uint64_t bytes) {
    const Fns& v = fns();
    if (!v.ok) return 0;
    const int dev = current_device();
    const CUmemAllocationProp prop = device_prop(dev);
    const CUdeviceptr p = reinterpret_cast<CUdeviceptr>(va);
    CUmemGenericAllocationHandle h = 0;
    if (v.create(&h, (size_t) bytes, &prop, 0) != CUDA_SUCCESS) return 0;
    if (v.map(p, (size_t) bytes, 0, h, 0) != CUDA_SUCCESS) {
        v.release(h);
        return 0;
    }
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = dev;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (v.set_access(p, (size_t) bytes, &access, 1) != CUDA_SUCCESS) {
        v.unmap(p, (size_t) bytes);
        v.release(h);
        return 0;
    }
    return (unsigned long long) h;
}

bool unmap(uint8_t* va, uint64_t bytes, unsigned long long handle) {
    const Fns& v = fns();
    if (!v.ok || handle == 0) return false;
    const bool a = v.unmap(reinterpret_cast<CUdeviceptr>(va), (size_t) bytes) == CUDA_SUCCESS;
    const bool b = v.release((CUmemGenericAllocationHandle) handle) == CUDA_SUCCESS;
    return a && b;
}

#ifdef _WIN32
bool can_map_device_memory() {
    if (!supported()) return false;
    const uint64_t bytes = granularity();
    if (bytes == 0) return false;
    uint8_t* va = reserve(bytes);
    if (!va) return false;
    const auto handle = map(va, bytes);
    const bool ok = handle != 0 && unmap(va, bytes, handle);
    release_range(va, bytes);
    return ok;
}
#endif

}  // namespace basalt::core::vmm
