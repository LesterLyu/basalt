// Individually releasable native expert RAM. A GPU home keeps no host allocation.
#pragma once
#include "basalt/core/ram_load.hpp"
#include "basalt/core/vmm.hpp"
#include "basalt/platform/memory.hpp"

namespace basalt::models::deepseek41 {
struct ExpertRam {
    uint8_t* data = nullptr;
    uint64_t bytes = 0, mapped = 0;
    core::RamPin pin;
    bool ready = false;

    ExpertRam() = default;
    ExpertRam(const ExpertRam&) = delete;
    ExpertRam& operator=(const ExpertRam&) = delete;
    ~ExpertRam() {
        if (!data) return;
        if (mapped) core::vmm::free_host(data, mapped);
        else { core::unpin_ram(data, pin); platform::free_aligned(data); }
    }
    bool allocate(uint64_t size) {
        bytes = size;
        if constexpr (platform::kDirectHostVmmExchange) data = core::vmm::alloc_host(bytes, mapped);
        if (!data) {
            data = core::alloc_ram(bytes);
            if (!data) return false;
            pin = core::pin_ram(data, bytes, bytes, {}, false);
        }
        return true;
    }
    uint64_t allocated() const { return data ? (mapped ? mapped : bytes) : 0; }
    uint64_t registered() const { return mapped ? mapped : pin.registered_bytes; }
};
}
