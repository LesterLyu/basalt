// Individually releasable native expert RAM. A GPU home keeps no host allocation.
#pragma once
#include "basalt/core/ram_load.hpp"
#include "basalt/core/vmm.hpp"
#include "basalt/platform/memory.hpp"

namespace basalt::models::deepseek41 {
// Host VMM may be addressable from TCC without being a fast asynchronous host-copy endpoint.
inline uint64_t expert_host_vmm_granularity() {
    if constexpr (platform::kDirectHostVmmExchange)
        if (!core::vmm::tcc_mode()) return core::vmm::host_granularity();
    return 0;
}

struct ExpertRam {
    uint8_t* data = nullptr;
    uint64_t bytes = 0;
    core::vmm::HostMapping mapping;
    core::RamPin pin;
    bool ready = false;

    ExpertRam() = default;
    ExpertRam(const ExpertRam&) = delete;
    ExpertRam& operator=(const ExpertRam&) = delete;
    ~ExpertRam() {
        if (!data) return;
        if (mapping.device) core::vmm::free_host(mapping);
        else { core::unpin_ram(data, pin); platform::free_aligned(data); }
    }
    bool allocate(uint64_t size) {
        bytes = size;
        if (expert_host_vmm_granularity()) {
            mapping = core::vmm::alloc_host(bytes);
            data = mapping.host;
        }
        if (!data) {
            data = core::alloc_ram(bytes);
            if (!data) return false;
            pin = core::pin_ram(data, bytes, bytes, {}, false);
        }
        return true;
    }
    uint64_t allocated() const { return data ? (mapping.device ? mapping.bytes : bytes) : 0; }
    uint64_t registered() const { return mapping.device ? mapping.bytes : pin.registered_bytes; }
};
}
