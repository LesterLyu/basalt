#pragma once

#include <cstdint>

namespace basalt::core {

// Diagnostic snapshots only; no queries or synchronization on the ordinary path.
void enable_memory_audit(bool enabled);
bool memory_audit_enabled();

class MemoryAuditScope {
public:
    explicit MemoryAuditScope(const char* phase, void* stream = nullptr);
    ~MemoryAuditScope();
    MemoryAuditScope(const MemoryAuditScope&) = delete;
    MemoryAuditScope& operator=(const MemoryAuditScope&) = delete;
    void requested(uint64_t bytes) { requested_ += bytes; }
    void finish();
private:
    const char* phase_ = nullptr;
    uint64_t id_ = 0, parent_ = 0, requested_ = 0;
    int device_ = -1;
    int64_t cuda_free_ = -1, physical_free_ = -1;
};

}  // namespace basalt::core
