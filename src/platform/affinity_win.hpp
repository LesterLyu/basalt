#pragma once

#include "basalt/platform/win.hpp"

#include <cstddef>
#include <vector>

namespace basalt::platform::detail {

inline bool get_thread_cpu_sets(std::vector<unsigned long>& ids) {
    ULONG count = 0;
    if (!GetThreadSelectedCpuSets(GetCurrentThread(), nullptr, 0, &count) &&
        GetLastError() != ERROR_INSUFFICIENT_BUFFER) return false;
    ids.resize(count);
    if (count == 0) return true;
    if (!GetThreadSelectedCpuSets(GetCurrentThread(), ids.data(), count, &count)) return false;
    ids.resize(count);
    return true;
}

inline std::vector<SYSTEM_CPU_SET_INFORMATION> cpu_sets() {
    ULONG bytes = 0;
    if (!GetSystemCpuSetInformation(nullptr, 0, &bytes, GetCurrentProcess(), 0) &&
        GetLastError() != ERROR_INSUFFICIENT_BUFFER) return {};
    std::vector<unsigned char> buffer(bytes);
    if (bytes && !GetSystemCpuSetInformation(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data()),
                                            bytes, &bytes, GetCurrentProcess(), 0)) return {};
    std::vector<SYSTEM_CPU_SET_INFORMATION> result;
    constexpr size_t header = offsetof(SYSTEM_CPU_SET_INFORMATION, CpuSet);
    for (size_t offset = 0; offset + header <= bytes;) {
        const auto* info = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
        if (info->Size < header || info->Size > bytes - offset) break;
        if (info->Type == CpuSetInformation && info->Size >= sizeof(SYSTEM_CPU_SET_INFORMATION) &&
            (!info->CpuSet.Allocated || info->CpuSet.AllocatedToTargetProcess)) result.push_back(*info);
        offset += info->Size;
    }
    return result;
}

}  // namespace basalt::platform::detail
