// Optional host-side attribution for expert transfer stalls. Nested scopes are not additive.
#pragma once

#include "basalt/log.hpp"

#include <array>
#include <chrono>
#include <cstdint>

namespace basalt::models::deepseek41 {

struct TierProfile {
    enum Part {
        Acquire, Recycle, Claim, Pump, DrainWait, Publish, CpuWait, PeerSubmit, RouteWait, Boundary,
        RamPoolHit, RamPoolMiss, RamPoolFree, Routed, Fetch, CpuStart, PeerLaunch, PeerFinish, GpuSubmit, Count
    };
    bool enabled = false;
    std::array<uint64_t, Count> ns{}, calls{};

    void report(const char* tag = "total") const {
        if (!enabled) return;
        static constexpr const char* names[] = {
            "ram_acquire", "ram_recycle", "claim", "pump", "copy_wait", "publish", "cpu_wait", "peer_submit",
            "route_wait", "window_boundary", "ram_pool_hit", "ram_pool_miss", "ram_pool_free", "routed", "fetch",
            "cpu_start", "peer_launch", "peer_finish", "gpu_submit"
        };
        static_assert(std::size(names) == Count);
        for (int i = 0; i < Count; ++i)
            basalt::log::info("deepseek41 tier timing", "%s %s %.3f ms, %llu calls", tag, names[i], ns[i] / 1e6,
                              (unsigned long long) calls[i]);
    }
};

struct TierScope {
    TierProfile* p;
    TierProfile::Part part;
    std::chrono::steady_clock::time_point start;

    TierScope(TierProfile& v, TierProfile::Part k) : p(v.enabled ? &v : nullptr), part(k) {
        if (p) start = std::chrono::steady_clock::now();
    }
    TierScope(const TierScope&) = delete;
    TierScope& operator=(const TierScope&) = delete;
    ~TierScope() { stop(); }

    void stop() {
        if (!p) return;
        p->ns[part] += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count();
        ++p->calls[part];
        p = nullptr;
    }
};

}  // namespace basalt::models::deepseek41
