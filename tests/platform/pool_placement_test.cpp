#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/platform/cpu.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include "affinity_win.hpp"
#else
#include "affinity_linux.hpp"
#endif

namespace {
int failures = 0;
void check(bool ok, const char* text) {
    std::printf("placement %-56s %s\n", text, ok ? "ok" : "FAILED");
    failures += !ok;
}

std::string placement() {
#if defined(_WIN32)
    GROUP_AFFINITY group{};
    if (!GetThreadGroupAffinity(GetCurrentThread(), &group)) return "query failed";
    std::string result = std::to_string(group.Group) + ":" + std::to_string(group.Mask);
    std::vector<unsigned long> ids;
    if (!basalt::platform::detail::get_thread_cpu_sets(ids)) return "query failed";
    for (auto id : ids) result += ":" + std::to_string(id);
    return result;
#else
    std::vector<unsigned long> mask;
    std::vector<int> cpus;
    if (basalt::platform::detail::get_thread_affinity(mask, &cpus) != 0) return "query failed";
    std::string result;
    for (int cpu : cpus) result += ":" + std::to_string(cpu);
    return result;
#endif
}

int hard_cpu() {
#if defined(_WIN32)
    GROUP_AFFINITY group{};
    if (!GetThreadGroupAffinity(GetCurrentThread(), &group)) return -1;
    const auto mask = (unsigned long long) group.Mask;
    if (!mask || (mask & (mask - 1))) return -1;
    for (int i = 0; i < 64; ++i)
        if (mask & (1ull << i)) return group.Group * 64 + i;
    return -1;
#else
    std::vector<unsigned long> mask;
    std::vector<int> cpus;
    if (basalt::platform::detail::get_thread_affinity(mask, &cpus) != 0 || cpus.size() != 1) return -1;
    return cpus.front();
#endif
}

bool host_on_cpu(int cpu) {
#if defined(_WIN32)
    std::vector<unsigned long> selected;
    if (!basalt::platform::detail::get_thread_cpu_sets(selected) || selected.size() != 1) return false;
    for (const auto& info : basalt::platform::detail::cpu_sets())
        if (info.CpuSet.Id == selected.front())
            return (int) info.CpuSet.Group * 64 + info.CpuSet.LogicalProcessorIndex == cpu;
    return false;
#else
    return hard_cpu() == cpu;
#endif
}

struct Observations {
    std::vector<int> cpus;
    std::vector<std::string> masks;
    std::atomic<int> done{0};
};
void observe(void* context, int worker, int, const std::atomic<bool>& stop) {
    auto& o = *static_cast<Observations*>(context);
    o.cpus[(size_t) worker] = hard_cpu();
    o.masks[(size_t) worker] = placement();
    o.done.fetch_add(1, std::memory_order_release);
    while (!stop.load()) std::this_thread::yield();
}
void check_pool(const std::vector<int>& cpus, const std::string& original) {
    constexpr int workers = 3;
    Observations o;
    o.cpus.resize(workers, -2);
    o.masks.resize(workers);
    basalt::kernels::cpu::ExpertPool pool(workers, basalt::platform::CpuAffinity::All, cpus);
    check(pool.placement_ok(), "worker placement accepted");
    pool.idle_start(observe, &o);
    const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (o.done.load(std::memory_order_acquire) < workers && std::chrono::steady_clock::now() < limit)
        std::this_thread::yield();
    pool.idle_stop();
    check(o.done.load() == workers, "every worker reports its native placement");
    for (int i = 0; i < workers; ++i) {
        if (cpus.empty()) check(o.masks[(size_t) i] == original, "default keeps inherited worker placement");
        else {
            const size_t first = cpus.size() > 1 ? 1 : 0;
            check(o.cpus[(size_t) i] == cpus[first + (size_t) i % (cpus.size() - first)],
                  "worker follows ordered logical CPUs, including reuse");
        }
    }
}

void check_restricted_physical(const std::vector<int>& available) {
#if defined(_WIN32)
    if (GetActiveProcessorGroupCount() != 1) return;
    const auto physical = basalt::platform::detect_cpu_topology(false).worker_cores;
    for (int cpu : available) {
        if (std::find(physical.begin(), physical.end(), cpu) != physical.end()) continue;
        DWORD_PTR original = 0, system = 0;
        if (!GetProcessAffinityMask(GetCurrentProcess(), &original, &system)) return;
        if (!SetProcessAffinityMask(GetCurrentProcess(), DWORD_PTR(1) << cpu)) return;
        std::vector<int> selected;
        std::string err;
        const bool parsed = basalt::platform::parse_cpu_list("physical", selected, err);
        const bool restored = SetProcessAffinityMask(GetCurrentProcess(), original) != 0;
        check(restored, "process affinity is restored after restricted probe");
        check(parsed && selected == std::vector<int>{cpu}, "physical keeps a permitted SMT sibling when its primary is excluded");
        return;
    }
#else
    (void) available;
#endif
}
}

int main() {
    using namespace basalt::platform;
    const auto available = logical_cpus();
    check(!available.empty(), "available logical CPUs include permitted processors");
    if (available.empty()) return 1;
    check_restricted_physical(available);
    const std::string original = placement();
    std::vector<int> cpus;
    std::string err;
    check(parse_cpu_list("none", cpus, err) && cpus.empty(), "none leaves placement unchanged");
    check(parse_cpu_list("physical", cpus, err) && !cpus.empty(), "physical selects one allowed CPU per physical core");
    const auto physical = cpus;
    for (const char* bad : {"", "-1", "0-", "1-0", "0,", "0,,1", "none,0", "2147483648", "0-2147483647"})
        check(!parse_cpu_list(bad, cpus, err), "invalid or oversized CPU list is refused");
    const int first = available.front(), last = available.back();
    const std::string duplicate = std::to_string(first) + "," + std::to_string(first);
    check(!parse_cpu_list(duplicate, cpus, err) && cpus.empty(), "duplicate CPU is refused without a partial result");
    std::vector<int> selected{last};
    if (first != last) selected.push_back(first);
    const std::string list = std::to_string(last) + (first != last ? "," + std::to_string(first) : "");
    check(parse_cpu_list(list, cpus, err) && cpus == selected, "explicit logical IDs preserve order");
    for (size_t i = 1; i < available.size(); ++i) {
        if (available[i] != available[i - 1] + 1) continue;
        const std::string range = std::to_string(available[i - 1]) + "-" + std::to_string(available[i]);
        check(parse_cpu_list(range, cpus, err) && cpus == std::vector<int>{available[i - 1], available[i]},
              "inclusive ranges accept adjacent logical processors");
        break;
    }
    for (int cpu : available) {
        if (std::find(physical.begin(), physical.end(), cpu) != physical.end()) continue;
        check(parse_cpu_list(std::to_string(cpu), cpus, err) && cpus == std::vector<int>{cpu},
              "explicit CPU list also accepts an SMT sibling");
        break;
    }
    check_pool({}, original);
    check_pool(selected, original);
    check_pool({last}, original);
    ThreadPlacement previous;
    check(pin_current_thread(last, previous, err), "host placement is applied");
    check(host_on_cpu(last), "host uses the selected logical CPU");
    std::string helper;
    std::thread background([&] {
        std::string error;
        if (restore_thread_affinity(previous, error)) helper = placement();
    });
    background.join();
    check(helper == original, "helper recovers the placement before the host pin");
    check(restore_thread_affinity(previous, err) && placement() == original, "complete host placement is restored");
    check(restore_thread_affinity({}, err) && placement() == original, "empty restoration is a no-op");
    return failures ? 1 : 0;
}
