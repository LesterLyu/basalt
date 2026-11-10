#include "basalt/core/expert_hotspots.hpp"

#include <cstdio>
#include <thread>

int main() {
    using basalt::core::ExpertHotspots;
    int failures = 0;
    const auto check = [&](bool ok) { if (!ok) ++failures; };
    ExpertHotspots h;
    int32_t ids[] = {1, 1, 2, -1, 4};
    h.record(0, ids, 5, false);
    check(h.bytes() == 0);
    check(h.command("status", 2, 4).find("\"decode\":[]") != std::string::npos);
    check(h.bytes() == 0);
    h.command("start", 2, 4);
    h.record(0, ids, 5, false);
    h.record(1, ids, 5, true);
    auto s = h.command("pause", 2, 4);
    check(s.find("\"decode\":[0,2,1,0,0,0,0,0]") != std::string::npos);
    check(s.find("\"prefill\":[0,0,0,0,0,2,1,0]") != std::string::npos);
    h.record(0, ids, 5, false);
    check(h.command("status", 2, 4) == s);
    h.command("start", 2, 4);
    h.record(0, ids, 5, false);
    check(h.command("status", 2, 4).find("\"decode\":[0,4,2,0,0,0,0,0]") != std::string::npos);
    h.command("reset", 2, 4);
    check(h.bytes() == 0);
    check(h.command("status", 2, 4).find("\"recording\":false") != std::string::npos);
    h.command("start", 2, 4);
    int32_t one = 3;
    auto work = [&] { for (int i = 0; i < 10000; ++i) h.record(1, &one, 1, false); };
    std::thread a(work), b(work);
    for (int i = 0; i < 100; ++i) h.command("status", 2, 4);
    a.join(); b.join();
    check(h.command("pause", 2, 4).find("\"decode\":[0,0,0,0,0,0,0,20000]") != std::string::npos);
    check(h.command("bad", 2, 4).find("\"error\"") != std::string::npos);
    std::printf("expert_hotspots_test: %d failures\n", failures);
    return failures ? 1 : 0;
}
