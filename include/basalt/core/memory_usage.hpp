// Allocation ownership for serving telemetry; virtual address reservations are not physical bytes.
#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <sstream>
#include <string>

namespace basalt::core {

struct MemoryUsage {
    using Parts = std::map<std::string, uint64_t>;
    std::map<int, Parts> gpus;
    Parts host;
    std::set<const void*> owners;

    bool claim(const void* owner) { return owner && owners.insert(owner).second; }
    void device(int ordinal, const char* kind, uint64_t bytes) { gpus[ordinal][kind] += bytes; }
    std::string json() const {
        const auto parts = [](const Parts& p) {
            std::ostringstream s;
            s << '{';
            bool first = true;
            for (const auto& [key, n] : p) {
                if (!first) s << ',';
                first = false;
                s << '"' << key << "\":" << n;
            }
            return s.str() + '}';
        };
        std::ostringstream s;
        s << "{\"gpus\":{";
        bool first = true;
        for (const auto& [ordinal, p] : gpus) {
            if (!first) s << ',';
            first = false;
            s << '"' << ordinal << "\":" << parts(p);
        }
        s << "},\"host\":" << parts(host) << '}';
        return s.str();
    }
};

}  // namespace basalt::core
