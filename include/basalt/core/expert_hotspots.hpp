#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace basalt::core {

// Counts evaluated target-model routes, including rejected verify rows and re-read prompt rows.
// Controls run on the input reader; recording uses router IDs already available on the inference thread.
class ExpertHotspots {
public:
    void record(int layer, const int32_t* ids, int count, bool prefill) {
        if (!on_.load(std::memory_order_relaxed)) return;
        std::lock_guard<std::mutex> lock(mu_);
        if (!on_.load(std::memory_order_relaxed) || layer < 0 || layer >= layers_) return;
        auto& dest = prefill ? prefill_ : decode_;
        for (int i = 0; i < count; ++i)
            if (ids[i] >= 0 && ids[i] < experts_) ++dest[(size_t) layer * experts_ + ids[i]];
    }

    std::string command(const std::string& action, int layers, int experts) {
        std::vector<uint64_t> pp, tg;
        bool recording;
        uint64_t epoch;
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (action != "status" && action != "start" && action != "pause" && action != "reset")
                return "{\"error\":\"unknown hotspot action\"}";
            layers_ = layers;
            experts_ = experts;
            if (action == "reset") {
                on_.store(false, std::memory_order_relaxed);
                std::vector<uint64_t>().swap(prefill_);
                std::vector<uint64_t>().swap(decode_);
                ++epoch_;
            }
            if (action == "start" && prefill_.empty()) {
                prefill_.assign((size_t) layers * experts, 0);
                decode_.assign(prefill_.size(), 0);
                ++epoch_;
            }
            if (action == "start" || action == "pause") on_.store(action == "start", std::memory_order_relaxed);
            recording = on_.load(std::memory_order_relaxed);
            epoch = epoch_;
            pp = prefill_;
            tg = decode_;
        }
        std::ostringstream out;
        out << "{\"supported\":true,\"recording\":" << (recording ? "true" : "false")
            << ",\"epoch\":" << epoch << ",\"layers\":" << layers << ",\"experts\":" << experts;
        const auto array = [&](const char* key, const std::vector<uint64_t>& values) {
            out << ",\"" << key << "\":[";
            for (size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
            out << ']';
        };
        array("prefill", pp);
        array("decode", tg);
        out << '}';
        return out.str();
    }

    uint64_t bytes() const {
        std::lock_guard<std::mutex> lock(mu_);
        return (prefill_.capacity() + decode_.capacity()) * sizeof(uint64_t);
    }

private:
    mutable std::mutex mu_;
    std::atomic<bool> on_{false};
    int layers_ = 0, experts_ = 0;
    uint64_t epoch_ = 0;
    std::vector<uint64_t> prefill_, decode_;
};

}  // namespace basalt::core
