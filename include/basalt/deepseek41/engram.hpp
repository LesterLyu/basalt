#pragma once
#include "basalt/deepseek41/weights.hpp"
#include <array>

namespace basalt::deepseek41 {

class EngramHasher {
public:
    struct History { std::array<uint32_t,3> previous{}; };
    using Rows = std::array<uint32_t,48>;
    explicit EngramHasher(const GgufFile& metadata);
    EngramHasher(const GgufFile& metadata, std::vector<uint32_t> token_map);
    History initial_history() const { return {{pad_,pad_,pad_}}; }
    // The history is a value checkpoint: discard speculative tokens by restoring the saved value.
    Rows advance(int32_t token, History& history) const;
    uint64_t table_rows() const { return rows_; }
    uint32_t mapped_pad() const { return pad_; }
    size_t vocabulary() const { return token_map_.size(); }
private:
    std::vector<uint32_t> token_map_;
    std::array<uint64_t,8> multipliers_{};
    std::array<uint64_t,48> offsets_{};
    std::array<uint32_t,48> moduli_{};
    uint32_t pad_ = 0;
    uint64_t rows_ = 0;
};

}  // namespace basalt::deepseek41
