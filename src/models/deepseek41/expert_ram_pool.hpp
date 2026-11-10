#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace basalt::models::deepseek41 {

// The caller holds the pool mutex. Rejected blocks stay with the caller for release outside the lock.
template <class Block> class ExpertRamPool {
public:
    static constexpr uint64_t capacity = 1ull << 30;

    uint64_t bytes() const { return bytes_; }
    bool empty() const { return blocks_.empty(); }

    std::size_t usable(uint64_t bytes) const {
        std::size_t n = 0;
        for (const auto& block : blocks_)
            if (block->allocated() >= bytes && block->registered() >= bytes) ++n;
        return n;
    }

    std::unique_ptr<Block> take(uint64_t bytes) {
        std::size_t best = blocks_.size();
        for (std::size_t i = 0; i < blocks_.size(); ++i)
            if (blocks_[i]->allocated() >= bytes && blocks_[i]->registered() >= bytes &&
                (best == blocks_.size() || blocks_[i]->allocated() < blocks_[best]->allocated())) best = i;
        if (best == blocks_.size()) return nullptr;
        auto block = std::move(blocks_[best]);
        blocks_[best] = std::move(blocks_.back());
        blocks_.pop_back();
        bytes_ -= block->allocated();
        return block;
    }

    bool put(std::unique_ptr<Block>& block) {
        if (!block || block->registered() < block->bytes || bytes_ > capacity ||
            block->allocated() > capacity - bytes_) return false;
        block->ready = false;
        const uint64_t bytes = block->allocated();
        blocks_.push_back(std::move(block));
        bytes_ += bytes;
        return true;
    }

    void trim(uint64_t limit) {
        while (bytes_ > limit && !blocks_.empty()) {
            bytes_ -= blocks_.back()->allocated();
            blocks_.pop_back();
        }
    }

    void clear() { blocks_.clear(); bytes_ = 0; }

private:
    std::vector<std::unique_ptr<Block>> blocks_;
    uint64_t bytes_ = 0;
};

}  // namespace basalt::models::deepseek41
