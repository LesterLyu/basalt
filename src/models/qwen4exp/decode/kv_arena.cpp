// src/models/qwen4exp/decode/kv_arena.cpp - the elastic KV pools.  Read the header first.
#include "basalt/models/qwen4exp/decode/kv_arena.hpp"
#include "basalt/core/vmm.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <utility>

namespace basalt::core {

KvArena::~KvArena() { close(); }

bool KvArena::open(const std::vector<uint64_t>& page_bytes, int64_t pages, int64_t page_cells, std::string& err,
                   const std::vector<uint64_t>& tail_bytes, bool movable) {
    close();
    if (pages <= 0 || page_cells <= 0 || page_bytes.empty()) {
        err = "KvArena: empty layout";
        return false;
    }
    if (!movable && (!vmm::supported() || (gran_ = vmm::granularity()) == 0)) {
        err = "KvArena: the elastic KV needs the driver's virtual memory management, which this GPU or driver does "
              "not offer";
        return false;
    }
    movable_ = movable;
    if (movable_) gran_ = 256;   // alignment between arrays of the same allocation
    pages_ = pages;
    page_cells_ = page_cells;
    for (size_t i = 0; i < page_bytes.size(); ++i) {
        const uint64_t pb = page_bytes[i];
        Array a;
        a.page_bytes = pb;
        a.tail_bytes = i < tail_bytes.size() ? tail_bytes[i] : 0;
        a.reserved = (pb * (uint64_t) pages + a.tail_bytes + gran_ - 1) / gran_ * gran_;
        if (!movable_) a.base = vmm::reserve(a.reserved);
        if (!movable_ && a.base == nullptr) {
            err = "KvArena: cannot reserve the address range of a KV pool";
            close();
            return false;
        }
        arrays_.push_back(a);
    }
    return true;
}

void KvArena::set_contents(size_t i, std::vector<uint8_t> bytes) {
    if (i < arrays_.size()) arrays_[i].contents = std::move(bytes);
}

void KvArena::close() {
    if (movable_) {
        if (packed_) cudaFree(packed_);
    } else {
        if (!arrays_.empty()) cudaDeviceSynchronize();
        for (Array& a : arrays_) {
            for (const Extent& x : a.extents) vmm::unmap(a.base + x.off, x.bytes, x.handle);
            vmm::release_range(a.base, a.reserved);
        }
    }
    packed_ = nullptr;
    bind_ = {};
    movable_ = false;
    arrays_.clear();
    cells_ = 0;
    pages_ = 0;
}

uint64_t KvArena::packed_bytes(int64_t cells) const {
    uint64_t bytes = 0;
    for (const Array& a : arrays_) bytes += need(a, cells);
    constexpr uint64_t block = 2ull << 20;
    return (bytes + block - 1) / block * block;
}

bool KvArena::resize(int64_t cells, std::string& err) {
    const int64_t step = std::max(extent_cells_, page_cells_);
    cells = std::min(max_cells(), (std::max<int64_t>(cells, 0) + step - 1) / step * step);
    if (cells == cells_) return true;
    const uint64_t bytes = packed_bytes(cells);
    uint8_t* next = nullptr;
    auto fail = [&](cudaError_t code) {
        if (next) cudaFree(next);
        (void) cudaGetLastError();
        err = std::string("resizing the KV pools: ") + cudaGetErrorString(code);
        return false;
    };
    if (bytes != 0) {
        cudaError_t rc = cudaMalloc((void**) &next, (size_t) bytes);
        if (rc != cudaSuccess) return fail(rc);
        for (uint64_t off = 0; const Array& a : arrays_) {
            const uint64_t want = need(a, cells), kept = std::min(want, need(a, cells_));
            if (kept && (rc = cudaMemcpy(next + off, a.base, (size_t) kept, cudaMemcpyDeviceToDevice)) != cudaSuccess)
                return fail(rc);
            if (want > kept && (rc = cudaMemset(next + off + kept, 0, (size_t) (want - kept))) != cudaSuccess)
                return fail(rc);
            const uint64_t have = kept < a.contents.size() ? std::min<uint64_t>(want - kept, a.contents.size() - kept) : 0;
            if (have && (rc = cudaMemcpy(next + off + kept, a.contents.data() + kept, (size_t) have,
                                         cudaMemcpyHostToDevice)) != cudaSuccess)
                return fail(rc);
            off += want;
        }
        if ((rc = cudaDeviceSynchronize()) != cudaSuccess) return fail(rc);
    }
    if (packed_) {
        const cudaError_t rc = cudaFree(packed_);
        if (rc != cudaSuccess) return fail(rc);
    }
    packed_ = next;
    cells_ = cells;
    for (uint64_t off = 0; Array& a : arrays_) {
        a.base = next ? next + off : nullptr;
        off += need(a, cells);
    }
    if (bind_) bind_();
    return true;
}

uint64_t KvArena::need(const Array& a, int64_t cells) const {
    const int64_t pages = std::min<int64_t>(pages_, (std::max<int64_t>(cells, 0) + page_cells_ - 1) / page_cells_);
    if (pages == 0) return 0;
    return std::min(a.reserved, ((uint64_t) pages * a.page_bytes + a.tail_bytes + gran_ - 1) / gran_ * gran_);
}

void KvArena::recount() {
    int64_t pages = pages_;
    for (const Array& a : arrays_)
        pages = std::min<int64_t>(pages, (int64_t) ((a.mapped() > a.tail_bytes ? a.mapped() - a.tail_bytes : 0) / a.page_bytes));
    cells_ = pages * page_cells_;
}

uint64_t KvArena::mapped_bytes() const {
    if (movable_) return packed_bytes(cells_);
    uint64_t b = 0;
    for (const Array& a : arrays_) b += a.mapped();
    return b;
}

uint64_t KvArena::bytes_for(int64_t cells) const {
    if (movable_) {
        const int64_t step = std::max(extent_cells_, page_cells_);
        return packed_bytes(std::min(max_cells(), (std::max<int64_t>(cells, 0) + step - 1) / step * step));
    }
    uint64_t b = 0;
    for (const Array& a : arrays_) b += need(a, cells);
    return b;
}

uint64_t KvArena::grow_bytes(int64_t cells) const {
    if (movable_) {
        const uint64_t want = bytes_for(cells), have = mapped_bytes();
        return want > have ? want - have : 0;
    }
    uint64_t b = 0;
    for (const Array& a : arrays_) b += std::max(need(a, cells), a.mapped()) - a.mapped();
    return b;
}

bool KvArena::map(int64_t cells, std::string& err) {
    if (movable_) return cells <= cells_ || resize(cells, err);
    bool any = false;
    for (Array& a : arrays_) {
        const uint64_t want = need(a, cells);
        for (uint64_t at = a.mapped(); at < want; at = a.mapped()) {
            // this mapping ends at the first grid line past `at`; lines finer than a page are a page apart
            uint64_t end = want;
            const int64_t step = extent_cells_ > 0 ? std::max(extent_cells_, page_cells_) : 0;
            for (int64_t c = step; c > 0 && c < max_cells(); c += step)
                if (need(a, c) > at) { end = std::min(end, need(a, c)); break; }
            const unsigned long long h = vmm::map(a.base + at, end - at);
            if (h == 0) {
                (void) cudaGetLastError();
                recount();
                err = "the driver has no VRAM for the KV pools";
                return false;
            }
            a.extents.push_back({at, end - at, h});
            const uint64_t have = at < a.contents.size() ? std::min<uint64_t>(end - at, a.contents.size() - at) : 0;
            if (cudaMemset(a.base + at, 0, (size_t) (end - at)) != cudaSuccess ||
                (have > 0 && cudaMemcpy(a.base + at, a.contents.data() + at, (size_t) have, cudaMemcpyHostToDevice) != cudaSuccess)) {
                recount();
                err = std::string("zeroing new KV pages failed: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            any = true;
        }
    }
    recount();
    // the zeroing is ordered before work on any stream that follows
    if (any && cudaDeviceSynchronize() != cudaSuccess) {
        err = std::string("zeroing new KV pages failed: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
}

bool KvArena::unmap(int64_t cells, std::string& err) {
    if (movable_) return cells >= cells_ || resize(cells, err);
    bool synced = false;
    for (Array& a : arrays_) {
        const uint64_t want = need(a, cells);
        while (!a.extents.empty() && a.extents.back().off >= want) {
            if (!synced) {
                if (cudaDeviceSynchronize() != cudaSuccess) {
                    err = std::string("the device failed before the KV pools shrank: ") +
                          cudaGetErrorString(cudaGetLastError());
                    return false;
                }
                synced = true;
            }
            const Extent x = a.extents.back();
            if (!vmm::unmap(a.base + x.off, x.bytes, x.handle)) {
                recount();
                err = "the driver refused to release a KV pool mapping";
                return false;
            }
            a.extents.pop_back();
        }
    }
    recount();
    return true;
}

}  // namespace basalt::core
