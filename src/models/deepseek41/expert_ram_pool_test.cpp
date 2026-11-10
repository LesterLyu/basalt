#include "models/deepseek41/expert_ram_pool.hpp"

#include <cstdio>
#include <latch>
#include <limits>
#include <mutex>
#include <thread>

using basalt::models::deepseek41::ExpertRamPool;
namespace {
int failures = 0;
#define CHECK(expr) do { if (!(expr)) { std::printf("FAIL line %d: %s\n", __LINE__, #expr); ++failures; } } while (0)

// Fake backing checks pool ownership and arithmetic without allocating or pinning payloads.
struct Block {
    uint64_t bytes, allocation, pinned;
    bool ready = true;
    uint64_t allocated() const { return allocation; }
    uint64_t registered() const { return pinned; }
};
using Pool = ExpertRamPool<Block>;
constexpr uint64_t mib = 1ull << 20;
auto block(uint64_t bytes) { return std::make_unique<Block>(Block{bytes,bytes,bytes}); }

void capacity() {
    Pool pool;
    auto first = block(Pool::capacity - 8*mib), last = block(8*mib);
    CHECK(pool.put(first));
    CHECK(!first);
    CHECK(pool.put(last));
    CHECK(pool.bytes() == Pool::capacity);
    auto excess = block(1);
    CHECK(!pool.put(excess));
    CHECK(excess != nullptr);
    CHECK(pool.bytes() == Pool::capacity);
    auto huge = block(std::numeric_limits<uint64_t>::max());
    CHECK(!pool.put(huge));
    CHECK(huge != nullptr);
    auto taken = pool.take(8*mib);
    CHECK(taken && taken->allocated() == 8*mib && !taken->ready);
    CHECK(pool.bytes() == Pool::capacity - 8*mib);
    CHECK(pool.put(excess));
    CHECK(pool.bytes() == Pool::capacity - 8*mib + 1);
    pool.clear();
    CHECK(pool.empty() && pool.bytes() == 0);
}

void sizes_and_registration() {
    Pool pool;
    for (auto size : {6*mib,8*mib,10*mib}) { auto p = block(size); CHECK(pool.put(p)); }
    CHECK(pool.usable(12*mib) == 0);
    auto replacement = block(12*mib);
    CHECK(pool.put(replacement));
    CHECK(pool.usable(12*mib) == 1);
    auto picked = pool.take(7*mib);
    CHECK(picked && picked->allocated() == 8*mib);
    CHECK(pool.usable(12*mib) == 1);
    picked = pool.take(12*mib);
    CHECK(picked && picked->allocated() == 12*mib);
    CHECK(pool.usable(12*mib) == 0);
    CHECK(!pool.take(12*mib));
    CHECK(pool.bytes() == 16*mib);
    pool.trim(10*mib);
    CHECK(pool.bytes() <= 10*mib);

    auto unpinned = block(8*mib);
    unpinned->pinned = 4*mib;
    CHECK(!pool.put(unpinned));
    CHECK(unpinned != nullptr);
    auto padded = block(6*mib);
    padded->allocation = 8*mib;
    CHECK(pool.put(padded));
    CHECK(pool.usable(7*mib) == 0);
    CHECK(!pool.take(7*mib));
    CHECK(pool.take(6*mib) != nullptr);
}

void in_flight_refill() {
    Pool pool;
    std::mutex mutex;
    std::latch allocation_started(1), foreground_finished(1);
    bool admitted = true, retained = false;
    std::thread refill([&] {
        auto pending = block(8*mib);
        allocation_started.count_down();
        foreground_finished.wait();
        {
            std::lock_guard lock(mutex);
            admitted = pool.put(pending);
        }
        retained = pending != nullptr;
    });
    allocation_started.wait();
    {
        std::lock_guard lock(mutex);
        auto full = block(Pool::capacity);
        CHECK(pool.put(full));
    }
    foreground_finished.count_down();
    refill.join();
    CHECK(!admitted && retained);
    CHECK(pool.bytes() == Pool::capacity);
    auto another = block(8*mib);
    CHECK(!pool.put(another));
    CHECK(another != nullptr);
    CHECK(pool.bytes() == Pool::capacity);
    CHECK(pool.take(Pool::capacity) != nullptr);
    CHECK(pool.put(another));
    CHECK(pool.bytes() == 8*mib);
}
}  // namespace

int main() {
    capacity();
    sizes_and_registration();
    in_flight_refill();
    std::printf("expert RAM pool: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
