// Manual RAM/device transfer measurements. Validation and allocation are outside the timed interval.
#include "basalt/core/vmm.hpp"
#include "basalt/platform/dynlib.hpp"
#include "basalt/platform/memory.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
constexpr size_t MiB = 1ull << 20;
constexpr size_t GiB = 1ull << 30;
size_t measured_rows = 0;
double seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }
void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void driver_check(CUresult e) {
    if (e == CUDA_SUCCESS) return;
    const char* s = nullptr;
    cuGetErrorString(e, &s);
    throw std::runtime_error(s ? s : "CUDA driver error");
}
struct Unsupported : std::runtime_error { using std::runtime_error::runtime_error; };

std::vector<std::string> split(std::string_view s) {
    std::vector<std::string> out;
    while (!s.empty()) {
        const auto pos = s.find(',');
        out.emplace_back(s.substr(0, pos));
        if (out.back().empty()) throw std::runtime_error("empty list item");
        if (pos == s.npos) return out;
        s.remove_prefix(pos + 1);
        if (s.empty()) throw std::runtime_error("empty list item");
    }
    throw std::runtime_error("empty list");
}
size_t number(std::string_view s) {
    size_t n = 0;
    auto r = std::from_chars(s.data(), s.data() + s.size(), n);
    if (s.empty() || r.ec != std::errc{} || r.ptr != s.data() + s.size())
        throw std::runtime_error("invalid unsigned integer: " + std::string(s));
    return n;
}
size_t scaled(size_t n, size_t scale) {
    if (n > std::numeric_limits<size_t>::max() / scale) throw std::runtime_error("size overflow");
    return n * scale;
}
size_t byte_size(std::string s) {
    size_t scale = 1;
    if (!s.empty()) {
        const char c = s.back();
        if (c == 'K' || c == 'M' || c == 'G') {
            scale = c == 'K' ? 1024 : c == 'M' ? MiB : GiB;
            s.pop_back();
        }
    }
    const size_t n = scaled(number(s), scale);
    if (!n) throw std::runtime_error("copy sizes must be positive");
    return n;
}
bool has(const std::vector<std::string>& v, const std::string& s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}
struct Options {
    std::vector<int> gpus;
    std::vector<std::string> modes = {"pageable", "registered", "pinned", "portable", "vmm-primary", "vmm-all"};
    std::vector<std::string> directions = {"h2d", "d2h", "d2d"};
    std::vector<size_t> sizes = {MiB, 8 * MiB};
    size_t host_bytes = 0;
    int reps = 32, warmup = 4, rounds = 2, gate_ms = 0;
    bool concurrent = true;
    std::string peer = "both";
};
bool parse(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") {
            std::puts("transfer_bench [options]\n"
                "  --gpus 0,1                 Default: all visible GPUs; first GPU owns host allocations\n"
                "  --host-modes LIST          pageable,registered,pinned,portable,vmm-primary,vmm-all\n"
                "  --sizes 1M,8M              Bytes, or binary K/M/G suffix\n"
                "  --host-mib N               Host allocation capacity (default: at least 64 MiB)\n"
                "  --reps N --warmup N        Copies per row and untimed warmup copies (32 / 4)\n"
                "  --rounds N                 Reverse host-method order on alternate rounds (2)\n"
                "  --directions h2d,d2h,d2d   Default: all\n"
                "  --peer-access off|on|both  Peer API with access disabled/enabled (both)\n"
                "  --no-concurrent            Omit simultaneous RAM transfers on all selected GPUs\n"
                "  --gate-ms N                Optional queued host callback delay, serial rows only (0)\n"
                "Markdown on stdout; errors on stderr. Exit 77: no devices; 1: allocation/copy/validation failure.\n"
                "No model required. This is a manual microbench, not an inference benchmark.");
            return false;
        }
        if (key == "--no-concurrent") { o.concurrent = false; continue; }
        if (i + 1 == argc) throw std::runtime_error("missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--gpus") {
            o.gpus.clear();
            for (const auto& x : split(value)) {
                const auto n = number(x);
                if (n > 1024) throw std::runtime_error("GPU index too large");
                const int id = static_cast<int>(n);
                if (std::find(o.gpus.begin(), o.gpus.end(), id) != o.gpus.end())
                    throw std::runtime_error("duplicate GPU index");
                o.gpus.push_back(id);
            }
        } else if (key == "--host-modes" || key == "--directions") {
            auto v = split(value);
            const auto allowed = key == "--host-modes" ? Options{}.modes : Options{}.directions;
            for (const auto& x : v) if (!has(allowed, x)) throw std::runtime_error("unknown list item: " + x);
            (key == "--host-modes" ? o.modes : o.directions) = std::move(v);
        } else if (key == "--sizes") {
            o.sizes.clear();
            for (const auto& x : split(value)) o.sizes.push_back(byte_size(x));
        } else if (key == "--host-mib") o.host_bytes = scaled(number(value), MiB);
        else if (key == "--peer-access") {
            if (value != "off" && value != "on" && value != "both") throw std::runtime_error("invalid peer-access mode");
            o.peer = value;
        } else {
            const auto n = number(value);
            if (n > 1000000) throw std::runtime_error("iteration/delay value too large");
            const int k = static_cast<int>(n);
            if (key == "--reps") o.reps = k;
            else if (key == "--warmup") o.warmup = k;
            else if (key == "--rounds") o.rounds = k;
            else if (key == "--gate-ms") o.gate_ms = k;
            else throw std::runtime_error("unknown option: " + key);
        }
    }
    if (!o.reps || !o.rounds || o.gate_ms > 10000) throw std::runtime_error("reps/rounds must be positive; gate <= 10000 ms");
    return true;
}

struct Host {
    unsigned char* p = nullptr;
    uint64_t mapped = 0;
    bool registered = false;
    int owner = 0;
    std::string mode;
    Host() = default;
    Host(const Host&) = delete;
    Host& operator=(const Host&) = delete;
    void init(const std::string& kind, size_t bytes, const std::vector<int>& gpus) {
        mode = kind;
        owner = gpus.front();
        check(cudaSetDevice(owner));
        const uint64_t avail = basalt::platform::available_memory();
        const uint64_t floor = std::max<uint64_t>(512 * MiB, avail / 20);
        if (avail && (avail <= floor || bytes > avail - floor))
            throw std::runtime_error("host allocation would exhaust available RAM");
        if (mode == "vmm-primary" || mode == "vmm-all") {
            if (!basalt::core::vmm::host_granularity()) throw Unsupported("host NUMA VMM not supported");
            p = basalt::core::vmm::alloc_host(bytes, mapped, basalt::core::vmm::HostBacking::Bounded);
            if (!p) throw std::runtime_error("host VMM allocation failed");
            if (mode == "vmm-all") {
                for (int gpu : gpus) {
                    CUmemAccessDesc access{};
                    access.location = {CU_MEM_LOCATION_TYPE_DEVICE, gpu};
                    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    driver_check(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(p), mapped, &access, 1));
                }
            }
        } else if (mode == "pageable" || mode == "registered") {
            p = static_cast<unsigned char*>(basalt::platform::alloc_aligned(bytes, 65536));
            if (!p) throw std::bad_alloc();
            if (mode == "registered") {
                check(cudaHostRegister(p, bytes, cudaHostRegisterPortable | cudaHostRegisterMapped));
                registered = true;
            }
        } else {
            const unsigned flags = mode == "portable" ? cudaHostAllocPortable | cudaHostAllocMapped : cudaHostAllocDefault;
            check(cudaHostAlloc(reinterpret_cast<void**>(&p), bytes, flags));
        }
        // Touch the entire requested allocation so a capacity test does not only fault its copy buffers.
        std::memset(p, 0, bytes);
    }
    ~Host() {
        if (!p) return;
        cudaSetDevice(owner);
        if (mapped) basalt::core::vmm::free_host(p, mapped, basalt::core::vmm::HostBacking::Bounded);
        else if (mode == "pageable" || mode == "registered") {
            if (registered) cudaHostUnregister(p);
            basalt::platform::free_aligned(p);
        } else cudaFreeHost(p);
    }
};

struct Gpu {
    int id;
    void* a = nullptr;
    void* b = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    explicit Gpu(int n) : id(n) {}
    Gpu(const Gpu&) = delete;
    Gpu& operator=(const Gpu&) = delete;
    void init(size_t bytes) {
        check(cudaSetDevice(id));
        check(cudaMalloc(&a, bytes));
        check(cudaMalloc(&b, bytes));
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cudaEventCreate(&begin));
        check(cudaEventCreate(&end));
    }
    ~Gpu() {
        cudaSetDevice(id);
        if (stream) cudaStreamSynchronize(stream);
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
        if (stream) cudaStreamDestroy(stream);
        if (a) cudaFree(a);
        if (b) cudaFree(b);
    }
};

// The callback never calls CUDA; its independent timer releases it even if an enqueue fails.
struct Gate {
    std::atomic<bool> release{false}, entered{false};
    std::thread timer;
    explicit Gate(cudaStream_t stream, int ms) {
        if (!ms) return;
        timer = std::thread([this, ms] {
            while (!entered.load(std::memory_order_acquire) && !release.load()) std::this_thread::yield();
            if (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(ms));
            release.store(true, std::memory_order_release);
        });
        const auto error = cudaLaunchHostFunc(stream, [](void* ptr) {
            auto& g = *static_cast<Gate*>(ptr);
            g.entered.store(true, std::memory_order_release);
            while (!g.release.load(std::memory_order_acquire)) std::this_thread::yield();
        }, this);
        if (error != cudaSuccess) { release = true; timer.join(); check(error); }
    }
    ~Gate() {
        release = true;
        if (timer.joinable()) timer.join();
    }
};
struct Timing { double submit_us = 0, stream_s = 0, wall_s = 0; bool pending = false; };
Timing measure(Gpu& g, const Options& o, const std::function<void()>& copy, int gate_ms) {
    check(cudaSetDevice(g.id));
    Timing t;
    Gate gate(g.stream, gate_ms);
    try {
        const auto wall = Clock::now();
        check(cudaEventRecord(g.begin, g.stream));
        const auto submit = Clock::now();
        for (int i = 0; i < o.reps; ++i) copy();
        t.submit_us = seconds(Clock::now() - submit) * 1e6 / o.reps;
        t.pending = gate_ms && !gate.release.load();
        check(cudaEventRecord(g.end, g.stream));
        check(cudaEventSynchronize(g.end));
        t.wall_s = seconds(Clock::now() - wall);
        float ms = 0;
        check(cudaEventElapsedTime(&ms, g.begin, g.end));
        t.stream_s = ms * .001;
    } catch (...) {
        gate.release = true;
        cudaStreamSynchronize(g.stream);  // callback state outlives the callback on an error too
        throw;
    }
    return t;
}
void pattern(unsigned char* p, size_t n, unsigned seed) {
    uint32_t x = seed | 1;
    for (size_t i = 0; i < n; ++i) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        p[i] = static_cast<unsigned char>(x);
    }
}
void verify(const unsigned char* p, size_t n, unsigned seed) {
    uint32_t x = seed | 1;
    for (size_t i = 0; i < n; ++i) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        if (p[i] != static_cast<unsigned char>(x))
            throw std::runtime_error("byte validation failed at offset " + std::to_string(i));
    }
}
void row(int round, const std::string& mode, const std::string& route, size_t n, const Timing& t,
         const Options& o, size_t links = 1) {
    ++measured_rows;
    const double bytes = static_cast<double>(n) * o.reps * static_cast<double>(links);
    std::printf("| %d | %s | %s | %.4f | %.2f | ", round, mode.c_str(), route.c_str(), n / double(MiB), t.submit_us);
    if (links == 1 && t.stream_s > 0) std::printf("%.2f", bytes / t.stream_s / 1e9);
    else std::printf("—");
    std::printf(" | %.2f | %s | exact |\n", bytes / t.wall_s / 1e9,
                links > 1 || !o.gate_ms ? "—" : t.pending ? "yes" : "no");
    std::fflush(stdout);
}

using Copy = std::function<void()>;
void warm(Gpu& g, const Options& o, const Copy& copy) {
    check(cudaSetDevice(g.id));
    for (int i = 0; i < o.warmup; ++i) copy();
    check(cudaStreamSynchronize(g.stream));
}
void readback(Gpu& g, unsigned char* verify_buffer, size_t n, unsigned seed) {
    check(cudaSetDevice(g.id));
    check(cudaMemcpy(verify_buffer, g.b, n, cudaMemcpyDeviceToHost));
    verify(verify_buffer, n, seed);
}
Timing simultaneous(const std::vector<std::unique_ptr<Gpu>>& gpus, const Options& o, const std::vector<Copy>& copies) {
    std::atomic<bool> go{false};
    std::atomic<size_t> ready{0};
    std::vector<Timing> times(gpus.size());
    std::vector<std::exception_ptr> errors(gpus.size());
    std::vector<std::thread> threads;
    try {
        for (size_t i = 0; i < gpus.size(); ++i) threads.emplace_back([&, i] {
            const auto device_error = cudaSetDevice(gpus[i]->id);
            ++ready;
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            try { check(device_error); times[i] = measure(*gpus[i], o, copies[i], 0); }
            catch (...) { errors[i] = std::current_exception(); }
        });
    } catch (...) {
        go = true;
        for (auto& t : threads) t.join();
        throw;
    }
    while (ready.load() != gpus.size()) std::this_thread::yield();
    const auto start = Clock::now();
    go.store(true, std::memory_order_release);
    for (auto& t : threads) t.join();
    Timing total;
    total.wall_s = seconds(Clock::now() - start);
    for (size_t i = 0; i < gpus.size(); ++i) {
        if (errors[i]) std::rethrow_exception(errors[i]);
        total.submit_us = std::max(total.submit_us, times[i].submit_us);
    }
    return total;
}

// Match NVML adapters by PCI id, since CUDA_VISIBLE_DEVICES can reorder CUDA ordinals.
class Power {
    basalt::platform::DynLib lib = basalt::platform::DynLib::open_nvml();
    int (*shutdown)() = nullptr;
    int (*handle)(const char*, void**) = nullptr;
    int (*limit)(void*, unsigned*) = nullptr;
    bool active = false;
public:
    Power() {
        const auto init = lib.sym<int (*)()>("nvmlInit_v2");
        shutdown = lib.sym<int (*)()>("nvmlShutdown");
        handle = lib.sym<int (*)(const char*, void**)>("nvmlDeviceGetHandleByPciBusId_v2");
        limit = lib.sym<int (*)(void*, unsigned*)>("nvmlDeviceGetEnforcedPowerLimit");
        active = init && shutdown && init() == 0;
    }
    ~Power() { if (active) shutdown(); }
    std::string watts(const char* bus) const {
        void* h = nullptr;
        unsigned mw = 0;
        if (!active || !handle || !limit || handle(bus, &h) || limit(h, &mw)) return "unknown";
        char text[32]{};
        std::snprintf(text, sizeof text, "%.1f", mw / 1000.0);
        return text;
    }
};

void host_rows(const Options& o, const std::vector<std::unique_ptr<Gpu>>& gpus, Host& host,
               unsigned char* verifier, size_t stride, int round, unsigned& seed, std::string& phase) {
    for (size_t n : o.sizes) for (const std::string direction : {"h2d", "d2h"}) {
        if (!has(o.directions, direction)) continue;
        const bool h2d = direction == "h2d";
        for (size_t i = 0; i < gpus.size(); ++i) {
            auto& g = *gpus[i];
            phase = host.mode + " " + direction + " GPU" + std::to_string(g.id) + " bytes=" + std::to_string(n);
            auto* p = host.p + (h2d ? 0 : (i + 1) * stride);
            const unsigned key = ++seed * 2 + 1;
            check(cudaSetDevice(g.id));
            if (h2d) pattern(p, n, key);
            else { pattern(verifier, n, key); check(cudaMemcpy(g.a, verifier, n, cudaMemcpyHostToDevice)); }
            Copy copy = [&g, p, n, h2d] {
                check(h2d ? cudaMemcpyAsync(g.b, p, n, cudaMemcpyHostToDevice, g.stream)
                          : cudaMemcpyAsync(p, g.a, n, cudaMemcpyDeviceToHost, g.stream));
            };
            warm(g, o, copy);
            if (h2d) {
                check(cudaMemsetAsync(g.b, 0xa5, n, g.stream));
                check(cudaStreamSynchronize(g.stream));
            }
            else std::memset(p, 0xa5, n);
            const auto t = measure(g, o, copy, o.gate_ms);
            if (h2d) readback(g, verifier, n, key); else verify(p, n, key);
            row(round, host.mode, direction + " GPU" + std::to_string(g.id), n, t, o);
        }
        if (!o.concurrent || gpus.size() < 2) continue;
        phase = host.mode + " " + direction + " concurrent bytes/GPU=" + std::to_string(n);
        const unsigned key = ++seed * 2 + 1;
        std::vector<Copy> copies;
        for (size_t i = 0; i < gpus.size(); ++i) {
            auto& g = *gpus[i];
            auto* p = host.p + (h2d ? 0 : (i + 1) * stride);
            check(cudaSetDevice(g.id));
            pattern(h2d ? p : verifier, n, key + static_cast<unsigned>(h2d ? 0 : i * 2));
            if (!h2d) check(cudaMemcpy(g.a, verifier, n, cudaMemcpyHostToDevice));
            copies.emplace_back([&g, p, n, h2d] {
                check(h2d ? cudaMemcpyAsync(g.b, p, n, cudaMemcpyHostToDevice, g.stream)
                          : cudaMemcpyAsync(p, g.a, n, cudaMemcpyDeviceToHost, g.stream));
            });
            warm(g, o, copies.back());
            if (h2d) {
                check(cudaMemsetAsync(g.b, 0xa5, n, g.stream));
                check(cudaStreamSynchronize(g.stream));
            } else std::memset(p, 0xa5, n);
        }
        const auto t = simultaneous(gpus, o, copies);
        for (size_t i = 0; i < gpus.size(); ++i) {
            if (h2d) readback(*gpus[i], verifier, n, key);
            else verify(host.p + (i + 1) * stride, n, key + static_cast<unsigned>(i * 2));
        }
        row(round, host.mode, direction + " ALL aggregate", n, t, o, gpus.size());
    }
}

struct PeerAccess {
    int dst, src;
    bool enabled = false;
    PeerAccess(int d, int s, bool on) : dst(d), src(s) {
        check(cudaSetDevice(dst));
        if (!on) return;
        const auto e = cudaDeviceEnablePeerAccess(src, 0);
        if (e == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); return; }
        check(e);
        enabled = true;
    }
    ~PeerAccess() { if (enabled) { cudaSetDevice(dst); cudaDeviceDisablePeerAccess(src); } }
};
void device_rows(const Options& o, const std::vector<std::unique_ptr<Gpu>>& gpus, unsigned char* verifier,
                 unsigned& seed, std::string& phase) {
    if (!has(o.directions, "d2d")) return;
    for (size_t n : o.sizes) for (const auto& destination : gpus) for (const auto& source : gpus) {
        auto& dst = *destination;
        auto& src = *source;
        int capable = 0;
        if (dst.id != src.id) check(cudaDeviceCanAccessPeer(&capable, dst.id, src.id));
        for (const bool on : {false, true}) {
            if (dst.id == src.id && on) continue;
            if (dst.id != src.id && (o.peer == "on" ? !on : o.peer == "off" && on)) continue;
            if (on && !capable) continue;
            phase = "GPU" + std::to_string(src.id) + " -> GPU" + std::to_string(dst.id) +
                    " peer-access=" + (on ? "on" : "off") + " bytes=" + std::to_string(n);
            PeerAccess peer(dst.id, src.id, on);
            const unsigned key = ++seed * 2 + 1;
            pattern(verifier, n, key);
            check(cudaSetDevice(src.id));
            check(cudaMemcpy(src.a, verifier, n, cudaMemcpyHostToDevice));
            check(cudaSetDevice(dst.id));
            Copy copy = [&] {
                if (dst.id == src.id) check(cudaMemcpyAsync(dst.b, src.a, n, cudaMemcpyDeviceToDevice, dst.stream));
                else check(cudaMemcpyPeerAsync(dst.b, dst.id, src.a, src.id, n, dst.stream));
            };
            warm(dst, o, copy);
            check(cudaMemsetAsync(dst.b, 0xa5, n, dst.stream));
            check(cudaStreamSynchronize(dst.stream));
            const auto t = measure(dst, o, copy, o.gate_ms);
            readback(dst, verifier, n, key);
            row(0, dst.id == src.id ? "device-local" : on ? "peer-access-on" : "peer-access-off",
                "GPU" + std::to_string(src.id) + " -> GPU" + std::to_string(dst.id), n, t, o);
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    std::string phase = "arguments";
    try {
        Options o;
        if (!parse(argc, argv, o)) return 0;
        int count = 0;
        const auto status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice || (status == cudaSuccess && !count)) {
            std::puts("SKIP: no CUDA devices"); return 77;
        }
        check(status);
        if (o.gpus.empty()) for (int i = 0; i < count; ++i) o.gpus.push_back(i);
        for (int id : o.gpus) if (id >= count) throw std::runtime_error("GPU index out of range");
        const size_t stride = *std::max_element(o.sizes.begin(), o.sizes.end());
        const size_t minimum = scaled(stride, o.gpus.size() + 1);
        if (!o.host_bytes) o.host_bytes = std::max<size_t>(64 * MiB, minimum);
        if (o.host_bytes < minimum) throw std::runtime_error("--host-mib too small for disjoint per-GPU buffers");
        int runtime = 0, driver = 0;
        check(cudaRuntimeGetVersion(&runtime)); check(cudaDriverGetVersion(&driver));
        std::printf("# CUDA transfer microbench\n\n%s; %s; CUDA headers %d, runtime %d, driver %d.\n\n",
                    TRANSFER_BENCH_SYSTEM, TRANSFER_BENCH_COMPILER, CUDART_VERSION, runtime, driver);
        std::printf("Host %.2f MiB; owner GPU%d; reps %d; warmup %d; host rounds %d; gate %d ms.\n\n",
                    o.host_bytes / double(MiB), o.gpus.front(), o.reps, o.warmup, o.rounds, o.gate_ms);
        Power power;
        std::vector<std::unique_ptr<Gpu>> gpus;
        for (int id : o.gpus) {
            phase = "initialize GPU" + std::to_string(id);
            check(cudaSetDevice(id)); check(cudaFree(nullptr));
            cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, id));
            char bus[32]{}; check(cudaDeviceGetPCIBusId(bus, sizeof bus, id));
            size_t free = 0, total = 0; check(cudaMemGetInfo(&free, &total));
            std::printf("- GPU%d: %s; PCI %s; free/total %.2f/%.2f GiB; power limit %s W\n",
                        id, prop.name, bus, free / double(GiB), total / double(GiB), power.watts(bus).c_str());
            gpus.emplace_back(std::make_unique<Gpu>(id));
            gpus.back()->init(stride);
        }
        std::puts("\nDirected P2P capability (accessor -> owner):");
        for (int a : o.gpus) for (int b : o.gpus) if (a != b) {
            int can = 0; check(cudaDeviceCanAccessPeer(&can, a, b));
            std::printf("- GPU%d -> GPU%d: %s%s\n", a, b, can ? "yes" : "no", can ? "" : " (access-on rows SKIP)");
        }
        Host verifier;
        verifier.init("portable", stride, o.gpus);
        std::puts("\nSubmit is CPU microseconds/copy; stream GB/s includes enqueue gaps. Wall GB/s includes completion.\n"
                  "ALL rows: aggregate bytes/common wall interval; submit is the slowest worker's mean.\n"
                  "Gate-pending says enqueue returned before the optional gate opened. Units: binary MiB, SI GB/s.\n\n"
                  "| Round | Host / peer mode | Direction | MiB/copy | Submit us/copy | Stream GB/s | Wall GB/s | Gate pending | Validation |\n"
                  "|---:|---|---|---:|---:|---:|---:|---|---|");
        unsigned seed = 1;
        std::vector<std::string> allocations;
        for (int round = 1; round <= o.rounds; ++round) {
            auto modes = o.modes;
            if (!(round % 2)) std::reverse(modes.begin(), modes.end());
            for (const auto& mode : modes) {
                if (!has(o.directions, "h2d") && !has(o.directions, "d2h")) break;
                phase = mode + " allocation, round " + std::to_string(round);
                Host host;
                const auto start = Clock::now();
                try { host.init(mode, o.host_bytes, o.gpus); }
                catch (const Unsupported& e) {
                    allocations.push_back("SKIP " + mode + ": " + e.what()); continue;
                }
                char note[256]{};
                std::snprintf(note, sizeof note, "Round %d %s: %.2f MiB %s; allocation + touch %.3f s.",
                              round, mode.c_str(), (host.mapped ? host.mapped : o.host_bytes) / double(MiB),
                              host.mapped ? "mapped" : "requested",
                              seconds(Clock::now() - start));
                allocations.emplace_back(note);
                phase = mode + " RAM transfers, round " + std::to_string(round);
                host_rows(o, gpus, host, verifier.p, stride, round, seed, phase);
            }
        }
        phase = "device transfers";
        device_rows(o, gpus, verifier.p, seed, phase);
        std::puts("\nAllocations and skips:\n");
        for (const auto& note : allocations) std::printf("- %s\n", note.c_str());
        if (!measured_rows) { std::puts("\nSKIP: no supported transfer rows."); return 77; }
        std::printf("\nPASS: %zu measured rows passed a full byte comparison. Unsupported cases are explicitly skipped.\n",
                    measured_rows);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "transfer_bench: %s: %s\n", phase.c_str(), e.what());
        return 1;
    }
}
