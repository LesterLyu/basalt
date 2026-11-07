#include "basalt/core/peer_copy.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
void must(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}
void check(bool ok, const std::string& msg) {
    if (!ok) { std::fprintf(stderr, "%s\n", msg.c_str()); std::exit(1); }
}

void direction(int src, int dst) {
    constexpr size_t blob = 2764816, count = 67;
    std::vector<uint8_t> input(blob * count), output(input.size(), 0);
    std::mt19937 rng(7);
    for (auto& b : input) b = (uint8_t) rng();
    must(cudaSetDevice(src));
    uint8_t* a = nullptr;
    must(cudaMalloc((void**) &a, input.size()));
    must(cudaMemcpy(a, input.data(), input.size(), cudaMemcpyHostToDevice));
    must(cudaSetDevice(dst));
    uint8_t* b = nullptr;
    cudaStream_t stream = nullptr;
    must(cudaMalloc((void**) &b, input.size()));
    must(cudaMemset(b, 0, input.size()));
    must(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    must(cudaDeviceSynchronize());
    basalt::core::PeerCopy copy;
    std::string err;
    check(!copy.open(src, src, blob, 16, err), "accepted same-device transport");
    check(copy.open(src, dst, blob, 16, err), err);
    check(copy.host_bytes() == blob * 16, "incorrect pinned-memory budget");
    check(!copy.copy(b, a, blob + 1, stream, err), "accepted an oversized transfer");
    must(cudaSetDevice(src));
    const auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < count; ++i) {
        const size_t bytes = i % 3 == 0 ? blob : i % 3 == 1 ? 4097 : 1;
        check(copy.copy(b + i * blob, a + i * blob, bytes, stream, err), err);
        int current = -1;
        must(cudaGetDevice(&current));
        check(current == src, "copy changed the caller's device");
    }
    // Close with copies outstanding: no pinned buffer or event may be freed early.
    copy.close();
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    must(cudaSetDevice(dst));
    must(cudaMemcpy(output.data(), b, output.size(), cudaMemcpyDeviceToHost));
    size_t transferred = 0;
    for (size_t i = 0; i < count; ++i) {
        const size_t bytes = i % 3 == 0 ? blob : i % 3 == 1 ? 4097 : 1;
        check(std::memcmp(input.data() + i * blob, output.data() + i * blob, bytes) == 0, "copied bytes differ");
        check(std::all_of(output.begin() + i * blob + bytes, output.begin() + (i + 1) * blob,
                          [](uint8_t value) { return value == 0; }), "copy overwrote its requested range");
        transferred += bytes;
    }
    check(copy.host_bytes() == 0, "close retained pinned buffers");
    check(copy.open(src, dst, blob, 16, err), err);
    for (size_t i = 0; i < count; ++i) check(copy.copy(b, a, blob, stream, err), err);
    copy.close();
    must(cudaMemcpy(output.data(), b, blob, cudaMemcpyDeviceToHost));
    check(std::memcmp(input.data(), output.data(), blob) == 0, "reopened transport differs");
    must(cudaStreamDestroy(stream));
    must(cudaFree(b));
    must(cudaSetDevice(src));
    must(cudaFree(a));
    int direct = 0;
    must(cudaDeviceCanAccessPeer(&direct, src, dst));
    std::printf("peer_copy %d->%d p2p=%d: byte identity, 16 buffers %.2f MiB, %.1f us/copy, %.2f GB/s\n",
                src, dst, direct, (double) (blob * 16) / (1 << 20), sec * 1e6 / count, transferred / sec / 1e9);
}
}  // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
        std::puts("peer_copy_test: needs two CUDA devices");
        return 77;
    }
    direction(0, 1);
    direction(1, 0);
    std::puts("peer_copy_test: ok");
}
