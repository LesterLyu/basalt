// Compare every uploaded dense byte with its source pack, including exact BF16 router widening.
#include "models/deepseek41/model.hpp"
#include "basalt/platform/file.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {
void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
void ck(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }
}

int main(int argc, char** argv) {
    if (argc != 2) return 77;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    int fd = -1;
    try {
        ck(cudaSetDevice(0));
        ck(cudaFree(nullptr));
        size_t before = 0, after = 0, total = 0;
        ck(cudaMemGetInfo(&before, &total));
        {
            basalt::models::deepseek41::DsModel model;
            std::string err;
            require(model.load(argv[1], err), err);
            fd = basalt::platform::open_fd_utf8(argv[1]);
            require(fd >= 0, "open reference pack");
            uint64_t payload = 0, end = 0, count = 0;
            const char* base = nullptr;
            std::vector<uint8_t> source, expected, actual;
            for (const auto& [name, tensor] : model.pack().tensors) {
                const bool skipped = name == "token_embd.weight" || name == "per_layer_token_embd.weight";
                const auto* loaded = model.tensor(name);
                if (skipped) { require(!loaded, "file-backed tensor was uploaded: " + name); continue; }
                require(loaded && loaded->ptr, "missing dense tensor: " + name);
                const bool widen = tensor.type == 30 && name.find("ffn_gate_inp.weight") != std::string::npos;
                const uint64_t bytes = tensor.bytes * (widen ? 2 : 1);
                require(loaded->type == (widen ? 0 : tensor.type) && loaded->bytes == bytes && loaded->shape == tensor.shape,
                        "dense metadata differs: " + name);
                if (!base) base = (const char*) loaded->ptr;
                end = (end + 255) / 256 * 256;
                require((const char*) loaded->ptr == base + end, "tensor is outside its planned view: " + name);
                require((uintptr_t) loaded->ptr % 256 == 0, "unaligned tensor: " + name);
                for (uint64_t off = 0; off < tensor.bytes;) {
                    const size_t n = (size_t) std::min<uint64_t>(64ull << 20, tensor.bytes - off);
                    source.resize(n);
                    require(basalt::platform::pread_all(fd, source.data(), n, tensor.offset + off), "reference read: " + name);
                    expected.resize(n * (widen ? 2 : 1));
                    if (widen) {
                        require(n % 2 == 0, "odd BF16 byte count");
                        for (size_t i = 0; i < n / 2; ++i) {
                            uint16_t word;
                            std::memcpy(&word, source.data() + 2 * i, 2);
                            const uint32_t bits = (uint32_t) word << 16;
                            std::memcpy(expected.data() + 4 * i, &bits, 4);
                        }
                    } else std::memcpy(expected.data(), source.data(), n);
                    actual.resize(expected.size());
                    ck(cudaMemcpy(actual.data(), (const char*) loaded->ptr + off * (widen ? 2 : 1), actual.size(), cudaMemcpyDeviceToHost));
                    require(actual == expected, "uploaded bytes differ: " + name);
                    off += n;
                }
                end += bytes; payload += bytes; ++count;
            }
            basalt::core::MemoryUsage usage;
            model.memory_usage(usage);
            require(usage.gpus[0]["weights"] == end, "weight telemetry does not match arena capacity");
            ck(cudaMemGetInfo(&after, &total));
            std::printf("dense_load_test: %llu tensors, %llu payload bytes, %llu arena bytes, CUDA delta %.3f MiB; all bytes exact\n",
                        (unsigned long long) count, (unsigned long long) payload, (unsigned long long) end,
                        ((double) before - after) / 1048576.0);
        }
        basalt::platform::close_fd(fd); fd = -1;
        ck(cudaMemGetInfo(&after, &total));
        require(after + (2ull << 20) >= before, "dense arena was not released");
        return 0;
    } catch (const std::exception& e) {
        if (fd >= 0) basalt::platform::close_fd(fd);
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
