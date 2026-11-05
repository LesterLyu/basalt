// Clamped SwiGLU before activation quantization, against scalar double arithmetic.
#include "basalt/ggml/ggml.h"
#define GGML_COMMON_DECL_CPP
#include "basalt/ggml/ggml_common.h"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/f16_bits.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = basalt::kernels;
namespace cpu = k::cpu;
static int failed = 0;
static void ck(cudaError_t e) { if (e != cudaSuccess) { std::printf("CUDA: %s\n", cudaGetErrorString(e)); std::exit(2); } }
template<class T> static T* alloc(size_t n) { T* p = nullptr; ck(cudaMalloc(&p, n * sizeof(T))); return p; }
static double reference(double g, double u, double limit) {
    if (limit > 0) { g = std::min(g, limit); u = std::clamp(u, -limit, limit); }
    return (g / (1.0 + std::exp(-g))) * u;
}
static void cpu_check(std::mt19937& rng) {
    for (int type : {16, 19, 39}) {
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt(type, 19, 256, 256, f, err)) { std::printf("%s\n", err.c_str()); ++failed; continue; }
        std::vector<uint8_t> blob(f.bytes);
        for (auto& b : blob) b = uint8_t(rng());
        const int block = type == 39 ? 32 : 256;
        const size_t bs = ggml_row_size(ggml_type(type), block);
        for (size_t i = 0; i < f.down_off; i += bs) {
            if (type == 39) blob[i] = 123;
            else { const uint16_t d = k::f16_from_f32(1.f / 128.f); std::memcpy(blob.data() + i, &d, 2); }
        }
        std::vector<float> weights(2 * 256 * 256), x(256), result(256);
        ggml_get_type_traits(ggml_type(type))->to_float(blob.data(), weights.data(), weights.size());
        // Isolate the activation from cancellation in the row reductions (covered by the dense dot tests).
        x[0] = 128.f;
        std::vector<uint8_t> q(f.act_bytes);
        cpu::native_quant_act(f, x.data(), q.data());
        std::vector<double> act(256);
        for (int i = 0; i < 256; ++i) {
            if (f.gu_act == GGML_TYPE_Q8_K) {
                const auto* b = reinterpret_cast<const block_q8_K*>(q.data());
                act[i] = double(b->d) * b->qs[i];
            } else {
                const auto* b = reinterpret_cast<const block_q8_0*>(q.data()) + i / 32;
                act[i] = double(k::f32_from_f16(b->d)) * b->qs[i % 32];
            }
        }
        const void* inputs[] = {q.data()};
        float* outputs[] = {result.data()};
        double max_abs = 0, max_rel = 0;
        for (float limit : {0.f, 10.f}) {
            f.swiglu_limit = limit;
            cpu::native_gu_rows(f, blob.data(), inputs, 1, outputs, 0, 256);
            for (int r = 0; r < 256; ++r) {
                double g = 0, u = 0;
                for (int i = 0; i < 256; ++i) {
                    g += weights[r * 256 + i] * act[i];
                    u += weights[(r + 256) * 256 + i] * act[i];
                }
                const double ref = reference(g, u, limit), abs = std::abs(result[r] - ref);
                max_abs = std::max(max_abs, abs);
                max_rel = std::max(max_rel, abs / (std::abs(ref) + 1e-30));
                if (!std::isfinite(result[r]) || abs > 2e-5 * std::max(1., std::abs(ref))) {
                    std::printf("CPU FAIL type %d clamp %g row %d: got %.9g ref %.9g gate %.9g up %.9g\n",
                                type, limit, r, result[r], ref, g, u);
                    ++failed;
                }
            }
        }
        std::printf("CPU type %d clamp 0/10: max abs %.3g rel %.3g\n", type, max_abs, max_rel);
    }
}
static void gpu_check(std::mt19937& rng) {
    const size_t n = 3 * 5120 + 17;
    std::vector<float> gate(n), up(n), out(n);
    std::normal_distribution<float> nd(0.f, 25.f);
    for (size_t i = 0; i < n; ++i) { gate[i] = nd(rng); up[i] = nd(rng); }
    const float edge[] = {-100.f, -11.f, -10.f, -1.f, 0.f, 1.f, 10.f, 11.f, 100.f};
    for (int i = 0; i < 9; ++i) for (int j = 0; j < 9; ++j) { gate[i * 9 + j] = edge[i]; up[i * 9 + j] = edge[j]; }
    auto* g = alloc<float>(n); auto* u = alloc<float>(n); auto* y = alloc<float>(n);
    ck(cudaMemcpy(g, gate.data(), n * 4, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(u, up.data(), n * 4, cudaMemcpyHostToDevice));
    for (float limit : {0.f, 10.f}) {
        k::native_swiglu_f32(g, u, y, n, limit, nullptr);
        ck(cudaMemcpy(out.data(), y, n * 4, cudaMemcpyDeviceToHost));
        double max_abs = 0, max_rel = 0;
        for (size_t i = 0; i < n; ++i) {
            const double ref = reference(gate[i], up[i], limit), abs = std::abs(out[i] - ref);
            max_abs = std::max(max_abs, abs); max_rel = std::max(max_rel, abs / (std::abs(ref) + 1e-30));
            failed += !std::isfinite(out[i]) || abs > 2e-6 * std::max(1., std::abs(ref));
        }
        std::printf("GPU clamp %g: max abs %.3g rel %.3g\n", limit, max_abs, max_rel);
        cudaEvent_t a, b; ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b));
        for (int i = 0; i < 10; ++i) k::native_swiglu_f32(g, u, y, n, limit, nullptr);
        ck(cudaEventRecord(a));
        for (int i = 0; i < 100; ++i) k::native_swiglu_f32(g, u, y, n, limit, nullptr);
        ck(cudaEventRecord(b)); ck(cudaEventSynchronize(b));
        float ms = 0; ck(cudaEventElapsedTime(&ms, a, b));
        std::printf("SwiGLU %zu values: %.2f us/call, %.2f GB/s\n", n, ms * 10, n * 12 / (ms * 1e4));
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    cudaFree(g); cudaFree(u); cudaFree(y);
}
int main(int argc, char** argv) {
    bool gpu = true; int dev = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--cpu") gpu = false;
        else if (a == "--device" && i + 1 < argc) dev = std::stoi(argv[++i]);
        else return 2;
    }
    std::mt19937 rng(5120);
    cpu_check(rng);
    if (gpu) { ck(cudaSetDevice(dev)); gpu_check(rng); }
    std::printf("native_swiglu_parity: %d failures\n", failed);
    return failed ? 1 : 0;
}
