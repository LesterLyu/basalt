// Native GGUF MXFP4: independent scalar reference, window invariance and optional real DSpark slices.
#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/f16_bits.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
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
static int failures = 0;
static void ck(cudaError_t e) {
    if (e != cudaSuccess) { std::printf("CUDA: %s\n", cudaGetErrorString(e)); std::exit(2); }
}
template<class T> static T* alloc(size_t n) { T* p = nullptr; ck(cudaMalloc(&p, n * sizeof(T))); return p; }
static float scalar(const uint8_t* b, int i) {
    const int values[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
    return std::ldexp(float(values[(b[1 + i % 16] >> (i < 16 ? 0 : 4)) & 15]), int(b[0]) - 128);
}
static std::vector<uint8_t> random_matrix(int n, int rows, std::mt19937& rng) {
    std::vector<uint8_t> w(size_t(n / 32) * rows * 17);
    for (auto& b : w) b = uint8_t(rng());
    for (size_t i = 0; i < w.size(); i += 17) w[i] = uint8_t(114 + rng() % 12);
    return w;
}
static void check_matrix(const std::vector<uint8_t>& w, int n, int rows, bool gpu, bool bench, std::mt19937& rng) {
    constexpr int cols = 11;
    const size_t rb = size_t(n / 32) * 17;
    std::vector<float> wf(size_t(rows) * n), x(cols * n);
    ggml_get_type_traits(GGML_TYPE_MXFP4)->to_float(w.data(), wf.data(), wf.size());
    for (size_t i = 0; i < wf.size(); ++i)
        failures += wf[i] != scalar(w.data() + i / 32 * 17, i % 32);
    std::normal_distribution<float> nd;
    for (auto& v : x) v = nd(rng);
    double cpu_abs = 0, cpu_rel = 0;
    for (int isa : {2, 3}) {
        if (isa == 3 && !k::cpu::cpu_avx512_ok()) continue;
        const auto* traits = isa == 2 ? ggml_get_type_traits_cpu_avx2(GGML_TYPE_MXFP4)
                                     : ggml_get_type_traits_cpu_avx512(GGML_TYPE_MXFP4);
        std::vector<uint8_t> q(size_t(n / 32) * 34);
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(x.data(), q.data(), n);
        for (int r = 0; r < rows; ++r) {
            double expected = 0, magnitude = 0;
            for (int i = 0; i < n; ++i) {
                const uint8_t* b = q.data() + (i / 32) * 34;
                uint16_t d; std::memcpy(&d, b, 2);
                const double v = double(wf[r * n + i]) * k::f32_from_f16(d) * int8_t(b[2 + i % 32]);
                expected += v; magnitude += std::abs(v);
            }
            float out = 0;
            traits->vec_dot(n, &out, 0, w.data() + r * rb, 0, q.data(), 0, 1);
            const double abs = std::abs(out - expected);
            cpu_abs = std::max(cpu_abs, abs);
            cpu_rel = std::max(cpu_rel, abs / (std::abs(expected) + 1e-30));
            failures += !std::isfinite(out) || abs > 2e-6 * magnitude;
        }
    }
    std::printf("MXFP4 CPU %dx%d: max abs %.3g rel %.3g\n", rows, n, cpu_abs, cpu_rel);
    if (!gpu) return;
    auto* dw = alloc<uint8_t>(w.size());
    auto* dx = alloc<float>(x.size());
    auto* q = alloc<uint8_t>(size_t(cols) * n / 32 * 36);
    auto* dy = alloc<float>(std::max(size_t(cols * rows), wf.size()));
    ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    k::iq_dequant_f32(39, dw, wf.size(), dy, nullptr);
    std::vector<float> dq(wf.size());
    ck(cudaMemcpy(dq.data(), dy, dq.size() * 4, cudaMemcpyDeviceToHost));
    failures += std::memcmp(dq.data(), wf.data(), wf.size() * 4) != 0;
    k::quantize_q8_1_rows(dx, cols, n, q, nullptr);
    std::vector<uint8_t> hq(size_t(cols) * n / 32 * 36);
    ck(cudaMemcpy(hq.data(), q, hq.size(), cudaMemcpyDeviceToHost));
    std::vector<float> one(cols * rows), out(one.size());
    for (int c = 0; c < cols; ++c)
        k::native_mmvq(39, dw, q + size_t(c) * n / 32 * 36, dy + c * rows, n, rows, 1, nullptr);
    ck(cudaMemcpy(one.data(), dy, one.size() * 4, cudaMemcpyDeviceToHost));
    double max_abs = 0, max_rel = 0;
    for (int nc : {1, 2, 3, 5, 8, 11}) for (bool old : {true, false}) {
        k::iq_set_old_kernels(old);
        k::native_mmvq(39, dw, q, dy, n, rows, nc, nullptr);
        ck(cudaMemcpy(out.data(), dy, nc * rows * 4, cudaMemcpyDeviceToHost));
        failures += std::memcmp(out.data(), one.data(), nc * rows * 4) != 0;
    }
    for (int c = 0; c < cols; ++c) for (int r = 0; r < rows; ++r) {
        double expected = 0, magnitude = 0;
        for (int i = 0; i < n; ++i) {
            const uint8_t* b = hq.data() + (size_t(c) * n / 32 + i / 32) * 36;
            uint16_t d; std::memcpy(&d, b, 2);
            const double v = double(wf[r * n + i]) * k::f32_from_f16(d) * int8_t(b[4 + i % 32]);
            expected += v; magnitude += std::abs(v);
        }
        const double abs = std::abs(out[c * rows + r] - expected);
        max_abs = std::max(max_abs, abs);
        max_rel = std::max(max_rel, abs / (std::abs(expected) + 1e-30));
        failures += !std::isfinite(out[c * rows + r]) || abs > 2e-6 * magnitude;
    }
    std::printf("MXFP4 GPU %dx%d: max abs %.3g rel %.3g; columns/positions byte comparison\n", rows, n, max_abs, max_rel);
    k::iq_set_old_kernels(false);
    if (bench) {
        cudaEvent_t a, b;
        ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b));
        for (int i = 0; i < 10; ++i) k::iq_mmvq(39, dw, q, dy, n, rows, 3, nullptr);
        ck(cudaEventRecord(a));
        for (int i = 0; i < 100; ++i) k::iq_mmvq(39, dw, q, dy, n, rows, 3, nullptr);
        ck(cudaEventRecord(b)); ck(cudaEventSynchronize(b));
        float ms = 0; ck(cudaEventElapsedTime(&ms, a, b));
        std::printf("MXFP4 %dx%d nc=3: %.2f us/call, %.2f weight GB/s\n", rows, n, ms * 10, w.size() / (ms * 1e4));
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    cudaFree(dw); cudaFree(dx); cudaFree(q); cudaFree(dy);
}

static void dequant_edges(bool gpu) {
    // Every stored exponent and nibble; the reference uses ldexp rather than exponent-bit construction.
    std::vector<uint8_t> w(256 * 17);
    for (int e = 0; e < 256; ++e) {
        w[e * 17] = uint8_t(e);
        for (int j = 0; j < 16; ++j) w[e * 17 + 1 + j] = uint8_t(j | ((15 - j) << 4));
    }
    std::vector<float> cpu(8192), out(cpu.size());
    ggml_get_type_traits(GGML_TYPE_MXFP4)->to_float(w.data(), cpu.data(), cpu.size());
    for (size_t i = 0; i < cpu.size(); ++i) failures += cpu[i] != scalar(w.data() + i / 32 * 17, i % 32);
    if (gpu) {
        auto* dw = alloc<uint8_t>(w.size()); auto* df = alloc<float>(out.size());
        ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
        k::iq_dequant_f32(39, dw, out.size(), df, nullptr);
        ck(cudaMemcpy(out.data(), df, out.size() * 4, cudaMemcpyDeviceToHost));
        failures += std::memcmp(out.data(), cpu.data(), out.size() * 4) != 0;
        cudaFree(dw); cudaFree(df);
    }
    std::printf("MXFP4 all exponent/nibble codes: scalar/CPU%s comparison\n", gpu ? "/GPU" : "");
}

int main(int argc, char** argv) {
    bool gpu = true, bench = false;
    int dev = 0;
    std::string gguf;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--cpu") gpu = false;
        else if (a == "--bench") bench = true;
        else if (a == "--device" && i + 1 < argc) dev = std::stoi(argv[++i]);
        else if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
        else return 2;
    }
    if (gpu) { ck(cudaSetDevice(dev)); std::printf("CUDA device %d\n", dev); }
    std::mt19937 rng(39);
    dequant_edges(gpu);
    for (int n : {2304, 5120}) check_matrix(random_matrix(n, 67, rng), n, 67, gpu, bench, rng);
    if (!gguf.empty()) {
        basalt::GgufFile f(gguf);
        int count = 0;
        for (const auto& t : f.tensors()) if (t.type == 39 && t.shape.size() == 3) {
            const int n = int(t.shape[0]), rows = int(t.shape[1]), experts = int(t.shape[2]);
            if (n % 256 || rows <= 0 || experts <= 0) return 2;
            const size_t rb = size_t(n / 32) * 17, bytes = rb * rows;
            std::vector<uint8_t> w;
            for (int e : {0, experts / 2, experts - 1}) for (int r : {0, rows / 2, rows - 1}) {
                const uint8_t* p = f.tensor_data(t) + e * bytes + r * rb;
                w.insert(w.end(), p, p + rb);
            }
            std::printf("Real tensor: %s, experts 0/%d/%d, rows 0/%d/%d\n", t.name.c_str(), experts / 2, experts - 1, rows / 2, rows - 1);
            check_matrix(w, n, 9, gpu, false, rng);
            ++count;
        }
        failures += count != 9;
        std::printf("DSpark MXFP4 tensors checked: %d\n", count);
    }
    std::printf("mxfp4_parity: %d failures\n", failures);
    return failures ? 1 : 0;
}
