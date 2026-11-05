#include "deepseek_attention.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace basalt::vision;

static void cuda_check(cudaError_t result) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(result));
        std::exit(1);
    }
}

int main() {
    constexpr int width = 1024, head = 64, stride = width * 3;
    std::mt19937 rng(173);
    std::uniform_real_distribution<float> random(-1.f, 1.f);
    for (int n : {63, 65, 193}) {
        std::vector<__nv_bfloat16> input(size_t(n) * stride), out(size_t(n) * width);
        for (auto& v : input) v = __float2bfloat16_rn(random(rng));
        uint16_t *qkv = nullptr, *dst = nullptr;
        cuda_check(cudaMalloc(&qkv, input.size() * 2));
        cuda_check(cudaMalloc(&dst, out.size() * 2));
        cuda_check(cudaMemcpy(qkv, input.data(), input.size() * 2, cudaMemcpyHostToDevice));
        deepseek_attention::attention<<<dim3((n + 63) / 64, 16), 128>>>(qkv, dst, n, .125f * 1.4426950408889634f);
        cuda_check(cudaGetLastError());
        cuda_check(cudaMemcpy(out.data(), dst, out.size() * 2, cudaMemcpyDeviceToHost));
        cuda_check(cudaFree(qkv));
        cuda_check(cudaFree(dst));
        double peak = 0, max_abs = 0, sqerr = 0, sqref = 0;
        std::vector<double> scores(n);
        for (int q = 0; q < n; ++q) for (int h = 0; h < width; h += head) {
            const auto* qr = input.data() + q * stride + h;
            for (int k = 0; k < n; ++k) {
                const auto* kr = input.data() + k * stride + width + h;
                double score = 0;
                for (int i = 0; i < head; ++i) score += double(float(qr[i])) * float(kr[i]);
                scores[k] = score * .125;
            }
            const auto max_score = *std::max_element(scores.begin(), scores.end());
            double total = 0;
            for (auto& s : scores) total += (s = std::exp(s - max_score));
            for (int d = 0; d < head; ++d) {
                double expected = 0;
                for (int k = 0; k < n; ++k) expected += scores[k] * float(input[k * stride + width * 2 + h + d]);
                expected /= total;
                const double delta = float(out[q * width + h + d]) - expected;
                peak = std::max(peak, std::abs(expected));
                max_abs = std::max(max_abs, std::abs(delta));
                sqerr += delta * delta;
                sqref += expected * expected;
            }
        }
        std::printf("patches=%d max_abs=%.6g peak_normalized=%.6g rms_relative=%.6g\n", n, max_abs,
                    max_abs / peak, std::sqrt(sqerr / sqref));
        if (!(max_abs < .002 && sqerr / sqref < .000025)) return 1;
    }
    constexpr int full = 270 * 270;  // 90×90 aligner cells
    uint16_t *qkv = nullptr, *out = nullptr;
    cuda_check(cudaMalloc(&qkv, size_t(full) * stride * 2));
    cuda_check(cudaMalloc(&out, size_t(full) * width * 2));
    cuda_check(cudaMemset(qkv, 0, size_t(full) * stride * 2));
    const auto launch = [&] {
        deepseek_attention::attention<<<dim3((full + 63) / 64, 16), 128>>>(
            qkv, out, full, .125f * 1.4426950408889634f);
        cuda_check(cudaGetLastError());
    };
    launch();
    cudaEvent_t begin, end;
    cuda_check(cudaEventCreate(&begin));
    cuda_check(cudaEventCreate(&end));
    cuda_check(cudaEventRecord(begin));
    launch();
    cuda_check(cudaEventRecord(end));
    cuda_check(cudaEventSynchronize(end));
    float ms;
    cuda_check(cudaEventElapsedTime(&ms, begin, end));
    std::printf("patches=%d ms=%.3f attention_TFLOPS=%.3f\n", full, ms, 4.0 * full * full * width / (ms * 1e9));
    cuda_check(cudaEventDestroy(begin));
    cuda_check(cudaEventDestroy(end));
    cuda_check(cudaFree(qkv));
    cuda_check(cudaFree(out));
}
