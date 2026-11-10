// tests/kernels/cpu/iq_avx512_test.cpp - the AVX-512 i-quant rows against their frozen output hashes and a plain
// double-precision reference.
//
//     build/iq_avx512_test [--print] [--bench]
//
// Gate/up rows (`iq512_gu_rows`) and plain dot rows (`iq512_rows`) of IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S and IQ2_S at the
// model's row geometry, 1 to 8 tokens, on random weights, on weights pinned to their extreme codes (every grid index and
// sign bit set, and every one clear) and with saturating activations.  Checks: (1) every dot-row word hashes to the value
// the kernels gave when the hash was frozen (the decode may change, the bits may not); (2) a token's rows do not depend on
// how many tokens share the call; (3) the dot rows and the gate/up rows against a double reference (dequantize, then sum in
// double), whose error is reported.  The dot rows are hashed: their integer decode and float reduction are the same on
// every compiler.  The gate/up rows carry the silu, whose expf comes from the C library and does not agree between MSVC
// and glibc, so they are checked against the reference instead of a frozen bit pattern.  The fixture uses only the raw
// generator output, so it is the same on every standard library.
// `--print` prints the hashes, `--bench` one in-cache thread's rate per format and token count.
#include "basalt/models/qwen4exp/geometry.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/iq_avx512.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = basalt::kernels::cpu;

namespace {

constexpr int H = cpu::H, FF = cpu::FF, MAXT = 8, MODES = 4;

struct Frozen {
    int type;
    uint64_t hash;
};
// FNV-1a over every dot-row word of every mode and token count, in the order main() computes them
constexpr Frozen kFrozen[] = {
    {16, 0x5863fdcf90fa4725ull}, {17, 0x3acf759ebce5f2a1ull}, {18, 0x5861282d89bf5a41ull},
    {21, 0x931a4e9ae9822c6aull}, {22, 0xacaaeba7bf448c1eull},
};

// mode 0 random bytes, 1 every code bit set, 2 random, 3 every code bit clear; a valid small fp16 scale at the start of
// each block (IQ2_XS / IQ3_S / IQ2_S scale nibbles stay in the code bytes)
void fill(std::vector<uint8_t>& v, int type, uint64_t seed, int mode) {
    std::mt19937_64 rng(seed);
    for (size_t i = 0; i + 8 <= v.size(); i += 8) {
        const uint64_t x = mode == 1 ? ~0ull : mode == 3 ? 0ull : rng();
        std::memcpy(v.data() + i, &x, 8);
    }
    const size_t bs = ggml_type_size((ggml_type) type);
    for (size_t b = 0; b + bs <= v.size(); b += bs) {
        const uint16_t d = (uint16_t) (0x2800 + (rng() & 0xff));
        std::memcpy(&v[b], &d, 2);
    }
}

// block_q8_K: float d, int8 qs[256], int16 bsums[16]
void dequant_q8_K(const uint8_t* p, float* y, int n) {
    for (int b = 0; b < n / 256; ++b, p += 292) {
        float d;
        std::memcpy(&d, p, 4);
        for (int i = 0; i < 256; ++i) *y++ = d * (float) (int8_t) p[4 + i];
    }
}

void fnv(uint64_t& h, const float* p, size_t n) {
    const auto* b = (const unsigned char*) p;
    for (size_t i = 0; i < 4 * n; ++i) h = (h ^ b[i]) * 1099511628211ull;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Limited-SwiGLU rows must keep the single-token ggml rounding, including when a token moves within a window.
int check_exact_iq2s(int n, int rows, int mode) {
    constexpr int T = 8;
    const size_t stride = ggml_row_size(GGML_TYPE_IQ2_S, n), up = stride * rows;
    std::vector<uint8_t> weights(2 * up);
    fill(weights, 22, 104 + n + mode, mode);
    const auto* dot = ggml_get_type_traits_cpu_avx512(GGML_TYPE_IQ2_S);
    const auto* quant = ggml_get_type_traits_cpu_avx512(GGML_TYPE_Q8_K);
    const auto* dw = ggml_get_type_traits(GGML_TYPE_IQ2_S);
    std::vector<std::vector<uint8_t>> packed(T, std::vector<uint8_t>(ggml_row_size(GGML_TYPE_Q8_K, n)));
    std::vector<std::vector<float>> x(T, std::vector<float>(n));
    std::vector<std::vector<float>> ref(T, std::vector<float>(rows)), gu(T, std::vector<float>(rows));
    std::vector<std::vector<float>> got(T, std::vector<float>(rows)), got_gu(T, std::vector<float>(rows));
    const void* act[T];
    float *out[T], *out_gu[T];
    const bool scaled = mode == 1;
    const float sg = scaled ? .5f : 1.f, su = scaled ? 1.25f : 1.f, limit = 10.f;
    std::mt19937 rng(29 + n + mode);
    for (int t = 0; t < T; ++t) {
        for (float& v : x[t]) v = mode == 2 ? ((rng() & 1) ? 1.f : -1.f) : (float) ((int) (rng() % 2001) - 1000) / 400.f;
        quant->from_float(x[t].data(), packed[t].data(), n);
        dequant_q8_K(packed[t].data(), x[t].data(), n);
        act[t] = packed[t].data(); out[t] = got[t].data(); out_gu[t] = got_gu[t].data();
        for (int r = 0; r < rows; ++r) {
            float a = 0, b = 0;
            dot->vec_dot(n, &a, 0, weights.data() + stride * r, 0, act[t], 0, 1);
            dot->vec_dot(n, &b, 0, weights.data() + up + stride * r, 0, act[t], 0, 1);
            ref[t][r] = a;
            if (scaled) { a *= sg; b *= su; }
            a = std::fmin(a, limit); b = std::fmax(std::fmin(b, limit), -limit);
            gu[t][r] = (a / (1.f + std::exp(-a))) * b;
        }
    }
    size_t different = 0;
    for (int nt = 1; nt <= T; ++nt) for (int offset = 0; offset + nt <= T; ++offset) {
        cpu::iq_exact_512_rows(22, weights.data(), stride, n, act + offset, nt, out, 0, rows);
        cpu::iq_exact_512_gu_rows(22, weights.data(), stride, up, n, sg, su, scaled, limit, act + offset, nt, out_gu, 0, rows);
        for (int t = 0; t < nt; ++t) for (int r = 0; r < rows; ++r)
            different += std::memcmp(&got[t][r], &ref[t + offset][r], 4) != 0 ||
                         std::memcmp(&got_gu[t][r], &gu[t + offset][r], 4) != 0;
    }
    for (int t = 0; t < T; ++t) {
        std::fill(got[t].begin(), got[t].end(), -12345.f);
        std::fill(got_gu[t].begin(), got_gu[t].end(), -12345.f);
    }
    cpu::iq_exact_512_rows(22, weights.data(), stride, n, act + 2, 3, out, 1, rows - 1);
    cpu::iq_exact_512_gu_rows(22, weights.data(), stride, up, n, sg, su, scaled, limit, act + 2, 3, out_gu, 1, rows - 1);
    for (int t = 0; t < T; ++t) for (int r = 0; r < rows; ++r) {
        const bool inside = t < 3 && r >= 1 && r < rows - 1;
        const float a = inside ? ref[t + 2][r] : -12345.f, b = inside ? gu[t + 2][r] : -12345.f;
        different += std::memcmp(&got[t][r], &a, 4) != 0 || std::memcmp(&got_gu[t][r], &b, 4) != 0;
    }
    double max_abs = 0, max_rel = 0, max_gu_abs = 0, max_gu_rel = 0;
    int bad = 0;
    std::vector<float> w(n), u(n);
    for (int r : {0, 1, rows / 2, rows - 1}) {
        dw->to_float(weights.data() + stride * r, w.data(), n);
        dw->to_float(weights.data() + up + stride * r, u.data(), n);
        for (int t = 0; t < T; ++t) {
            double a = 0, b = 0;
            for (int i = 0; i < n; ++i) { a += (double) w[i] * x[t][i]; b += (double) u[i] * x[t][i]; }
            const double err = std::abs(ref[t][r] - a);
            max_abs = std::max(max_abs, err); max_rel = std::max(max_rel, err / std::max(1e-3, std::abs(a)));
            bad += !std::isfinite(ref[t][r]) || err > 1e-4 + 1e-3 * std::abs(a);
            a = std::min(a * sg, (double) limit); b = std::max(std::min(b * su, (double) limit), -(double) limit);
            const double want = a / (1 + std::exp(-a)) * b, eg = std::abs(gu[t][r] - want);
            max_gu_abs = std::max(max_gu_abs, eg); max_gu_rel = std::max(max_gu_rel, eg / std::max(1e-3, std::abs(want)));
            bad += !std::isfinite(gu[t][r]) || eg > 1e-4 + 1e-3 * std::abs(want);
        }
    }
    std::printf("  exact IQ2_S n=%d rows=%d mode=%d: ggml/width/position differences %zu; vs double dot abs %.3g rel %.3g, "
                "gate/up abs %.3g rel %.3g\n", n, rows, mode, different, max_abs, max_rel, max_gu_abs, max_gu_rel);
    return bad + (different != 0);
}

}  // namespace

int main(int argc, char** argv) {
    bool print = false, bench = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        print |= a == "--print";
        bench |= a == "--bench";
    }
    if (!cpu::cpu_avx512_ok()) {
        std::printf("iq_avx512_test SKIPPED: this CPU lacks the AVX-512 set the kernels need\n");
        return 0;
    }
    std::printf("iq_avx512_test: %s\n", cpu::cpu_name().c_str());
    int bad = 0;
    for (const Frozen& fz : kFrozen) {
        const int type = fz.type;
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt(type, 20, H, FF, f, err)) { std::printf("native_fmt(%d): %s\n", type, err.c_str()); return 1; }
        const ggml_type_traits* tr = ggml_get_type_traits((ggml_type) type);
        uint64_t h = 1469598103934665603ull;
        double max_rel = 0, max_gu = 0;
        size_t tok_diff = 0;
        std::mt19937 rng(11 + (unsigned) type);
        for (int mode = 0; mode < MODES; ++mode) {
            std::vector<uint8_t> blob(2 * f.up_off);
            fill(blob, type, 100 + (uint64_t) type * 7 + (uint64_t) mode, mode);
            std::vector<std::vector<uint8_t>> act(MAXT, std::vector<uint8_t>(cpu::kNativeActBytes));
            std::vector<std::vector<float>> actf(MAXT, std::vector<float>(H));
            std::vector<float> x(H);
            for (int t = 0; t < MAXT; ++t) {
                for (auto& v : x) v = mode == 2 ? ((rng() & 1) ? 1.f : -1.f) : (float) ((int) (rng() % 2001) - 1000) / 400.f;
                cpu::native_quant_act(f, x.data(), act[(size_t) t].data());
                dequant_q8_K(act[(size_t) t].data(), actf[(size_t) t].data(), H);
            }
            // each token alone: the rows every wider call must reproduce
            std::vector<std::vector<float>> one_ff(MAXT, std::vector<float>(FF)), one_d(MAXT, std::vector<float>(FF));
            for (int t = 0; t < MAXT; ++t) {
                const void* a1[1] = {act[(size_t) t].data()};
                float* f1[1] = {one_ff[(size_t) t].data()};
                float* d1[1] = {one_d[(size_t) t].data()};
                cpu::iq512_gu_rows(type, blob.data(), f.gu_row, f.up_off, H, a1, 1, f1, 0, FF);
                cpu::iq512_rows(type, blob.data(), f.gu_row, H, a1, 1, d1, 0, FF);
            }
            for (int nt = 1; nt <= MAXT; ++nt) {
                const void* ap[MAXT];
                std::vector<std::vector<float>> ff(nt, std::vector<float>(FF)), dd(nt, std::vector<float>(FF));
                float *fp[MAXT], *dp[MAXT];
                for (int t = 0; t < nt; ++t) {
                    ap[t] = act[(size_t) t].data();
                    fp[t] = ff[(size_t) t].data();
                    dp[t] = dd[(size_t) t].data();
                }
                cpu::iq512_gu_rows(type, blob.data(), f.gu_row, f.up_off, H, ap, nt, fp, 0, FF);
                cpu::iq512_rows(type, blob.data(), f.gu_row, H, ap, nt, dp, 0, FF);
                for (int t = 0; t < nt; ++t) {
                    fnv(h, dp[t], FF);
                    for (int r = 0; r < FF; ++r)
                        tok_diff += std::memcmp(&fp[t][r], &one_ff[(size_t) t][(size_t) r], 4) != 0 ||
                                    std::memcmp(&dp[t][r], &one_d[(size_t) t][(size_t) r], 4) != 0;
                }
                std::vector<float> wrow(H), wup(H);
                for (int r : {0, 1, 97, FF - 1}) {
                    tr->to_float(blob.data() + (size_t) r * f.gu_row, wrow.data(), H);
                    tr->to_float(blob.data() + f.up_off + (size_t) r * f.gu_row, wup.data(), H);
                    for (int t = 0; t < nt; ++t) {
                        double want = 0, up = 0;
                        for (int i = 0; i < H; ++i) {
                            const double a = (double) actf[(size_t) t][(size_t) i];
                            want += (double) wrow[(size_t) i] * a;
                            up += (double) wup[(size_t) i] * a;
                        }
                        max_rel = std::max(max_rel, std::fabs(dp[t][r] - want) / std::max(1e-3, std::fabs(want)));
                        const double gu = (want / (1.0 + std::exp(-want))) * up;   // silu(gate_r . a) * (up_r . a)
                        max_gu = std::max(max_gu, std::fabs(fp[t][r] - gu) / std::max(1e-3, std::fabs(gu)));
                    }
                }
            }
        }
        const bool hash_ok = h == fz.hash;
        std::printf("  type %2d: hash %016llx %s, token-count dependence %zu words, dot rows vs double: max rel %.3g, "
                    "gate/up rows vs double: max rel %.3g\n",
                    type, (unsigned long long) h, hash_ok ? "(frozen)" : "(DIFFERS from the frozen hash)", tok_diff, max_rel,
                    max_gu);
        if (print) std::printf("    {%d, 0x%016llxull},\n", type, (unsigned long long) h);
        bad += !hash_ok;
        bad += tok_diff != 0;
        if (max_rel > 1e-3) { std::printf("REFERENCE ERROR too large for type %d (dot rows)\n", type); ++bad; }
        if (max_gu > 1e-4) { std::printf("REFERENCE ERROR too large for type %d (gate/up rows)\n", type); ++bad; }
    }
    for (int mode = 0; mode < MODES; ++mode) {
        bad += check_exact_iq2s(256, 33, mode);
        bad += check_exact_iq2s(5120, 2304, mode);
        bad += check_exact_iq2s(2304, 5120, mode);
    }
    std::printf("iq_avx512_test: %s\n", bad ? "FAIL" : "PASS");

    if (bench) {   // 200 gate/up rows (the pool's task) per call, the weights cache-resident: one thread's decode rate
        std::mt19937 rng(5);
        for (const Frozen& fz : kFrozen) {
            const int type = fz.type;
            cpu::NativeFmt f;
            std::string err;
            cpu::native_fmt(type, 20, H, FF, f, err);
            std::vector<uint8_t> blob(2 * f.up_off);
            fill(blob, type, 5, 0);
            std::vector<uint8_t> act(cpu::kNativeActBytes);
            std::vector<float> x(H);
            for (auto& v : x) v = (float) ((int) (rng() % 2001) - 1000) / 400.f;
            cpu::native_quant_act(f, x.data(), act.data());
            for (int nt : {1, 2, 4}) {
                const void* ap[MAXT];
                std::vector<std::vector<float>> ff(nt, std::vector<float>(FF));
                float* fp[MAXT];
                for (int t = 0; t < nt; ++t) { ap[t] = act.data(); fp[t] = ff[(size_t) t].data(); }
                const int reps = 400;
                double best = 1e30;
                for (int k = 0; k < 5; ++k) {
                    const double t0 = now_ms();
                    for (int r = 0; r < reps; ++r) cpu::iq512_gu_rows(type, blob.data(), f.gu_row, f.up_off, H, ap, nt, fp, 0, 200);
                    best = std::min(best, now_ms() - t0);
                }
                std::printf("  bench type %2d nt %d: %.2f us per 200 rows, %.2f GB/s (in cache, one thread)\n", type, nt,
                            best * 1e3 / reps, (double) reps * 400.0 * (double) f.gu_row / (best * 1e6));
            }
        }
    }
    return bad == 0 ? 0 : 1;
}
