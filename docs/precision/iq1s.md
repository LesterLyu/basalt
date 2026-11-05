# IQ1_S expert primitives

IQ1_S uses the original GGUF blocks: 256 values in 50 bytes. A block has one fp16 scale, eight odd sub-block scales,
four 11-bit grid indices per 32 values, and one signed 1/8 offset per sub-block. Expert bytes are not requantized.

The CPU dot uses Q8_K activations and exact integer sums of `(8 * grid + sign) * activation`, followed by the
block scale divided by eight. The AVX2 and AVX-512 builds use the same reduction. Zero activation blocks return zero
without reading unused activation sums.

The GPU dot uses Q8_1 activations, dp4a integer sums, then fp32 scaling. The per-column and decode-once kernels
evaluate the same expression. The general matrix entry, embedding and prompt dequantizers, and grouped experts
accept IQ1_S. Grouped down projections also accept IQ2_XXS, including mixed IQ1_S/IQ2_XXS experts.

## Primitive validation

Development checks: Ryzen 9950X, MSVC 14.51, CUDA 13.3. GPU power limits: RTX 5090 402.5 W and RTX 5060 Ti 180 W.

- All 2,048 grid entries, both offsets and all eight sub-block scales: GPU fp32 dequantization is byte-identical to
  the CPU dequantizer. Independent NumPy fixtures also give zero dequantization error.
- CPU rows of 256, 2,304 and 5,120 values: maximum absolute error against double accumulation `1.55e-7`;
  maximum relative error `7.03e-8`. AVX2 and AVX-512 results are byte-identical.
- Both GPUs, 2,304- and 5,120-value rows: maximum absolute error against a double dot on the kernels' quantized
  activations `9.05e-7`. Maximum pointwise relative error `1.50e-4` occurs near zero; the check also bounds error
  against the sum of absolute products. Windows of 1, 2, 3, 5, 8 and 11 columns match individual columns byte for byte.
- CPU repeated-expert windows share decoded IQ1_S grids across tokens while preserving each token's fp32 block
  order. Widths 1..12, zero activations and 1/4/15 workers retain frozen hashes `5803e8d96ebe0773` (19/19)
  and `0907b09e3f173959` (19/16) at hidden 5,120 / FF 2,304. Single-token IQ1_S rows use the same block-sum
  kernel; vectorized Q8_K intermediate quantization preserves the scalar quantizer's bits. Direct row checks
  against ggml and CPU double accumulation at widths 1..12 give maximum absolute error `1.28e-6`
  and L1-relative error `1.35e-8`. A 15-worker synthetic three-job window measured
  719 -> 583 us and 648 -> 526 us respectively; these are CPU microbench results, not model throughput.
- Grouped experts at 5,120 hidden / 2,304 FF, all four IQ1_S/IQ2_XXS pairs: per-entry and decode-once paths match
  byte for byte, including scattered destinations and unused output rows.

These validate expert primitives. The [DeepSeek execution checks](deepseek41.md) additionally cover real target
windows, rollback and shared-device DSpark/helper scheduling. End-to-end Q8_0 PPL/KLD remains pending.

Run `iq1s_parity --cpu`, `iq1s_parity [--device ID]`, `iq_parity FIXTURE_DIR` and `iq_multi_parity`.
`iq1s_parity --bench` adds row-dot timings; these small matrices are not end-to-end throughput measurements.

## Prompt MMQ

The IQ1_S tile loader reuses the IQ2_XXS MMQ shared-memory layout and int8 tensor-core dot. It represents each
weight as integer `8 * grid +/- 1` with scale `d * odd_scale / 8`, including the signed eighth offset without
requantizing weights. Activations use MMQ's per-32 FP32 scales (D4), unlike the native dot's fp16 Q8_1 scales.

At N/K 4608/5120 and 5120/2304, widths 1/8/33/128, zero rows, outliers and fallback row tiles pass CPU-double
checks on both GPUs: maximum absolute error 1.24e-6 against the same quantized activations. Relative RMS against
native MMVQ is approximately 2.2e-4. The clamped MMQ SwiGLU maximum absolute error is 1.46e-5 against double.
Run `prefill_mmq_iq1s_test [DEVICE]`. These arithmetic checks do not measure whole-model quality.
