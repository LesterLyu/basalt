# DeepSeek execution (experimental)

The [DeepSeek pack](../deepseek41.md) preserves raw Unsloth weights. Its main routed experts use IQ1_S and
IQ2_XXS; the embedded DSpark uses MXFP4. The optional Maya XS2 draft keeps its IQ2_XXS experts and
Q8_0 projections; BF16 norms and router weights are widened exactly where the runtime requires F32.
These are distinct from NVFP4 and are not converted to that format.

MXFP4 stores 32 E2M1 values and one E8M0 scale in 17 bytes. The CPU dot uses Q8_0 activations; the GPU dot uses
Q8_1 activations and integer accumulation before fp32 scaling. Byte loads support unaligned block strides.
The dequantizers retain ggml's scale semantics, including subnormal and overflow cases.

DeepSeek SwiGLU clamps the gate above at 10 and the up value to [-10, 10] before activation. The zero-limit
default preserves Qwen's existing path. Routing ranks by sqrt(softplus(logit)) plus selection bias, then
normalizes the unbiased selected scores and scales them by 1.5. Hyper-connections use four streams, fp32
projection/reduction and 20 alternating row/column Sinkhorn normalizations.

## Primitive validation

Development checks: MSVC 14.51, CUDA 13.3; GPU power limits 5090 402.5 W, 5060 Ti 180 W.

- MXFP4: all exponent/nibble combinations match CPU dequantization; nine original DSpark expert tensors,
  sampled at three expert ids and three rows each, pass CPU-double-reference and GPU column-position checks.
- Clamped SwiGLU: maximum absolute error 1.31e-5 and pointwise relative error 3.43e-6 against double evaluation.
  Grouped per-entry, fused and PDL paths agree. Existing unclamped grouped output hash remains `e54c638ab34f4aa6`.
- Real hyper-connection matrices at target layers 0/14/39 and draft layers 0/2 pass double-reference checks:
  projection maximum absolute error 8.71e-6; Sinkhorn 1.04e-7; residual expansion 8.79e-7. Batched columns and
  individual columns agree bitwise, including in-place residual updates. Router tie and weighting checks pass.
- Engram hashing uses the original token map and hash parameters. Copying its three-token history provides
  speculative rollback without changing row ids. No arithmetic precision changes occur during packaging.

## Attention and execution

DSpark's verification length uses confidence calibrated from conditional acceptance observations; positions
after the first rejection are not observations. Calibration resets per request. The controller changes only
the proposed suffix length; target sampling, absolute draw counters, weight formats and kernel arithmetic
are unchanged. Frozen greedy output (18 tokens) and seeded sampling (12 tokens), including window caps
2 through 5, match the single-token target path.

Sliding-window KV uses FP8 E4M3 values with E8M0 scales per 32 values. Compressed KV uses FP4 E2M1 with E4M3
scales per 16; index keys and queries use FP4 E2M1 with E8M0 scales per 32. Adjacent complex RoPE pairs occupy
the final 64 dimensions. Compressed layers use the model's YaRN frequencies. Index scores include the 1/64
head/width factor. Attention applies the learned sink and inverse RoPE before output projection.

- Cache packing/unpacking matches independent CPU packing; RMS/RoPE, ratio-1/2 compression, Engram injection
  and sparse attention pass CPU-double checks on both GPUs. Sparse-attention maximum absolute error is
  `3.5e-7`; Engram with real gate weights is approximately `4e-7`.
- F32/BF16 dense projections, expert combination, masked index scores, candidate selection and stable top-k
  pass reference and window-position checks. Scaled index-score maximum absolute error is `2.27e-7`.
- CPU row scheduling at hidden 5120 / FF 2304, IQ1_S/IQ1_S, IQ1_S/IQ2_XXS and MXFP4/MXFP4, is byte-identical
  to serial native row kernels with 1, 4 and 15 workers, including 11-token repeated-expert groups.

Strided quantized projections gather/scatter a window around the existing exact batched MMVQ path. A real
11-token target window retains byte-identical logits and partial-commit replay. Stride-packing scratch is
20 MiB for the 128-token primary workspace and 1.72 MiB for the unchanged draft workspace.
The resident helper uses the existing compact-copy landing places for eviction. A mixed IQ1_S/IQ2_XXS fixture
checks byte-identical incoming/evicted experts, exclusive ownership and restored free places without file reads.
Adaptive placement can change CPU/GPU rounding; fixed-tier identity does not validate those placement changes.
The no-DSpark primary-tier learner uses the same compact-copy spare places and excludes helper-resident
experts. Its mixed-format exchange fixture checks exact bytes, residency, no file reads and restored places.
With DSpark disabled, three repeated 128-token greedy JSON prefixes retain all 384 token IDs when primary
adaptation is enabled. This short agreement check does not replace reference-model quality validation.

The shared-KV attention path unpacks each query's selected latent vectors once for all heads, retaining FP32
dot products, softmax and their accumulation order. Random real-geometry checks at 1/5/11/32/128 tokens match
the packed kernel byte for byte; the CPU-double maximum absolute error is 6.52e-7. All signed E4M3 code values
also match the independent CPU decoding bits, including signed zero.
Shared-expert projections are queued after routing is read back, allowing overlap with helper work.
Long helper windows run bounded batches on a host task concurrently with the primary, then join before
expert combination or placement changes. Serial/background 29-token checks with a partial final batch
match byte for byte for all four tested expert formats; untouched output rows remain unchanged.

The native prefill workspace admits 128 known tokens; DSpark and helper staging remain bounded independently.
At 262144-token capacity with frozen expert placement, a real 128-token window spanning positions 123..250
matches 11-token windows byte for byte, including partial-commit replay. Warm 512-token A/B/B/A checks retain
identical final target logits and DSpark proposals: mean 17.270 s (11) versus 13.524 s (128), or 29.65/37.86 tok/s,
at the power limits above. Attention reference checks now include 32- and 128-token windows.

Real Unsloth UD-IQ1_S development checks use fixed diagnostic expert tiers, 256-token capacity, and the power
limits above. All target logits for an 11-token window equal individual forwards byte for byte. Positions
123..133 also agree byte for byte across the 128-row ring boundary; partial and zero commits pass rollback.
Eighteen greedy output tokens agree with and without DSpark verification. In that shared-device check the
second GPU completed 239 helper entries and seven DSpark rounds; both device roles were exercised.

The token benchmark protocol reuses the existing CUDA sampler, with top-k/top-p before temperature. DSpark
verification compares proposals with counter-based target draws. Draw counters name output positions and are
advanced only for committed tokens. A fixed-seed 12-token sampled sequence (temperature 0.6, top-k 20, top-p 0.95)
matches individual target steps with DSpark verification enabled. The greedy 18-token identity still passes.
The resident DSpark block uses the existing GPU expert planner and grouped kernels, with unchanged row
ordering. Its Markov picks stay on the device: lowest-ID ties match the previous argmax, and any nonfinite
score still rejects the request. BF16 gathers match CPU and prior GPU conversion bits on both devices.
Three frozen anchors retain identical proposals and confidence values with the device block, picks, or
both enabled; greedy and counter-sampled checks pass. Complete JSON A/B runs retain all 6461 token IDs.

These establish execution/state consistency, not reference-model quality. Full-model Q8_0 perplexity/KLD and
long-context quality remain unvalidated. Short development timings are not production throughput claims.

The optional tensor prefill path uses Qwen's FP16/BF16 dense GEMM and grouped MMQ with bounded asynchronous
staging. It retains FP32 routing, Sinkhorn and KV formats. Dense and expert activation rounding changes; this is
not a bitwise-neutral path. On a 512-token benchmark filler with fixed tiers, teacher-forced comparison against
the quantized legacy path gives PPL 1.60571/1.58529, mean KLD 0.01754, maximum KLD 3.5141 and top-1 502/512.
This limited comparison is not the Q8_0 quality gate, and the tensor path remains disabled by default.

Diagnostic prompt tensor attention splits queries and probabilities into BF16 high/low parts, retains FP32
softmax, and stages exact BF16 latent KV rows. It changes reduction order. Real-geometry CPU-double tests
at 1/5/11/32/128 queries give maximum absolute error 6.13e-6; query positions retain identical output bits.
The bounded long-prompt expert path accumulates routed products in FP64, then rounds the token output to
FP32. Random hidden-5120/top-6 tests match CPU-double within 7.94e-7 after FP32 rounding and give identical
FP32 outputs at row batch widths 1/13/768. These operators are not a full-model Q8_0 quality gate.

FP32 sparse attention normalizes each shared probability once before output accumulation; the frozen
real-geometry fixture retains hash `69b730441d3b9c25`. Packed tensor-attention KV staging uses vector loads
and stores with unchanged value bits. Predecoded output accumulation uses the zero-filled invalid rows
directly instead of re-reading selection indices; packed/predecoded outputs retain identical bits.
Batched ratio-1/2 compression matches sequential compressor output
and history byte for byte, including odd positions and 2048-row windows, on both devices.
Ring commits retain only the final 128 rows through at most two contiguous copies; wrap and partial-prefix
tests match sequential appends byte for byte.
Long prefill seeds DSpark with the final 128 rows across physical tile boundaries. With fixed expert
placement, a 2,112-token prompt produces the same 12 greedy target tokens on two slots with single-token
and speculative verification. This does not imply bitwise agreement across different prompt tiles or tiers.

Grouped Q5_K/Q6_K output projections retain the quantizer, per-row partial sums and output-lane order.
Eight groups at N=1024/K=4096 and partial-row groups at N=17/K=512 match separate launches bitwise
at token widths 1..6 on both devices. Real-weight/random-input CPU-double maximum absolute error
is 1.03e-6 (L1-relative <=2.8e-8).
Pooled asynchronous timing events remain outside graph capture and are queried after existing readback.
Optional front-stage metrics capture external event nodes around decode subregions and query them after
the same readback. Instrumented versus uninstrumented frozen-window logits match byte for byte at
widths 1..6; the timing data includes instrumentation overhead and is not a throughput benchmark.
Metrics and portable pinned transfer buffers preserve frozen-window logits and 128 greedy output tokens.
Selecting every candidate compacts reachable positions in ascending order. Both devices match radix selection
and CPU references byte for byte, including unreachable rows, nonfinite scores and output canaries.
Device expert plans match the sorted host plan exactly at token widths 1..6, including duplicate entries. The static primary-expert graph and mapped host
result path retain frozen-window logits while exercising the original helper's direct-output operation.
Adaptive primary admission can promote a hot peer expert through the existing peer-copy transport; the
peer receives the primary victim. Weight bytes and expert arithmetic are unchanged. Three blob classes
pass twelve promotions across bounded batches with exact incoming/victim bytes, restored RAM landing
places and no file reads. Cold and warm complete JSON generations retain all 6461 greedy token IDs.
Long-prompt admission refreshes the device pointer table before decoding; a forced long-prompt check
reproduces the stale-table failure without that refresh.
Optional helper graph replay reads the current expert pointers and route plan on every launch; shared-format
graphs retain frozen-window logits and partial-commit behavior. The warp-parallel router preserves the
serial router's selected IDs, tie order and FP32 weight-sum order. Frozen routing hashes match on both
devices; CPU-double maximum absolute weight error is 1.19e-7, with unchanged window/position bits.
The diagnostic single-token matrix-attention path batches 64 heads using cuBLAS FP32 BF16x9 emulation,
retaining FP32 softmax and the existing packed KV formats. CPU-double operator maximum absolute error
is 2.45e-7. Different reduction order still changes FP8 rounding: a same-input layer check gives raw error
8.34e-7 but quantized error 0.001953125 in two components; the two-token model check reaches logit error
0.2803032 with graphs disabled as well. This candidate fails model consistency and stays disabled.
Single-token attention splits scores and values from 128 keys, and tiles independent key dots from
256 keys. Each dot, the 256-thread softmax reduction and the FP32 value accumulation retain their order.
Both devices match the prior GPU output byte for byte across 1..640 keys, tile boundaries and invalid
selections; CPU-double maximum absolute error is 2.97e-7. Frozen model windows, partial commits,
long-position logits and 2048 alternating greedy output tokens agree with the previous path.
Smaller rows retain the previous predecoded implementation; batched verification and prefill are unchanged.

Decode retains vocabulary logits on-device. The existing finite-argmax primitive checks every row before
the unchanged target sampler returns IDs and an error flag; public forward and quality calls still return
full logits. Frozen greedy 18-token and seeded 12-token sequences, including verify widths 2..5,
match the single-token target path exactly.

Multi-slot decode has separate sequence state and sampler counters, sharing weights and expert ownership.
CPU jobs from the same layer share a batch; GPU target operations retain their per-sequence row order.
Two-slot greedy/sampled tests compare against frozen single-token draws while reversing request order,
varying verify widths, and resetting/reusing one slot without changing the other's continuation.
Experimental batch staging copies only pinned RAM experts with repeated assignments. It uses the GPU's
grouped expert arithmetic in place of the CPU arithmetic for those rows. The weights, routing rule and
output combination are unchanged. Greedy output can change; the experiment is off by default.

The experimental suffix predictor uses only committed token IDs. Proposed rows use the existing target
sampler and prefix commit; rejected rows neither advance sequence state nor consume target draws.
Adjacent and nonadjacent history matches use the same target verification; consensus can shorten
the proposed prefix. The complete greedy Basalt JSON fixture matches all 6,461 tokens of the
unassisted target in both benchmark rounds. This fixture does not establish sampled chat identity.

Run `mxfp4_parity`, `native_swiglu_parity`, `deepseek41_ops_test`, `deepseek41_attention_test`,
`deepseek41_cpu_test`, `deepseek41_packed_test` and `deepseek41_model_test`.
