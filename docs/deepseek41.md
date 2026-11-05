# DeepSeek inference (experimental)

The DeepSeek-V4.1-Flash-1017 Unsloth Dynamic V3.1 UD-IQ1_S weights and its Q8_0 DSpark sidecar can be packed
into one `.basalt` v2 container. The native backend supports token-id generation and the `GEN`
benchmark protocol, including multiple active sequences. The production Qwen loading path is unchanged. The HTTP frontend tokenizes DeepSeek text in Python; full-model quality validation is pending.

```sh
python tools/convert/convert.py --gguf DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf \
  --dspark dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf --out DeepSeek.basalt
```

All five target shards must be together. The converter preserves every tensor's original quantized bytes and
the DeepSeek MIT license. Experts are interleaved as gate/up/down using the existing expert layout. DSpark
and the target have separate metadata and expert sections; DSpark shares the target's embedding and output head.
The Engram table remains in its original IQ4_NL encoding, with its token map stored separately as U32.

Conversion uses bounded streaming buffers, requires space for the output beside the sources, and verifies
all section SHA-256 hashes before publishing the output. A failure preserves the previous output. An existing
`.partial` file is refused rather than overwritten. Source GGUF files are retained.

The native reader exposes the target and DSpark views, model-local expert layouts and Engram map directly from
the container. There is no additional expert cache file. Container size does not determine resident RAM usage.

To use the [Maya XS2 draft](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF/tree/main/dspark)
with the existing Unsloth 1017 target, keep `dspark-xs2.bin` and `dspark-xs2.json` together and pass
`--dspark-model /path/to/dspark-xs2.bin`. The engine reads the original files without repacking.
On two GPUs, the helper sizes its expert cache after loading the draft.

## Generation

```sh
basalt --model DeepSeek.basalt --tokens-file prompt.ids --max-context 262144 \
  --gpu 0,1 --max-new 64 --stop-eos
```

`prompt.ids` contains comma- or whitespace-separated IDs, including the model's chat template. The Python
`tools.tokenizer.Tokenizer.from_gguf()` reader supports the source GGUF's `joyai-llm` tokenizer. Native C++ text
tokenization is not connected to this backend yet. Do not send DeepSeek text through the Qwen tokenizer.

The first selected GPU holds target dense weights, compressed KV and its expert tier. DSpark and the existing
`PeerExperts` helper share the second selected GPU. Without a second selected GPU, DSpark uses the main GPU.
DSpark shares the target embedding and head; it keeps its own weights and accepted-prefix KV on its device.

DSpark proposes up to five future tokens. The target verifies the seed and proposals in one window, commits
only the matching greedy prefix, and restores sliding-window KV, compressor history and Engram history for
rejected positions. Helper kernels and pending refills finish before DSpark uses the same GPU. Cross-device
transfers use the existing bounded `PeerCopy` transport. `--spec 0` disables DSpark.

The `--serve` token protocol supports `GEN` / `T` / `DONE` / `STOP` / `QUIT`, temperature, top-k, top-p, min-p and seed.
The HTTP `basalt serve` frontend uses this protocol for DeepSeek models. Sampling reuses Basalt's CUDA sampler; target draws use absolute
output counters, so discarded proposals do not consume draws. DSpark accepts only proposals matching the
target draw. The token protocol has no grammar, penalties or text-tokenization support.

`--slots N` (1..8) shares target/draft weights and expert tiers across independent KV, Engram history,
sampler counters and draft calibration. `--max-context` is the capacity **per slot**; allocate all slot state
before sizing the expert tiers. More slots reduce expert VRAM capacity rather than duplicating expert RAM.
Concurrent decode schedules slots by layer, groups CPU expert jobs across them and overlaps their resident
GPU expert work. Prompt admission uses bounded chunks so existing decoders keep advancing.
Each idle slot can retain its last completed text conversation. `--prompt-cache 0` disables reuse.
With `--conversation-cache-mib N` (default 0), DeepSeek saves a prompt checkpoint just before generation
and can also park the completed prefix when its idle slot is about to be overwritten. A new text prompt
can resume from the longest matching token prefix in an idle slot or in RAM. If replay rewrites the
previous reply, a matching prompt checkpoint still lets the engine skip that prompt. An image in a
cached prefix must match the embedding fingerprint, length, and token position. The engine also saves
the prefix before an image so a replacement can reuse the preceding text. Only image metadata is cached
alongside the state; the image embeddings themselves are not retained. The text between images can use
the long prefill path; image spans are processed separately.
Both types share `--conversation-cache-slots`, `--conversation-cache-min-tokens`, and
`--conversation-cache-min-free-mib`; older entries are evicted to fit the limits. A snapshot contains live KV and index rows,
compressor history, Engram history and DSpark's window. Expert weights are shared as before.
Parked prefixes are lost on restart; the disk tier is not supported.
Tagged requests add `rid=ID` to `GEN`; replies are `T ID token`, `PP ID ...`, `DONE ID ...` or `ERR ID ...`.
`STOP ID` cancels only that request. Untagged `GEN` remains compatible with `basalt-bench`.

### HTTP API

`basalt serve DeepSeek.basalt --alias deepseek-v4.1-flash --gpu 0,1 --slots 2 --spec 6` uses
the bundled DeepSeek reference codec and the Basalt HTTP server. Text and DSML tool calls use the standard
Responses stream. This backend cannot enforce JSON grammars.

With `basalt-deepseek-vision` next to the engine, add `--vision 1 --deepseek-mmproj PATH` for the matching
DeepSeek 1017 `deepseek41v` projector (F16 or BF16). The sidecar keeps its weights in host memory and
allocates its GPU workspace only while encoding an image. Leave room for it in the selected GPU's reserve.
Responses accepts inline base64 images. Large images use an overview plus detail tiles in row-major order, with
up to 8,192 tokens per uploaded image. Set `--vision-max-tokens 8192` to use one full image instead,
with BF16 tensor-core attention and an FP32 online softmax. The full image keeps cross-region attention;
its attention work grows quadratically with the patch count. Images and text share the request's context budget. The engine maps
the embeddings file and releases it after prefill. The browser exposes upload when
`/health` advertises images. Each slot consumes its
own embeddings, uses the visual routing bias and resets Engram history across image spans.
The monitor and `/metrics` report tracked VRAM allocations per GPU, including the Vision workspace
while encoding. A borrowed long-prompt workspace replaces its expert-cache bytes in this breakdown.
The difference from the sampled device usage is shown as unattributed; it is not a per-app reading.

`basalt-bench` needs fixtures encoded with this model's tokenizer and chat template, with thinking disabled;
the bundled Qwen token IDs cannot be used. Use `--fixtures DIR --no-windows` and explicitly select the matching
experimental `--engine` binary. The Qwen JSON fixture's fixed token-count check does not apply to other tokenizers.

The HTTP server enables token-level grammar by default for DSML tools and structured JSON. A strict
Responses request with tools accepts either a valid DSML call or a JSON answer matching `text.format`.
Required and named tool choices are enforced from the start of the answer. Each slot checks its own
DSpark output before publishing tokens. Tool arguments and Responses JSON are validated before being
sent as complete output; install `jsonschema` for schema validation. Schema features that the token
grammar cannot express are checked after generation, and invalid output is returned as an error.

Text-only API requests use the native JoyAI tokenizer, including with multiple slots. Image prompts and
continuations that append already generated token IDs use the token-ID path. Sampling supports repetition,
presence, and frequency penalties over the last 64 tokens by default; `penalty_last_n` accepts 0–4096.
DSpark target verification and grammar resampling use the same history for each row.

Set `reasoning.effort` (Responses), `reasoning_effort` (Chat Completions), or `output_config.effort`
(Messages) to an integer from 0 to 100. Zero disables thinking; 1–100 is the exact model effort, not
a token limit. The default is 75. Named levels are low 50, medium 62, high 75 and max/xhigh 100.

Use `--vram-elastic` to reserve a base of up to 16K tokens per slot and back more compressed KV and index pages as its context grows.
A live prefix stays mapped for reuse. Other slots can use released pages before the expert cache grows back.
The resident expert copy, runtime KV growth and prefill loans use a 4 GiB free-RAM floor.
The rolling KV, DSpark's short window and prefill scratch are independent of these pages.

Known prompts use `prefill()`. The default 128-token path groups CPU experts and retains the exact MMVQ
projections. An experimental tensor path supports up to 2048-token chunks using the existing Qwen GEMM/MMQ
components, an IQ1_S tile loader and a bounded asynchronous copy ring. The optional long-prompt path
keeps one logical chunk of up to 32768 tokens, scheduling attention in bounded tiles and grouping expert
members across the whole chunk. Its persistent GPU scratch reduces the space available for expert slots.
Quantized dense products use FP16
inputs/weights, native BF16 matrices use BF16, and MMQ experts retain their raw quantized blocks.
This changes rounding and remains disabled by default pending full-model reference-quality validation.
To select the tuned 32K path with two slots and DSpark:

```sh
basalt --model DeepSeek.basalt --serve --gpu 0,1 --spec 6 --slots 2 --max-context 524288 \
  --prefill 32768 --peer-reserve-mib 1024 --exp ds-prefill-32k-tuned=1
```

The server uses up to 32,768 tokens per text chunk when only one request is active. Multiple slots use
512-token physical tiles to limit the workspace. Image prompts retain the 512-token chunk limit.
DSpark seeds its sliding-window KV from the last 128 tokens of each long chunk.
Add `ds-prefill-borrow=1` to borrow target expert slots for the long workspace instead of allocating it
at startup. The engine copies affected experts to temporary RAM, restores them when the loan ends, and
falls back to physical tiles if there is not enough safe RAM to borrow.
Add `ds-prefill-cooperative=1` to let an already decoding slot advance between safe steps of a long
text prefill. The borrowed slots are removed from decode routing until restored, and expert placement
stays fixed during the loan. GPU work is interleaved on one execution owner. Borrowing beside decode
is capped to a quarter of the main expert arena and shrinks further when RAM is tight. Without this
switch, concurrent prompts keep the 512-token chunk limit.

With cooperative prefill enabled, `ds-prefill-background=1` runs each layer's expert matrix work
on a separate CPU worker and a lower-priority CUDA stream so the other slot can decode during
that work. Attention, expert output merging and DSpark seeding still share the execution owner.
The option is off by default.

Attention queries and index scores use small tiles rather than a chunk-by-context score allocation.
Selected latent KV rows are unpacked once per small query tile and shared across heads, preserving FP32
attention arithmetic. This bounded workspace uses 10 MiB on the target device.
Decode verification remains at most six tokens. DSpark seeding and helper computation fit their existing buffers.
Decode output projection batches eight independent Q5_K/Q6_K groups across up to six token rows into
one quantization and one matrix launch, preserving existing arithmetic without additional model copies.
For larger windows, helper batches run concurrently with primary processing and complete before expert
combination; the shared expert projections also overlap with the independent helper work.

GPU expert tiers are sized from actual free VRAM. Experts absent from both tiers use the existing compact RAM
complement, with a guarded file-mapping fallback when it cannot fit. `--mmap-experts` opts out of that RAM copy.
`--vram-reserve-mib` and `--peer-reserve-mib` retain the requested free VRAM on their respective cards.
The primary tier learns routed-expert heat between forwards, including DSpark verification. It exchanges
cold slots with hot RAM experts through existing spare places, excluding experts owned by the helper.
Decode admission is bounded per accepted token; prefill uses a separate bounded admission budget.
Engram uses bounded direct row reads and the existing `--ple-row-cache` / `--ple-inflight` settings; it is not
loaded as a full RAM table. The explicit Qwen full-resident mirror mode is not implemented for DeepSeek.

## Validation

`deepseek41_model_test MODEL window` compares 11-token target windows with individual token forwards;
`boundary` tests ring wrap and partial/zero commits; `draft` checks shared-device helper execution and DSpark
greedy identity. These tests require the real pack and use small diagnostic expert tiers, not production sizing.
`deepseek41_slots_test MODEL` checks two-slot greedy/sampled identity, reversed scheduling, mixed verify
widths and resetting/reusing one slot while the other continues.
They do not replace the pending Q8_0 perplexity/KLD gate. See [precision](precision/deepseek41.md).
