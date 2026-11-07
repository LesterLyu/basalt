# Windows transport identity

The native loader uses overlapped unbuffered reads for GPU-cache staging and the resident RAM
complement. Its registration retry maps a budgeted prefix of whole experts, leaving the tail
resident without a CUDA alias. These changes leave weight bytes and kernel arithmetic unchanged.

Transport identity check, NVFP4 experts with Q6 dense weights, CUDA 13.3, int8 KV, 262144 context,
MTP 7 with cutoff 0.7: pin the primary tier to 9172 slots, disable adaptive exchanges with
`--adapt-swaps 0`, and use `--pcie-frac 0` to keep CPU miss arithmetic fixed. The previous and updated
Windows binaries produce identical greedy token IDs for conv, struct and prose, 256 tokens each
(768/768 identical). With adaptive exchanges enabled and PCIe share 0.25, the full 7230-token
struct output is identical; conv and prose can diverge as experts move between the existing CPU
and GPU arithmetic paths. Adaptive runs are not an identity test.

This is a transport regression check, not a new PPL/KLD measurement. The Q8_0 base is still
unavailable for the Windows quality gate described above.

## Windows expert peer transport

Portable pinned buffers carry peer expert promotions and long-prefill weights when direct CUDA
P2P is unavailable. A persistent host thread submits the existing peer decode kernel, and
in-flight peer refills reserve their incoming experts against simultaneous primary promotion.
DXGI budgets bound portable host registration. These changes preserve expert bytes, activation
formats, reductions and kernel arithmetic. Different CPU/GPU expert placement can still change rounding.

Validation on Windows with CUDA 13.3:

| Check | Result |
| --- | --- |
| Peer transport, both directions | Byte identity, partial copies, ring reuse, close with outstanding work |
| Host arena lifecycle | Unregistered, partially registered and whole ranges release without stale CUDA errors |
| Fixed-tier greedy decode | 768/768 tokens match the stored full-arena reference |
| Adaptive low-RAM run | Three requests complete, including 16K prefill; zero runtime expert file reads |
| Continuous full-RAM runs | 35 requests complete per single/dual group, including 16K and 64K prefill |

The fixed-tier comparison uses conv, struct and prose, 256 tokens each: primary 8799 live
slots (8800 requested, aligned for elastic KV), peer 3106, prefill 256, spec 0, int8 KV and
adaptive exchanges disabled. Resident experts, peer-exclusive placement and the measured
default PCIe share match the stored reference.

Decode timing also reports input/prefetch, residency upload, peer launch, PLE collection and
cumulative PLE I/O. These host counters change no GPU arithmetic. The checks cover scheduling
and transport; they are not an identity claim across placements or a new PPL/KLD measurement.
