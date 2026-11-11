# RAM and GPU transfer microbench

`transfer_bench` measures transfer submission, completion and byte correctness without a model. It uses the same host VMM
allocator as the engine. Linux and Windows use the same benchmark source; AVX-512 is not required.

Configure the normal build with `-DBUILD_TESTING=ON`, then build only this target:

```sh
cmake --build build --target transfer_bench -j 10
```

Run `build/transfer_bench` on Linux or `build\transfer_bench.exe` on Windows. Multi-configuration generators put it in
`build/Release/`. Output is Markdown; redirect stdout to a report if desired.

```sh
# One GPU; default sizes 1 MiB and 8 MiB, 32 copies, 4 warmups, 2 rounds.
transfer_bench --gpus 0

# All visible GPUs, with simultaneous RAM transfers as well as directed GPU pairs.
transfer_bench

# Reproduce a primary-only host mapping versus portable host memory on a helper GPU.
transfer_bench --gpus 0,1 --host-modes vmm-primary,portable --sizes 8M --rounds 4

# Optional stream gate: does the copy call return while earlier work is still pending?
transfer_bench --gpus 0,1 --host-modes vmm-primary,portable --sizes 8M --gate-ms 20 --no-concurrent

# Larger pinned-capacity test; choose a size that leaves enough RAM for the OS.
transfer_bench --gpus 0,1 --host-modes portable --host-mib 4096 --sizes 8M --rounds 1
```

Host methods:

| Method | Allocation and access |
|---|---|
| `pageable` | Aligned ordinary RAM, unregistered |
| `registered` | Ordinary RAM registered with `cudaHostRegisterPortable | cudaHostRegisterMapped` |
| `pinned` | `cudaHostAllocDefault` |
| `portable` | `cudaHostAllocPortable | cudaHostAllocMapped` |
| `vmm-primary` | Host NUMA VMM; CPU and first selected GPU have explicit access |
| `vmm-all` | Same VMM backing, with explicit access granted to every selected GPU |

All selected contexts exist before host allocation. Unsupported host VMM is **SKIP**, never a substitute allocation.
The full requested host allocation is touched; transfers reuse front buffers rather than cycling over that entire capacity.
Each GPU has two buffers of the largest copy size. A separate portable verification buffer holds one copy.
Concurrent H2D shares a read-only source; concurrent D2H uses separate destinations and GPU-specific patterns.

Reading results:

- **Submit us/copy:** CPU time in the copy API, averaged over the measured loop. Pageable or incompletely mapped host
  endpoints can make an asynchronous copy call block.
- **Stream GB/s:** bytes divided by the CUDA event interval. Includes gaps caused by CPU submission; it is not isolated
  DMA engine time. Events on different GPUs are never subtracted from each other.
- **Wall GB/s:** bytes divided by CPU time through completion. Simultaneous rows use total bytes over one common wall
  interval, including thread wakeup/join; their submit column is the slowest worker's mean. These rows omit stream GB/s.
- **Gate pending:** with `--gate-ms`, a host callback delays the stream before the first event. `yes` means copy submission
  returned before its timer released the gate. The gate delay is included in wall time, excluded from the event interval.
  Gate rows diagnose blocking and should not be compared to ungated wall throughput.
- **Validation:** a fresh address-dependent pattern per case, destination poisoned after warmup, then full byte comparison
  outside timing. This checks final buffer correctness, not a separate readback after every timed copy.

Same-GPU copies use D2D; directed pairs use `cudaMemcpyPeerAsync`, both with peer access disabled and enabled when
supported (`--peer-access off|on|both`). A `no` P2P capability does not prevent measuring the API's fallback copy, and a
successful copy does not prove a physical P2P route. Device-local results may reflect cache residency.

Methods run in reverse order on alternate rounds (two methods and two rounds produce ABBA). GPU-pair rows run once after
host rounds. Run with idle GPUs, report power limits (automatically queried through optional NVML), and compare several
rounds. These measurements do not predict end-to-end model speed. Large allocation failures stop with a nonzero status;
there is no attempt to continue in a potentially failed CUDA context. `--help` lists filters and limits.
