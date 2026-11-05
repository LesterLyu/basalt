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
