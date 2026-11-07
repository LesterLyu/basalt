#pragma once

#include "common.hpp"
#include "layout.hpp"
#include "basalt/models/qwen4exp/prefill/prefill.hpp"
#include "basalt/models/qwen4exp/decode/mtp.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/core/peer_copy.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/models/qwen4exp/geometry.hpp"
#include "basalt/models/qwen4exp/decode/layer.hpp"
#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include "basalt/models/qwen4exp/prefill/gemm.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/models/qwen4exp/prefill/moe_fp4.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/prefill/peer_share.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>
#include <memory>

namespace basalt::prefill {
using namespace detail;

// multi-GPU: the peer GPU's share of a prompt chunk's experts, on the shared transport (prefill::PeerShare).  Per MoE
// layer the peer computes the rows routed to the experts it holds, with the products the primary would run on them (MMQ
// from its own slots gathered into group buffers; a streamed chunk the primary's FP4 kernels, read in place), and the
// rows land in the primary's Dm - the rows are laid out local-first, so the peer's are one contiguous block at the end.
struct PeerPrefill {
    core::PeerExperts* peer = nullptr;
    int dev = -1;
    int64_t cap_rows = 0, T_max = 0;
    // a group = up to MMQ_GROUP experts and at most G rows, one launch
    int64_t G = 0;
    // WITHOUT DIRECT ACCESS, a streamed chunk of up to T_fp4 tokens: the primary leaves the experts the peer holds out
    // of its stream, and the peer computes their rows with the primary's FP4 kernels (the same bits) from the layer's
    // token planes, in pieces of at most G rows (fp4::cut_pieces).
    int64_t T_fp4 = 0;
    void* f_a4x = nullptr;                // the layer's token planes and token scales (fp4::quantize_tokens)
    float* f_gsx = nullptr;
    float* f_h = nullptr;                 // a run's H rows
    void* f_a4h = nullptr;                // a run's H planes, token scales, tile table and relative bounds
    float* f_gsh = nullptr;
    int32_t* f_toh = nullptr;
    fp4::Pieces pieces;
    std::unique_ptr<PeerShare> share;
    std::vector<void*> owned;
    ~PeerPrefill() {
        share.reset();
        if (dev < 0) return;
        const core::OnDevice od(dev);
        for (void* p : owned) cudaFree(p);
    }
};

// A peer group buffer: gate/up of up to MMQ_GROUP experts and a zeroed tail, then from this offset their down matrices,
// a zeroed tail and the NVFP4 scale tails (grp_scales).
inline size_t peer_down_at(const MmqPlan& mp) { return al256(MMQ_GROUP * mp.gu_max + MMQ_TAIL); }
// The FP4 share's activations in the share's host buffer: the token planes, then from this offset the token scales.
inline size_t peer_scales_at(int64_t Tp) { return al256(fp4::token_bytes(Tp, N)); }

// Copies a blob from the peer card's memory to the primary through pinned host buffers with only asynchronous calls:
// the peer's device-to-host copies run ahead of the primary's host-to-device ones, and no host thread blocks in a
// synchronous staged peer copy.
struct PeerStage {
    static constexpr int kBufs = 16;
    core::PeerCopy transport;
    bool setup(int peer_dev, size_t blob_bytes, std::string& err) {
        int primary = 0;
        const cudaError_t e = cudaGetDevice(&primary);
        if (e != cudaSuccess) {
            err = std::string("prefill peer stage: ") + cudaGetErrorString(e);
            return false;
        }
        return transport.open(peer_dev, primary, blob_bytes, kBufs, err);
    }
    // queue `src` (on the peer) into `dst` (on the current device) behind `copy_stream`'s earlier work
    void copy(void* dst, const void* src, size_t nbytes, cudaStream_t copy_stream) {
        std::string err;
        if (!transport.copy(dst, src, nbytes, copy_stream, err)) {
            basalt::log::error("basalt", "prefill peer stage: %s", err.c_str());
            std::exit(1);
        }
    }
};

}  // namespace basalt::prefill
