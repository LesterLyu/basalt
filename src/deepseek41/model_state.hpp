#pragma once
#include "basalt/deepseek41/model.hpp"
#include "basalt/core/kv_arena.hpp"
#include "basalt/deepseek41/device_model.hpp"
#include "basalt/deepseek41/engram.hpp"
#include "basalt/deepseek41/cpu_experts.hpp"
#include "basalt/deepseek41/execution_ops.hpp"
#include "basalt/deepseek41/ops.hpp"
#include "basalt/deepseek41/draft_window.hpp"
#include "basalt/core/expert_source.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/core/peer_copy.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/ngram/ple_reader.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/log.hpp"
#include <cuda_runtime.h>
#include <map>
#include <array>
#include <optional>
#include <span>
#include "profile.hpp"
#include "prefill_experts.hpp"
#include "update_graph.hpp"
#include "prefill_index.hpp"

namespace basalt::deepseek41 {
struct HostTransferStorage {
    uint8_t* pinned=nullptr;
    uint8_t* mapped=nullptr;
    std::vector<uint8_t> pageable;
    HostTransferStorage(size_t bytes,bool pin,bool map=false);
    ~HostTransferStorage();
    uint8_t* data(){return pinned?pinned:pageable.data();}
};
struct RouteTransport {
    cudaStream_t stream=nullptr;
    cudaEvent_t ready=nullptr;
    RouteTransport();
    ~RouteTransport();
};
inline void cuda_check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(std::string("deepseek41: ") + cudaGetErrorString(e)); }
struct Shape {
    static constexpr int hidden=5120, ff=2304, heads=64, head=512, rotary=64, query_rank=1280;
    static constexpr int streams=4, vocab=129280, window=128, max_tokens=kPrefillWindow, draft_window=11, index_heads=32, index_dim=128;
    static constexpr int index_top=512, candidate_block=8, candidate_top=2048, draft_tokens=5, markov=256;
};
struct Work {
    explicit Work(DeviceModel& model, int context, bool draft, int capacity,bool tensor=false);
    float *h,*x,*qr,*q,*kv,*attn,*out_a,*branch,*mix,*pre,*a_pre,*f_pre,*post,*comb;
    float *gate,*up,*activated,*shared,*router,*weights,*gpu_parts,*cpu_parts,*head_x,*logits;
    float *index_q,*index_weights,*scores,*sorted_scores,*block_scores,*engram_rows,*engram_projected,*anchors;
    uint8_t *quant,*expert_scratch,*engram_raw,*sort_workspace,*candidates;
    int32_t *ids,*sort_ids,*sorted_ids,*selected,*candidate_ids,*group_start,*group_count,*entry_token,*entry_destination;
    unsigned long long* group_pointer;
    float *freq_window,*freq_compressed;
    size_t sort_bytes = 0;
    int score_stride = 0, candidate_stride = 0;
    int score_rows = 0;
    int logit_capacity = 0;
    uint8_t* logit_check=nullptr;
    std::unique_ptr<HostTransferStorage> host_transfer;
    std::span<float> host_x,host_parts;
    float* mapped_parts=nullptr;
    std::span<int32_t> host_ids;
    std::vector<int32_t> host_kind,host_start,host_entry_token,host_entry_destination;
    std::vector<unsigned long long> host_pointer;
    int32_t host_group_count = 0;
};
struct Layer {
    std::unique_ptr<UpdateGraph> front_graph,expert_graph;
    std::array<std::unique_ptr<UpdateGraph>,5> verify_graphs;
    std::array<std::unique_ptr<UpdateGraph>,5> verify_front_graphs;
    std::string prefix;
    int id=0,ratio=0,experts=0,top=0,source=-1;
    const float* image_bias=nullptr;
    bool kv_source=false,index_source=false,engram=false;
    const DeviceTensor *qa,*qb,*kv,*oa,*ob,*router,*shared_gate,*shared_up,*shared_down,*compress_kv=nullptr,
                       *compress_gate=nullptr,*index_q=nullptr,*index_k=nullptr,*index_weights=nullptr,*engram_kv=nullptr;
    const float *attn_norm,*ffn_norm,*q_norm,*kv_norm,*sinks,*bias,*a_fn,*a_scale,*a_base,*f_fn,*f_scale,*f_base,
                *compress_norm=nullptr,*index_norm=nullptr,*engram_gate=nullptr;
    uint8_t *ring=nullptr,*extended=nullptr,*compressed=nullptr,*index_keys=nullptr;
    float *raw_kv=nullptr,*raw_gate=nullptr,*pooled=nullptr,*index_latent=nullptr,*kv_history=nullptr,*score_history=nullptr,
          *saved_kv=nullptr,*saved_score=nullptr;
};
struct LongWork {
    LongWork(DeviceModel& device,int context,int capacity);
    size_t layout(uint8_t* base,int tokens,int end_position);
    float *h=nullptr,*x=nullptr,*pre=nullptr,*post=nullptr,*comb=nullptr,*weights=nullptr;
    double* accumulator=nullptr;
    int32_t *ids=nullptr,*selected=nullptr;
    uint8_t* candidates=nullptr;
    int capacity=0,candidate_stride=0;
    bool borrowed=false;
    float* rotary=nullptr;
    int rotary_first=0,rotary_count=0;
    uint16_t *compressed_bf16=nullptr,*window_bf16=nullptr;
    const uint8_t* compressed_source=nullptr;
    int compressed_count=0;
    bool compressed_active=false;
    std::vector<int32_t> host_ids;
};

struct SequenceState {
    std::array<uint64_t,3> decode_expert_routes{};
    EngramHasher::History history,saved_history;
    std::unique_ptr<Work> main_work,draft_work;
    std::vector<Layer> layers,draft_layers;
    std::unique_ptr<core::KvArena> kv;
    ngram::PleReader::Ticket engram_ticket;
    bool engram_pending=false;
    std::unique_ptr<DraftGpuStages> draft_gpu_stages;
    std::unique_ptr<UpdateGraph> draft_pick_graph,draft_body_graph;
    std::vector<float> host_embedding;
    std::span<const uint16_t> image_rows;
    std::vector<int32_t> pending_tokens;
    std::vector<uint8_t> host_engram;
    std::vector<uint32_t> row_ids;
    int pos=0,base_pos=0,logit_rows=0;
    bool logits_checked=true,pending=false;
    DraftWindow draft_window;
};
struct Model::Impl : SequenceState {
    PackedWeights packed;
    RunOptions options;
    CpuExperts cpu;
    EngramHasher hasher;
    std::unique_ptr<DeviceModel> primary, drafter;
    std::unique_ptr<LongWork> long_work;
    uint64_t borrowed_prefill_bytes=0;
    int64_t prompt_loan_first=INT64_MAX;
    float* grammar_bias_device=nullptr;
    int32_t* penalty_rows_device=nullptr;
    size_t penalty_rows_capacity=0;
    std::vector<int32_t> penalty_rows_host;
    bool cooperative_prefill=false;
    std::function<void()> memory_changed;
    std::unique_ptr<PrefillExperts> prompt_experts;
    std::unique_ptr<PrefillIndex> prompt_index;
    std::unique_ptr<RouteTransport> route_transport;
    std::unique_ptr<HostTransferStorage> host_primary_lookup;
    unsigned long long* primary_lookup=nullptr;
    core::FileExpertSource source;
    core::ExpertCache cache;
    std::vector<uint8_t> elastic_homes;
    core::PeerExperts helper;
    core::PeerCopy to_draft,to_primary;
    ngram::PleReader engram_reader;
    uint8_t* draft_experts=nullptr;
    uint8_t* prefill_experts=nullptr;
    float* attention_scratch=nullptr;
    float* draft_attention_scratch=nullptr;
    float* attention_check_output=nullptr;
    size_t prefill_expert_stride=0;
    bool batch_stage=false;
    uint64_t batch_staged_experts=0,batch_staged_rows=0;
    bool prefill_streaming=false;
    bool prefill_tensor=false;
    bool prefill_active=false;
    int window_capacity=128;
    int32_t* markov_cache_keys=nullptr;
    float* markov_cache=nullptr;
    float *draft_main_x=nullptr,*draft_logits=nullptr,*markov_x=nullptr,*markov_bias=nullptr,*confidence_input=nullptr,*confidence_output=nullptr;
    int32_t* draft_picks=nullptr;
    unsigned long long* draft_lookup=nullptr;
    uint8_t* draft_pick_workspace=nullptr;
    std::vector<float> usage;
    std::vector<int32_t> primary_residency;
    int deferred_adapt_limit=0;
    bool broken=false;
    std::vector<SequenceState> sequences;
    int active_slot=0;
    std::unique_ptr<HostTransferStorage> batch_transfer;
    std::span<float> batch_x,batch_parts;
    std::span<int32_t> batch_ids;
    std::vector<int32_t> batch_kind;
    ExecutionStats execution;
    Profile profile;

    Impl(const std::string& path,RunOptions opts);
    ~Impl();
    Layer make_layer(DeviceModel& device,int id,bool draft);
    void initialize_slots();
    void select_slot(int slot);
    void initialize_cache();
    void initialize_kv(SequenceState& sequence);
    bool fit_kv(int end_position,std::string& error,bool during_prefill=false);
    uint64_t trim_kv(int position,bool replenish=true);
    void regrow_experts(uint64_t released_bytes);
    void embeddings(const std::vector<int32_t>& tokens,DeviceModel& device,Work& work);
    void attention(DeviceModel& device,Work& work,Layer& layer,int p0,int tokens,bool draft);
    void expert_layer(DeviceModel& device,Work& work,Layer& layer,int tokens,bool draft);
    void resident_experts(DeviceModel& device,Work& work,Layer& layer,int tokens);
    void shared_experts(DeviceModel& device,Work& work,Layer& layer,int tokens);
    void batch_experts(const std::vector<int>& slots,const std::vector<int>& counts,int layer);
    void batch_staged_experts_for_slot(Work& work,Layer& layer,int count,const int* staged,int nstaged);
    void issue_experts(DeviceModel& device,Work& work,Layer& layer,int tokens);
    void grouped(DeviceModel& device,Work& work,Layer& layer,int tokens,bool draft);
    void grouped_device(DeviceModel& device,Work& work,Layer& layer,int tokens);
    void refresh_primary_lookup();
    void pause_prefill(const std::function<void()>& decode);
    int primary_slot(int layer,int expert) const {
        const int slot=cache.slot_of(layer,expert);
        return slot<std::min(prompt_loan_first,cache.slots())?slot:core::kNotResident;
    }
    void streamed(DeviceModel& device,Work& work,Layer& layer,int tokens);
    void block(DeviceModel& device,Work& work,Layer& layer,int p0,int tokens,bool draft);
    void block_prefix(DeviceModel& device,Work& work,Layer& layer,int p0,int tokens,bool draft);
    void enqueue_prefix(DeviceModel& device,Work& work,Layer& layer,int p0,int tokens,bool draft);
    void engram(Layer& layer,int tokens,int first_token=0);
    std::optional<ForwardResult> prefill_long(const std::vector<int32_t>& tokens,LogitsMode mode,
                                            const std::function<void()>& between_steps,const std::function<void()>& check_stop);
    void group_output(DeviceModel& device,Work& work,Layer& layer,int tokens);
    ForwardResult forward(const std::vector<int32_t>& tokens,LogitsMode mode,bool copy_logits=true);
    void begin_forward(const std::vector<int32_t>& tokens);
    ForwardResult finish_forward(LogitsMode mode,bool copy_logits);
    void forward_batch(const std::vector<int>& slots,const std::vector<std::vector<int32_t>>& tokens);
    void commit(int accepted,bool overlap_next_draft=false);
    void adapt_after_commit(int primary_limit);
    void finish_deferred_adapt();
    void seed_draft(int position,int count);
    DraftResult draft(int32_t anchor);
    void transfer(DeviceModel& destination,void* dst,DeviceModel& src,const void* from,size_t bytes);
    void reset(bool release_kv=true);
};
} // namespace basalt::deepseek41
