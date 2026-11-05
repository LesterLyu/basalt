#pragma once
#include "basalt/deepseek41/attention.hpp"

namespace basalt::deepseek41 {

// F32 and BF16 matrices use raw GGUF row-major bytes. Quantized matrices use the existing native MMVQ.
// 64 direct-mapped slots; initialize keys to -1 and keep calls on one stream.
inline constexpr int kMarkovCacheSlots=64;
void cached_markov_add(const uint16_t* matrix,const float* input,const int32_t* previous,
                       int32_t* keys,float* cache,float* logits,void* stream);
void router_linear_tiled(const float* weights,const float* input,float* output,int tokens,void* stream,int rows=4,int expert_tile=1);
void dense_linear(int type, const void* matrix, const float* x, float* y, int inputs, int outputs, int tokens,
                  size_t input_stride, size_t output_stride, void* stream);
void expand_embedding(const float* embedding, float* streams, float* pre_mix, int hidden, int tokens, void* stream);
// The low tokens*top bits identify rows produced on the host or helper GPU.
void combine_experts_masked(const float* gpu,const float* host,const float* weights,const float* shared,
                            float* output,int hidden,int top,int tokens,uint64_t host_rows,void* stream);
void stream_mean(const float* streams, float* destination, int hidden, int tokens,
                  size_t destination_stride, size_t offset, void* stream);
void combine_experts(const float* gpu, const float* cpu, const float* weights, const float* shared,
                      float* output, int hidden, int experts_per_token, int tokens, void* stream);
void add_rows(float* output, const float* addend, size_t count, void* stream);
void gather_bf16_row(const uint16_t* table,const int32_t* row,int rows,int width,float* output,void* stream);
size_t finite_argmax_workspace_bytes(int count);
// Lowest index wins ties; nonfinite values set the shared request error flag.
void finite_argmax(const float* values,int count,int32_t* token,int32_t* invalid,void* workspace,
                   size_t workspace_bytes,void* stream);
// Compact resident entries in expert order, retaining their original output destinations.
void plan_primary_experts(const unsigned long long* lookup,const int32_t* ids,int experts,int top,
                           unsigned long long* pointers,int32_t* starts,int32_t* groups,
                           int32_t* tokens,int32_t* destinations,void* stream,int token_count=1);
// Bounded prompt expert batches accumulate into one row per token instead of six full-sized part matrices.
void seed_experts_f64(const float* shared,double* accumulator,size_t count,void* stream);
void accumulate_experts_f64(const float* parts,const int32_t* entries,const float* weights,double* accumulator,
                            int rows,int hidden,int top,void* stream);
void finish_experts_f64(const double* accumulator,float* output,size_t count,void* stream);

struct IndexRows { int visible[kAttentionWindow]{}; };
void select_topk_rows(const float* scores,int stride,const IndexRows& rows,int tokens,int32_t* output,
                      int output_stride,void* stream);
void candidate_topk_rows(const float* scores,int stride,const IndexRows& rows,int tokens,uint8_t* mask,
                         int mask_stride,void* stream);
void reduce_index_tensor(const float* dots,const float* weights,const uint8_t* candidates,int candidate_stride,
                          float* scores,int score_stride,int tokens,int key_count,int first_key,const IndexRows& rows,void* stream);
// A compressed key is shared by all query heads; optional candidate blocks exclude keys before scoring.
void index_scores(const float* queries, const float* weights, PackedRows keys, const uint8_t* candidates,
                   int candidate_stride, int candidate_block_size, float* scores, int score_stride,
                   int heads, int width, int tokens, int key_count, const IndexRows& rows, void* stream,
                   float weight_scale = 1.f);

size_t topk_workspace_bytes(int count);
// Stable descending score order, then ascending position order for the selected IDs; -inf is unreachable.
void select_topk(const float* scores, int count, int selected, int32_t* input_ids, float* sorted_scores,
                  int32_t* sorted_ids, int32_t* result, void* workspace, size_t workspace_bytes, void* stream);
void candidate_block_scores(const float* scores, int count, int block_size, float* blocks, void* stream);
void candidate_mask(const int32_t* selected, int count, uint8_t* mask, int blocks, void* stream);

}  // namespace basalt::deepseek41
