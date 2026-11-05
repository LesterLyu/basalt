#pragma once
#include <cstddef>
#include <cstdint>

namespace basalt::deepseek41 {

enum class CacheFormat : int { none = 0, window_fp8 = 1, compressed_fp4 = 2, index_fp4 = 3 };
constexpr size_t cache_row_bytes(int width, CacheFormat format) {
    return format == CacheFormat::window_fp8 ? size_t(width + width / 32) :
           format == CacheFormat::compressed_fp4 ? size_t(width / 2 + width / 16) :
           format == CacheFormat::index_fp4 ? size_t(width / 2 + width / 32) : 0;
}

struct RowFinish {
    const float* norm = nullptr;
    const float* frequencies = nullptr;
    float epsilon = 1e-20f;
    int rotary = 0;
    int first_position = 0;
    int vectors_per_token = 1;
    int position_step = 1;
    bool inverse = false;
    CacheFormat quant = CacheFormat::none;
    const float* rotary_pairs = nullptr; // interleaved sin/cos, beginning at first_position
};

// Normalize, rotate adjacent pairs in the tail, then apply the cache's quantize/dequantize round trip.
// A packed destination preserves input rows; otherwise the rounded values replace input rows in place.
void finish_rows(float* rows, int vectors, int width, size_t stride, const RowFinish& finish,
                 uint8_t* packed, size_t packed_stride, void* stream);
void make_rotary_pairs(float* out,const float* frequencies,int first,int count,bool inverse,void* stream);
void unpack_rows(const uint8_t* packed, int rows, int width, size_t stride, CacheFormat format,
                 float* output, void* stream);
// Only the last 128 appended rows remain in the ring; its wrap needs at most two contiguous copies.
void update_window_ring(uint8_t* ring,const uint8_t* extended,int position,int tokens,size_t row_bytes,void* stream);

struct WindowRollbackLayer {
    uint8_t* ring;
    const uint8_t* extended;
    float *kv_history,*score_history;
    const float *saved_kv,*saved_score,*raw_kv,*raw_gate;
};
struct WindowRollback {
    WindowRollbackLayer layers[40];
};
// Restore rejected rows in a short target window and repair ratio-2 history.
void rollback_window(const WindowRollback& layers,int position,int proposed,int accepted,void* stream);
// Preserve the overwritten ring rows in the start of extended, then append the new rows.
void append_window_snapshot(uint8_t* ring,uint8_t* extended,int position,int tokens,void* stream);

// The ratio-2 history holds unnormalized projections. Only a completed group writes pooled.
// Ratio 1 directly normalizes the current projection. Returns whether pooled was written.
bool compress_step(const float* kv, const float* score, float* kv_history, float* score_history,
                   const float* norm, float* pooled, int width, int ratio, int position, float epsilon, void* stream);
int compress_rows(const float* kv,const float* score,float* kv_history,float* score_history,const float* norm,
                   float* pooled,int width,int ratio,int position,int tokens,float epsilon,void* stream);

void engram_add(float* streams, const float* projected, const float* gate_weight,
                int hidden, int tokens, float epsilon, void* stream);

inline constexpr int kAttentionWindow = 128;
inline constexpr int kPrefillWindow = 2048;
struct AttentionRows {
    int window_start[kAttentionWindow]{};
    int window_count[kAttentionWindow]{};
    int selected_count[kAttentionWindow]{};
    int window_capacity = 0; // zero: contiguous rows; positive: ring modulo this capacity
};
struct PackedRows {
    const uint8_t* data = nullptr;
    size_t stride = 0;
    CacheFormat format = CacheFormat::none;
};
// Selection IDs are relative to compressed; negative entries contribute nothing. The learned sink has zero value.
void sparse_attention(const float* queries, PackedRows window, PackedRows compressed, const int32_t* selection,
                      int selection_stride, const AttentionRows& rows, const float* sinks, int heads, int width,
                      int tokens, float scale, float* output, void* stream);

// Scratch holds eight query selections of 640 FP32 latent vectors, shared across all heads.
inline constexpr size_t kAttentionScratch = size_t(8)*640*512;
void sparse_attention_predecoded(const float* queries,PackedRows window,PackedRows compressed,const int32_t* selection,
                                 int selection_stride,const AttentionRows& rows,const float* sinks,int heads,int width,
                                 int tokens,float scale,float* output,float* scratch,void* stream,bool fixed_width=false,
                                 const uint8_t* ring=nullptr,int ring_start=0);
inline constexpr size_t kDraftAttentionScratch=size_t(128+5)*512;
// All five draft queries see the same contiguous accepted and provisional rows.
void sparse_attention_draft_shared(const float* queries,PackedRows extended,int first,int count,const float* sinks,
                                   float scale,float* output,float* scratch,void* stream);
// Decode packed window or compressed rows to the BF16 inputs used by tensor attention.
void decode_attention_bf16(PackedRows packed,int first,int rows,uint16_t* output,void* stream);

// Diagnostic tensor products split queries and probabilities into BF16 high/low parts; softmax remains FP32.
void sparse_attention_tensor(const float* queries,PackedRows window,PackedRows compressed,const int32_t* selection,
                             int selection_stride,const AttentionRows& rows,const float* sinks,int heads,int width,
                             int tokens,float scale,float* output,void* stream,const uint16_t* decoded=nullptr,
                             const uint16_t* decoded_window=nullptr);
void prepare_attention_matrix(PackedRows window,PackedRows compressed,const int32_t* selection,
                               const AttentionRows& rows,int stride,float* kv,float* kv_transposed,void* stream);
void sparse_attention_split(const float* q,PackedRows window,PackedRows compressed,const int32_t* selection,int selection_stride,
                              const AttentionRows& rows,const float* sinks,float scale,float* out,float* scratch,void* stream);
void softmax_attention_matrix(float* scores,const float* sinks,const int32_t* selection,const AttentionRows& rows,
                               int stride,float scale,void* stream);

}  // namespace basalt::deepseek41
