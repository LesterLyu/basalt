#pragma once
#include "basalt/deepseek41/packed_weights.hpp"
#include "basalt/deepseek41/attention.hpp"
#include <map>
#include <memory>

namespace basalt::deepseek41 {

enum class MemoryKind { weights, state, scratch, draft, long_prefill, mmq, dense_prefill };
struct DeviceTensor {
    const uint8_t* data = nullptr;
    uint32_t type = 0;
    std::vector<uint64_t> shape;
    uint64_t bytes = 0;
};

// Quantized rows stay in their source encoding. The dense slab and work buffers own distinct byte accounts.
class DeviceModel {
public:
    DeviceModel(const PackedWeights& weights, bool draft, int device, int max_window);
    ~DeviceModel();
    DeviceModel(const DeviceModel&) = delete;
    DeviceModel& operator=(const DeviceModel&) = delete;
    const DeviceTensor& tensor(const std::string& name, const std::vector<uint64_t>& expected = {}) const;
    const float* f32(const std::string& name, const std::vector<uint64_t>& expected = {}) const;
    uint8_t* allocate(uint64_t bytes, MemoryKind kind);
    template<class T> T* alloc(size_t count, MemoryKind kind = MemoryKind::scratch) {
        return reinterpret_cast<T*>(allocate(count * sizeof(T), kind));
    }
    void upload(void* destination, const void* source, size_t bytes);
    void copy(void* destination, const void* source, size_t bytes);
    void zero(void* destination, size_t bytes);
    void synchronize() const;
    void* stream() const;
    int device() const;
    std::vector<float> read(const float* source, size_t count) const;
    void read(void* destination, const void* source, size_t bytes) const;
    void load_draft_experts(void* destination, const PackedWeights& weights);
    void prepare_prefill();
    void prepare_decode_attention(float* scratch);
    void decode_attention(const float* q,PackedRows window,PackedRows compressed,const int32_t* selection,
                           const AttentionRows& rows,const float* sinks,float scale,float* out,float* scratch);
    void set_prefill_tensor(bool enabled);
    void linear(const DeviceTensor& tensor, const float* input, float* output, int tokens,
                 size_t input_stride = 0, size_t output_stride = 0);
    void linear(const std::string& name, const float* input, float* output, int tokens,
                 size_t input_stride = 0, size_t output_stride = 0);
    void linear_groups(const DeviceTensor& tensor, const float* input, float* output, int groups,int tokens=1);
    std::map<MemoryKind, uint64_t> memory() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace basalt::deepseek41
