#include "basalt/deepseek41/device_model.hpp"
#include "basalt/deepseek41/execution_ops.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/prefill/gemm.hpp"
#include "basalt/prefill/kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>
#include <optional>
#include "profile.hpp"
#include "decode_attention.hpp"

namespace basalt::deepseek41 {
namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(std::string("deepseek41 CUDA: ") + cudaGetErrorString(e)); }
uint64_t aligned(uint64_t n) { return (n + 255) / 256 * 256; }
std::string weights_target(int device, bool draft) {
    return (draft ? "dspark-" : "target-") + core::LoadProgress::cuda(device);
}
void upload_range(DeviceModel& device, void* destination, const core::FileRange& source,
                  std::vector<uint8_t>& buffer, core::LoadProgress& progress, bool widen_bf16 = false) {
    std::string error;
    const size_t ratio = widen_bf16 ? 2 : 1;
    const size_t chunk = widen_bf16 ? buffer.size()/4*2 : buffer.size();
    expect(chunk != 0, "empty tensor upload buffer");
    for (uint64_t at = 0; at < source.bytes;) {
        const size_t count = size_t(std::min<uint64_t>(chunk, source.bytes - at));
        expect(core::read_range_at(source, at, buffer.data(), count, error), error);
        if (widen_bf16) {
            for (size_t i = count/2; i-- > 0;) {
                const uint8_t low = buffer[2*i], high = buffer[2*i+1];
                buffer[4*i] = buffer[4*i+1] = 0;
                buffer[4*i+2] = low;
                buffer[4*i+3] = high;
            }
        }
        device.upload(static_cast<uint8_t*>(destination) + at*ratio, buffer.data(), count*ratio);
        device.synchronize();
        at += count;
        progress.add(count*ratio);
    }
}
}
struct DeviceModel::Impl {
    struct Allocation { void* pointer; uint64_t bytes; MemoryKind kind; };
    int device;
    bool draft;
    cudaStream_t stream = nullptr;
    std::vector<Allocation> allocations;
    std::map<std::string, DeviceTensor> tensors;
    uint8_t* quant = nullptr;
    float *strided_input=nullptr,*strided_output=nullptr;
    std::unique_ptr<prefill::Gemm> gemm;
    uint16_t* tensor_input=nullptr;
    bool tensor_prefill=false;
    Profile linear_profile;
    std::map<const uint8_t*,int> timed_weights;
    int max_window;
    Impl(int ordinal, bool is_draft, int window) : device(ordinal), draft(is_draft), max_window(window) {
        linear_profile.enabled=linear_profile.gpu_enabled=core::exp_flag("ds-linear-timing")!=0;
        int count = 0;
        ck(cudaGetDeviceCount(&count));
        expect(device >= 0 && device < count, "CUDA device unavailable");
        cudaDeviceProp properties{};
        ck(cudaGetDeviceProperties(&properties, device));
        expect(properties.major == 12, "DeepSeek requires a Blackwell CUDA device");
        const core::OnDevice on(device);
        if(!draft && core::exp_flag("ds-prefill-background")!=0) {
            int low=0,high=0;
            ck(cudaDeviceGetStreamPriorityRange(&low,&high));
            ck(cudaStreamCreateWithPriority(&stream,cudaStreamNonBlocking,high));
        } else ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    }
    ~Impl() {
        const core::OnDevice on(device);
        if (stream) cudaStreamSynchronize(stream);
        linear_profile.report();
        gemm.reset();
        for (const auto& allocation : allocations) cudaFree(allocation.pointer);
        if (stream) cudaStreamDestroy(stream);
    }
};
DeviceModel::DeviceModel(const PackedWeights& weights, bool draft, int device, int max_window)
    : impl_(std::make_unique<Impl>(device, draft, max_window)) {
    expect(max_window > 0 && max_window <= kPrefillWindow, "invalid execution window");
    const GgufFile& file = draft ? weights.draft_metadata_file() : weights.metadata_file();
    uint64_t total = 0, payload = 0;
    for (const auto& tensor : file.tensors()) {
        if (tensor.name == "per_layer_token_embd.weight" || tensor.name == "token_embd.weight") continue;
        expect(tensor.type == 0 || tensor.type == 30 || kernels::iq_supported(int(tensor.type)), "unsupported dense encoding " + tensor.name);
        const uint64_t bytes = tensor_payload_bytes(tensor);
        expect(bytes > 0 && bytes < (1ull << 40), "invalid dense tensor " + tensor.name);
        total = aligned(total) + bytes;
        payload += bytes;
    }
    const auto target = weights_target(device, draft);
    core::LoadProgress::plan(target, payload + (draft ? weights.draft_expert_layout().total : 0));
    core::LoadProgress progress(target, payload);
    uint8_t* slab = allocate(aligned(total), draft ? MemoryKind::draft : MemoryKind::weights);
    std::vector<uint8_t> buffer(size_t(std::min<uint64_t>(8 << 20, payload)));
    uint64_t position = 0;
    for (const auto& tensor : file.tensors()) {
        if (tensor.name == "per_layer_token_embd.weight" || tensor.name == "token_embd.weight") continue;
        position = aligned(position);
        const auto bytes = tensor_payload_bytes(tensor);
        const auto range = draft ? weights.draft_tensor_range(tensor) :
            DraftTensorRange{core::FileRange(file.path(), file.data_start() + tensor.offset, bytes),false};
        upload_range(*this, slab + position, range.source, buffer, progress, range.widen_bf16);
        impl_->tensors.emplace(tensor.name, DeviceTensor{slab + position, tensor.type, tensor.shape, bytes});
        position += bytes;
    }
    progress.finish();
    impl_->quant = allocate(size_t(max_window) * 32768 / 32 * 36, draft ? MemoryKind::draft : MemoryKind::scratch);
    impl_->strided_input=alloc<float>(size_t(max_window)*32768);
    impl_->strided_output=alloc<float>(size_t(max_window)*8192);
}
DeviceModel::~DeviceModel() = default;
uint8_t* DeviceModel::allocate(uint64_t bytes, MemoryKind kind) {
    if (!bytes) return nullptr;
    const core::OnDevice on(impl_->device);
    void* p = nullptr;
    ck(cudaMalloc(&p, size_t(bytes)));
    try { impl_->allocations.push_back({p, bytes, kind}); }
    catch (...) { cudaFree(p); throw; }
    return static_cast<uint8_t*>(p);
}
const DeviceTensor& DeviceModel::tensor(const std::string& name, const std::vector<uint64_t>& expected) const {
    const auto it = impl_->tensors.find(name);
    expect(it != impl_->tensors.end(), "missing device tensor " + name);
    expect(expected.empty() || it->second.shape == expected, "device tensor shape " + name);
    return it->second;
}
const float* DeviceModel::f32(const std::string& name, const std::vector<uint64_t>& expected) const {
    const auto& t = tensor(name, expected);
    expect(t.type == 0, "F32 tensor required " + name);
    return reinterpret_cast<const float*>(t.data);
}
int DeviceModel::device() const { return impl_->device; }
void* DeviceModel::stream() const { return impl_->stream; }
void DeviceModel::synchronize() const { const core::OnDevice on(device()); ck(cudaStreamSynchronize(impl_->stream)); }
void DeviceModel::upload(void* dst, const void* src, size_t bytes) {
    const core::OnDevice on(device()); if (bytes) ck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, impl_->stream));
}
void DeviceModel::copy(void* dst, const void* src, size_t bytes) {
    const core::OnDevice on(device()); if (bytes) ck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, impl_->stream));
}
void DeviceModel::zero(void* dst, size_t bytes) {
    const core::OnDevice on(device()); if (bytes) ck(cudaMemsetAsync(dst, 0, bytes, impl_->stream));
}
std::vector<float> DeviceModel::read(const float* p, size_t count) const {
    std::vector<float> data(count); read(data.data(), p, count * sizeof(float)); return data;
}
void DeviceModel::read(void* dst, const void* src, size_t bytes) const {
    const core::OnDevice on(device());
    if (bytes) ck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, impl_->stream));
    synchronize();
}
void DeviceModel::load_draft_experts(void* destination, const PackedWeights& weights) {
    const auto bytes = weights.draft_expert_layout().total;
    std::vector<uint8_t> buffer(size_t(std::min<uint64_t>(8 << 20, bytes)));
    core::LoadProgress progress(weights_target(device(), true), bytes);
    for (const auto& range : weights.draft_expert_ranges())
        upload_range(*this, static_cast<uint8_t*>(destination) + range.destination, range.source, buffer, progress);
    progress.finish();
}
void DeviceModel::linear(const std::string& name, const float* x, float* y, int tokens, size_t xs, size_t ys) {
    linear(tensor(name), x, y, tokens, xs, ys);
}
void DeviceModel::prepare_prefill() {
    const core::OnDevice on(device());
    if(impl_->gemm)return;
    constexpr int64_t elements=32768ll*1280;
    auto* scratch=alloc<uint16_t>(elements,MemoryKind::dense_prefill);
    auto* workspace=allocate(32ull<<20,MemoryKind::dense_prefill);
    impl_->tensor_input=alloc<uint16_t>(size_t(impl_->max_window)*32768,MemoryKind::dense_prefill);
    impl_->gemm=std::make_unique<prefill::Gemm>();std::string error;
    expect(impl_->gemm->init_external(stream(),scratch,elements,workspace,32ull<<20,error),error);
    zero(scratch,512);zero(impl_->tensor_input,512);
    impl_->gemm->f16(impl_->tensor_input,scratch,impl_->strided_output,16,16,16);
    synchronize();
}
void DeviceModel::set_prefill_tensor(bool enabled) {
    expect(!enabled || impl_->gemm!=nullptr,"tensor prefill not prepared");impl_->tensor_prefill=enabled;
}
void DeviceModel::prepare_decode_attention(float* scratch) {
    prepare_prefill();const core::OnDevice on(device());zero(scratch,kAttentionScratch*sizeof(float));
    const AttentionMatrices b(scratch);
    for(int stride=32;stride<=640;stride+=32) {
        impl_->gemm->f32_emulated(b.kv,b.kv,b.scores,64,stride,512);
        impl_->gemm->f32_emulated(b.scores,b.kt,b.temporary,64,512,stride);
    }
    synchronize();
}
void DeviceModel::decode_attention(const float* q,PackedRows w,PackedRows c,const int32_t* ids,
                                   const AttentionRows& rows,const float* sinks,float scale,float* out,float* scratch) {
    const core::OnDevice on(device());expect(bool(impl_->gemm),"decode attention matrix products not prepared");
    attention_matrix(*impl_->gemm,q,w,c,ids,rows,sinks,scale,out,scratch,stream());
}
void DeviceModel::linear_groups(const DeviceTensor& t, const float* x, float* y, int groups,int tokens) {
    const core::OnDevice on(device());
    expect(t.shape.size()==2 && (t.type==13 || t.type==14) && groups>0 && groups<=8 &&
           t.shape[1]%groups==0 && t.shape[0]*groups<=32768 && tokens>0 && tokens<=6 && tokens<=impl_->max_window,"invalid grouped matrix geometry");
    const int in=int(t.shape[0]),out=int(t.shape[1]/groups);
    expect(kernels::native_mmvq_weight_bytes(int(t.type),in,out)*groups==t.bytes,"grouped matrix payload size mismatch");
    kernels::quantize_q8_1_rows(x,groups*tokens,in,impl_->quant,stream());
    kernels::native_mmvq_grouped(int(t.type),t.data,impl_->quant,y,in,out,groups,stream(),tokens);
}
void DeviceModel::linear(const DeviceTensor& t, const float* x, float* y, int tokens, size_t xs, size_t ys) {
    const core::OnDevice on(device());
    expect(t.shape.size() == 2 && t.shape[0] <= 32768 && t.shape[1] <= 262144 &&
           tokens > 0 && tokens <= impl_->max_window, "invalid matrix/window geometry");
    const int in = int(t.shape[0]), out = int(t.shape[1]);
    std::string timing_name;std::optional<GpuProfileScope> timing;
    if(impl_->linear_profile.gpu_enabled && impl_->tensor_prefill && tokens>=16 && impl_->timed_weights[t.data]++==1) {
        timing_name="linear_type"+std::to_string(t.type)+"_N"+std::to_string(out)+"_K"+std::to_string(in)+"_T"+std::to_string(tokens);
        timing.emplace(impl_->linear_profile,timing_name.c_str(),stream());
    }
    const uint64_t row_bytes = t.type == 0 ? uint64_t(in) * 4 : t.type == 30 ? uint64_t(in) * 2 : kernels::iq_row_bytes(int(t.type), in);
    expect(row_bytes && row_bytes * uint64_t(out) == t.bytes, "matrix payload size mismatch");
    if (!xs) xs = in;
    if (!ys) ys = out;
    expect(xs >= size_t(in) && ys >= size_t(out), "invalid matrix strides");
    if(impl_->tensor_prefill && tokens>=16 && t.type!=0) {
        const float* input=x;
        const bool fused=xs!=size_t(in) && t.type!=30 && core::exp_flag("ds-prefill-strided-f16")!=0;
        if(xs!=size_t(in) && !fused) {
            ck(cudaMemcpy2DAsync(impl_->strided_input,size_t(in)*4,x,xs*4,size_t(in)*4,tokens,cudaMemcpyDeviceToDevice,impl_->stream));
            input=impl_->strided_input;
        }
        if(t.type==30) {
            prefill::to_bf16(input,impl_->tensor_input,size_t(tokens)*in,stream());
            impl_->gemm->bf16(impl_->tensor_input,reinterpret_cast<const uint16_t*>(t.data),y,tokens,out,in,ys);
        } else {
            if(fused)prefill::to_f16_strided(x,impl_->tensor_input,tokens,in,xs,stream());
            else prefill::to_f16(input,impl_->tensor_input,size_t(tokens)*in,stream());
            int rows=int(impl_->gemm->scratch_elems()/in);if(rows>=128)rows=rows/128*128;
            expect(rows>0,"tensor prefill weight scratch is too small");
            for(int first=0;first<out;first+=rows)
                impl_->gemm->native(impl_->tensor_input,int(t.type),t.data+uint64_t(first)*row_bytes,y+first,
                                    tokens,std::min(rows,out-first),in,ys);
        }
        return;
    }
    if (t.type == 0 || t.type == 30) dense_linear(int(t.type), t.data, x, y, in, out, tokens, xs, ys, stream());
    else {
        expect(kernels::iq_supported(int(t.type)) && in % 32 == 0, "unsupported quantized matrix");
        const bool packed=tokens>1 && (xs!=size_t(in) || ys!=size_t(out)) && out<=8192;
        const float* input=x;float* output=y;
        if(packed) {
            if(xs!=size_t(in)) {
                ck(cudaMemcpy2DAsync(impl_->strided_input,size_t(in)*4,x,xs*4,size_t(in)*4,tokens,
                                     cudaMemcpyDeviceToDevice,impl_->stream));
                input=impl_->strided_input;
            }
            if(ys!=size_t(out))output=impl_->strided_output;
        }
        if (packed || tokens==1 || (xs == size_t(in) && ys == size_t(out))) {
            kernels::quantize_q8_1_rows(input, tokens, in, impl_->quant, stream());
            for(int first=0;first<tokens;first+=8)
                kernels::native_mmvq(int(t.type), t.data, impl_->quant+size_t(first)*in/32*36,
                                     output+size_t(first)*out, in, out, std::min(8,tokens-first), stream());
            if(output!=y)ck(cudaMemcpy2DAsync(y,ys*4,output,size_t(out)*4,size_t(out)*4,tokens,
                                             cudaMemcpyDeviceToDevice,impl_->stream));
        } else for (int token = 0; token < tokens; ++token) {
            kernels::quantize_q8_1_rows(x + token * xs, 1, in, impl_->quant, stream());
            kernels::native_mmvq(int(t.type), t.data, impl_->quant, y + token * ys, in, out, 1, stream());
        }
    }
}
std::map<MemoryKind, uint64_t> DeviceModel::memory() const {
    std::map<MemoryKind, uint64_t> result;
    for (const auto& a : impl_->allocations) result[a.kind] += a.bytes;
    return result;
}
} // namespace basalt::deepseek41
