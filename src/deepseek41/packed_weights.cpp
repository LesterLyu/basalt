#include "basalt/deepseek41/packed_weights.hpp"
#include "basalt/ggml/ggml.h"
#include <algorithm>
#include <cstring>
#include <sstream>
#include <string_view>

namespace basalt::deepseek41 {
namespace {
constexpr uint32_t kF32 = 0, kBf16 = 30;
TensorView view(const GgufFile& file, const std::string& name) {
    const auto* t = file.find(name);
    const auto bytes = t ? tensor_payload_bytes(*t) : 0;
    expect(t && bytes && t->offset <= file.file_size() && bytes <= file.file_size() - t->offset,
           "missing or truncated packed tensor " + name);
    return {&file, t, bytes};
}
void parse_layout(const std::string& text, int layers, int experts, kernels::cpu::ExpertLayout& result) {
    std::string err;
    kernels::cpu::ExpertGeometry geometry;
    geometry.n_embd = 5120;
    geometry.n_ff = 2304;
    geometry.buffers = {8192, 4096};
    geometry.swiglu_limit = 10.f;
    expect(kernels::cpu::expert_layout_parse_text(text, layers, experts, result, err, geometry), err);
}
void layout(const core::ModelFile& file, const std::string& prefix, int layers, int experts,
            kernels::cpu::ExpertLayout& result) {
    std::vector<uint8_t> text;
    std::string err;
    expect(file.read(prefix + "experts.layout", text, err), err);
    parse_layout(std::string(text.begin(), text.end()), layers, experts, result);
    expect(file.range(prefix + "experts").size() == result.total, "packed expert size mismatch");
    for (const auto& f : result.fmt) {
        const bool draft = !prefix.empty();
        expect(draft ? f.gu_type == f.d_type && (f.gu_type == GGML_TYPE_MXFP4 || f.gu_type == GGML_TYPE_IQ2_XXS) :
               (f.gu_type == 19 || f.gu_type == 16) && (f.d_type == 19 || f.d_type == 16),
               "unsupported packed expert encoding");
    }
}

constexpr std::pair<std::string_view, std::string_view> xs2_global[] = {
    {"mtp.0.main_norm.weight", "enc.output_norm.weight"}, {"mtp.0.main_proj.weight", "fc.weight"},
    {"mtp.2.confidence_head.proj.weight", "conf_proj.weight"}, {"mtp.2.norm.weight", "output_norm.weight"},
    {"mtp.2.markov_head.embed.weight", "markov_w1.weight"}, {"mtp.2.markov_head.head.weight", "markov_w2.weight"},
};
constexpr std::pair<std::string_view, std::string_view> xs2_layer[] = {
    {"attn.attn_sink", "attn_sinks.weight"}, {"attn.kv_norm.weight", "attn_kv_a_norm.weight"},
    {"attn.q_norm.weight", "attn_q_a_norm.weight"}, {"attn.wkv.weight", "attn_kv.weight"},
    {"attn.wo_a.weight", "attn_output_a.weight"}, {"attn.wo_b.weight", "attn_output_b.weight"},
    {"attn.wq_a.weight", "attn_q_a.weight"}, {"attn.wq_b.weight", "attn_q_b.weight"},
    {"attn_norm.weight", "attn_norm.weight"}, {"ffn.gate.bias", "exp_probs_b.bias"},
    {"ffn.gate.weight", "ffn_gate_inp.weight"}, {"ffn_norm.weight", "ffn_norm.weight"},
    {"ffn.shared_experts.w1.weight", "ffn_gate_shexp.weight"}, {"ffn.shared_experts.w3.weight", "ffn_up_shexp.weight"},
    {"ffn.shared_experts.w2.weight", "ffn_down_shexp.weight"},
    {"hc_attn_base", "hc_attn_base.weight"}, {"hc_attn_fn", "hc_attn_fn.weight"}, {"hc_attn_scale", "hc_attn_scale.weight"},
    {"hc_ffn_base", "hc_ffn_base.weight"}, {"hc_ffn_fn", "hc_ffn_fn.weight"}, {"hc_ffn_scale", "hc_ffn_scale.weight"},
};

uint64_t unsigned_value(const core::Json& v, const std::string& name) {
    expect(v.type == core::Json::Type::Int && !v.neg, "invalid XS2 tensor extent: " + name);
    return v.u;
}
}

void PackedWeights::open_xs2(const std::string& path) {
    constexpr size_t max_manifest = 1 << 20, max_tensors = 2048;
    const core::FileRange manifest(path.substr(0, path.size()-4) + ".json");
    const auto manifest_bytes = manifest.size();
    const auto weights_bytes = core::FileRange(path).size();
    expect(weights_bytes > 0, "missing DSpark XS2 weights: " + path);
    expect(manifest_bytes > 0 && manifest_bytes <= max_manifest, "missing or oversized DSpark XS2 manifest: " + manifest.path);
    std::vector<uint8_t> text;
    core::Json root;
    std::string err;
    expect(core::read_range(core::FileRange(manifest.path, 0, manifest_bytes), text, err), err);
    expect(core::parse_json(reinterpret_cast<const char*>(text.data()), text.size(), root, err, max_tensors), err);
    const auto* entries = root.get("tensors");
    const auto* experts = root.get("experts");
    const auto* dense = root.get("dense");
    expect(experts && experts->s == "IQ2_XXS" && dense && dense->s == "Q8_0" &&
           entries && entries->is_array() && entries->count == entries->items.size(),
           "expected the Maya XS2 IQ2_XXS/Q8_0 manifest: " + manifest.path);
    std::map<std::string, TensorInfo> source;
    std::vector<std::pair<uint64_t,uint64_t>> extents;
    for (const auto& row : entries->items) {
        expect(row.is_array() && row.count == 5 && row.items[0].is_string() && row.items[1].is_string() &&
               row.items[4].is_array() && row.items[4].count > 0 && row.items[4].count <= 4, "invalid XS2 manifest entry");
        const auto& name = row.items[0].s;
        const auto& encoding = row.items[1].s;
        uint32_t type = encoding == "F32" ? kF32 : encoding == "BF16" ? kBf16 :
                        encoding == "Q8_0" ? GGML_TYPE_Q8_0 : encoding == "IQ2_XXS" ? GGML_TYPE_IQ2_XXS : UINT32_MAX;
        TensorInfo t{name, {}, type, unsigned_value(row.items[2], name)};
        for (const auto& dim : row.items[4].items) t.shape.push_back(unsigned_value(dim, name));
        std::reverse(t.shape.begin(), t.shape.end());
        const auto bytes = unsigned_value(row.items[3], name);
        expect(type != UINT32_MAX && bytes && bytes == tensor_payload_bytes(t) && t.offset <= weights_bytes &&
               bytes <= weights_bytes - t.offset && source.emplace(name, t).second, "invalid XS2 tensor: " + name);
        extents.emplace_back(t.offset, t.offset + bytes);
    }
    std::sort(extents.begin(), extents.end());
    for (size_t i=1; i<extents.size(); ++i)
        expect(extents[i-1].second <= extents[i].first, "overlapping XS2 tensor ranges");
    const auto find = [&](const std::string& name) -> const TensorInfo& {
        const auto it = source.find(name);
        expect(it != source.end(), "missing XS2 tensor: " + name);
        return it->second;
    };
    const auto add_dense = [&](const std::string& from, const std::string& to) {
        const auto& t = find(from);
        const auto* target = draft_->find(to);
        const bool widen = target && t.type == kBf16 && target->type == kF32;
        expect(target && t.shape == target->shape && (t.type == target->type || widen), "incompatible XS2 tensor: " + from);
        xs2_tensors_.emplace(to, DraftTensorRange{core::FileRange(path,t.offset,tensor_payload_bytes(t)),widen});
    };
    for (auto [from,to] : xs2_global) add_dense(std::string(from),std::string(to));
    for (int l=0; l<contract_.draft_layers; ++l)
        for (auto [from,to] : xs2_layer)
            add_dense("mtp."+std::to_string(l)+"."+std::string(from), "blk."+std::to_string(l)+"."+std::string(to));
    expect(xs2_tensors_.size() == draft_->tensors().size(), "XS2 does not cover the embedded draft schema");

    constexpr uint64_t hidden=5120, ff=2304;
    const int layers=contract_.draft_layers, count=contract_.draft_experts;
    uint64_t destination=0;
    std::ostringstream layout_text;
    layout_text << "# experts v5: XS2\n";
    for (int l=0; l<layers; ++l) {
        const auto layer_start=destination;
        for (int e=0; e<count; ++e) {
            const auto prefix="mtp."+std::to_string(l)+".ffn.experts."+std::to_string(e)+".";
            for (const std::string_view role : {"w1", "w3", "w2"}) {
                const auto& t = find(prefix+std::string(role)+".weight");
                const std::vector<uint64_t> shape = role == "w2" ? std::vector<uint64_t>{ff,hidden} : std::vector<uint64_t>{hidden,ff};
                expect(t.type == GGML_TYPE_IQ2_XXS && t.shape == shape, "invalid routed XS2 tensor: " + t.name);
                const auto bytes=tensor_payload_bytes(t);
                draft_expert_ranges_.push_back({core::FileRange(path,t.offset,bytes),destination});
                destination+=bytes;
            }
        }
        layout_text << l << ' ' << GGML_TYPE_IQ2_XXS << ' ' << GGML_TYPE_IQ2_XXS << ' '
                    << layer_start << ' ' << (destination-layer_start)/count << '\n';
    }
    parse_layout(layout_text.str(),layers,count,draft_layout_);
    expect(draft_layout_.total == destination, "XS2 expert layout size mismatch");
    std::sort(draft_expert_ranges_.begin(), draft_expert_ranges_.end(),
              [](const auto& a, const auto& b) { return a.source.offset < b.source.offset; });
}

PackedWeights::PackedWeights(const std::string& path, const std::string& draft_path) {
    std::string err;
    expect(file_.open(path, err), err);
    expect(file_.load_meta(err), err);
    expect(file_.register_gguf_view(err), err);
    const bool xs2=draft_path.ends_with(".bin");
    if(!draft_path.empty() && !xs2) {
        external_draft_=std::make_unique<core::ModelFile>();
        expect(external_draft_->open(draft_path,err) && external_draft_->load_meta(err),err);
        const auto* kind=external_draft_->meta().get("architecture");
        expect(kind && kind->is_string() && kind->s=="deepseek41-dspark","invalid external DSpark pack");
    }
    const auto& draft_file=external_draft_?*external_draft_:file_;
    expect(draft_file.register_gguf_view(err, "dspark"), err);
    target_ = std::make_unique<GgufFile>(path);
    draft_ = std::make_unique<GgufFile>(draft_file.path() + "#dspark");
    contract_ = Contract::read(target_->metadata(), draft_->metadata());
    layout(file_, "", contract_.layers, contract_.experts, target_layout_);
    if(xs2) open_xs2(draft_path);
    else {
        layout(draft_file, "dspark/", contract_.draft_layers, contract_.draft_experts, draft_layout_);
        draft_expert_ranges_.push_back({draft_file.range("dspark/experts"),0});
    }
    for (const auto& name : {"token_embd.weight", "output.weight"})
        expect(view(*target_, name).tensor->shape == std::vector<uint64_t>({5120, 129280}), "packed embedding/head geometry");
    const auto engram = view(*target_, "per_layer_token_embd.weight");
    expect(engram.tensor->shape == std::vector<uint64_t>({256, 768022850}) && engram.tensor->type == 20,
           "packed Engram geometry/encoding");
    expect(!draft_->find("token_embd.weight") && !draft_->find("output.weight"), "DSpark must share target embedding/head");
    for (const auto& [name, shape] : std::vector<std::pair<std::string, std::vector<uint64_t>>>{
             {"fc.weight", {15360, 5120}}, {"conf_proj.weight", {5376, 1}},
             {"markov_w1.weight", {256, 129280}}, {"markov_w2.weight", {256, 129280}}})
        expect(view(*draft_, name).tensor->shape == shape, "packed DSpark geometry: " + name);
    expect(file_.range("engram.map").size() == 129280 * sizeof(uint32_t), "packed Engram token map size");
    std::vector<uint8_t> map;
    expect(file_.read("engram.map", map, err), err);
    token_map_.resize(map.size() / sizeof(uint32_t));
    std::memcpy(token_map_.data(), map.data(), map.size());
}

TensorView PackedWeights::target_tensor(const std::string& name) const { return view(*target_, name); }
TensorView PackedWeights::draft_tensor(const std::string& name) const {
    expect(xs2_tensors_.empty(), "XS2 tensors must be read from their source ranges");
    return view(*draft_, name);
}
DraftTensorRange PackedWeights::draft_tensor_range(const TensorInfo& t) const {
    if(!xs2_tensors_.empty()) return xs2_tensors_.at(t.name);
    return {core::FileRange(draft_->path(),draft_->data_start()+t.offset,tensor_payload_bytes(t)),false};
}

}  // namespace basalt::deepseek41
