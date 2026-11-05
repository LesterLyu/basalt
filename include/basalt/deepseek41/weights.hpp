// DeepSeek target and DSpark GGUF views; every weight keeps its original encoding.
#pragma once
#include "basalt/artifact/gguf_reader.hpp"
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace basalt::deepseek41 {

using Metadata = std::map<std::string, MetaValue>;

inline const MetaValue& required(const Metadata& m, const std::string& key) {
    const auto it = m.find(key);
    if (it == m.end()) throw std::runtime_error("deepseek41: missing metadata " + key);
    return it->second;
}
inline int integer(const Metadata& m, const std::string& key) {
    const auto& v = required(m, key);
    if (!v.is_num() || v.num() < 0 || v.num() > 2147483647. || std::floor(v.num()) != v.num())
        throw std::runtime_error("deepseek41: invalid integer " + key);
    return int(v.num());
}
inline std::vector<int> integers(const Metadata& m, const std::string& key) {
    const auto& v = required(m, key);
    if (v.type != MetaType::ARRAY || v.count != v.items.size())
        throw std::runtime_error("deepseek41: incomplete array " + key);
    std::vector<int> out;
    for (const auto& x : v.items) {
        if (!x.is_num() || x.num() < 0 || x.num() > 2147483647. || std::floor(x.num()) != x.num())
            throw std::runtime_error("deepseek41: invalid array " + key);
        out.push_back(int(x.num()));
    }
    return out;
}
inline void expect(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error("deepseek41: " + what);
}

struct Contract {
    int hidden = 0, feed_forward = 0, layers = 0, experts = 0, top_k = 0;
    int draft_layers = 0, draft_experts = 0, draft_top_k = 0, block_size = 0, markov_rank = 0;
    int context = 0, noise_token = 0;
    std::vector<int> target_layers, compress_ratios;

    static Contract read(const Metadata& target, const Metadata& draft) {
        expect(required(target, "general.architecture").s == "deepseek41", "target architecture must be deepseek41");
        expect(required(draft, "general.architecture").s == "dflash", "DSpark architecture must be dflash");
        Contract c;
        c.hidden = integer(target, "deepseek41.embedding_length");
        c.feed_forward = integer(target, "deepseek41.expert_feed_forward_length");
        c.layers = integer(target, "deepseek41.block_count");
        c.experts = integer(target, "deepseek41.expert_count");
        c.top_k = integer(target, "deepseek41.expert_used_count");
        expect(c.hidden == 5120 && c.feed_forward == 2304 && c.layers == 40 && c.experts == 384 && c.top_k == 6,
               "unsupported target geometry");
        c.context = integer(target, "deepseek41.context_length");
        c.markov_rank = integer(target, "deepseek41.dspark.markov_rank");
        c.noise_token = integer(target, "deepseek41.dspark.noise_token_id");
        c.block_size = integer(target, "deepseek41.dspark.block_size");
        c.target_layers = integers(target, "deepseek41.dspark.target_layers");
        c.draft_layers = integer(draft, "dflash.block_count");
        c.draft_experts = integer(draft, "dflash.expert_count");
        c.draft_top_k = integer(draft, "dflash.expert_used_count");
        expect(c.draft_layers == 3 && c.draft_experts == 128 && c.draft_top_k == 3 && c.markov_rank == 256 &&
               c.block_size == 5 && c.noise_token == 128799 && c.context > 0, "unsupported DSpark geometry");
        expect(integer(target, "deepseek41.dspark.expert_count") == c.draft_experts &&
               integer(target, "deepseek41.dspark.expert_used_count") == c.draft_top_k, "target/DSpark routing mismatch");
        expect(integer(draft, "dflash.embedding_length") == c.hidden &&
               integer(draft, "dflash.expert_feed_forward_length") == c.feed_forward &&
               integer(draft, "dflash.block_size") == c.block_size &&
               integer(draft, "dflash.context_length") >= c.context, "target/DSpark geometry mismatch");
        expect(c.target_layers == std::vector<int>({37, 38, 39}) &&
               integers(draft, "dflash.target_layers") == c.target_layers, "target/DSpark anchor layers mismatch");
        expect(integer(draft, "dflash.sample_from_anchor") == 1 &&
               integer(draft, "dflash.has_confidence_head") == 1, "unsupported DSpark sampling contract");
        for (const auto& p : {std::string("deepseek41."), std::string("dflash.")}) {
            const auto& m = p == "deepseek41." ? target : draft;
            expect(integer(m, p + "expert_gating_func") == 4 && integer(m, p + "expert_weights_norm") == 1,
                   "unsupported expert routing");
            const auto& scale = required(m, p + "expert_weights_scale");
            expect(scale.is_num() && scale.num() == 1.5, "unsupported expert routing scale");
            expect(integer(m, p + "hyper_connection.count") == 4 &&
                   integer(m, p + "hyper_connection.sinkhorn_iterations") == 20, "unsupported hyper-connections");
            for (const auto& suffix : {"swiglu_clamp_exp", "swiglu_clamp_shexp"}) {
                const auto& a = required(m, p + suffix);
                const int layers = p == "deepseek41." ? c.layers : c.draft_layers;
                expect(a.type == MetaType::ARRAY && a.count == uint64_t(layers) && a.items.size() == size_t(layers),
                       "incomplete SwiGLU clamp array");
                for (const auto& v : a.items) expect(v.is_num() && v.num() == 10., "unsupported SwiGLU clamp");
            }
        }
        c.compress_ratios = integers(target, "deepseek41.attention.compress_ratios");
        expect(c.compress_ratios.size() == size_t(c.layers) ||
               c.compress_ratios.size() == size_t(c.layers + c.draft_layers), "incomplete compression schedule");
        c.compress_ratios.resize(c.layers);
        for (int r : c.compress_ratios) expect(r >= 0 && r <= 2, "unsupported compression ratio");
        return c;
    }
};

struct TensorView {
    const GgufFile* file = nullptr;
    const TensorInfo* tensor = nullptr;
    uint64_t bytes = 0;
    const uint8_t* data() const { return file->tensor_data(*tensor); }
    const uint8_t* expert(int e) const {
        expect(tensor->shape.size() == 3 && e >= 0 && uint64_t(e) < tensor->shape[2], "expert index out of range");
        return data() + uint64_t(e) * (bytes / tensor->shape[2]);
    }
};
struct ExpertWeights { TensorView gate, up, down; };

class Weights {
public:
    Weights(const std::string& target_path, const std::string& dspark_path)
        : target_(GgufModel::open(target_path)), draft_(GgufModel::open(dspark_path)),
          contract_(Contract::read(target_.meta().metadata(), draft_.meta().metadata())) {
        embedding_ = view(target_, "token_embd.weight", {5120, 129280});
        head_ = view(target_, "output.weight", {5120, 129280});
        engram_ = view(target_, "per_layer_token_embd.weight", {256, 768022850});
        expect(engram_.tensor->type == 20, "Engram must retain its IQ4_NL encoding");
        for (int l = 0; l < contract_.layers; ++l) experts_.push_back(expert_views(target_, l, 384, false));
        for (int l = 0; l < contract_.draft_layers; ++l) draft_experts_.push_back(expert_views(draft_, l, 128, true));
        (void) view(draft_, "fc.weight", {15360, 5120});
        (void) view(draft_, "conf_proj.weight", {5376, 1});
        (void) view(draft_, "markov_w1.weight", {256, 129280});
        (void) view(draft_, "markov_w2.weight", {256, 129280});
        expect(!draft_.find("token_embd.weight") && !draft_.find("output.weight"), "DSpark must share target embedding/head");
    }
    const Contract& contract() const { return contract_; }
    const GgufFile& metadata_file() const { return target_.meta(); }
    const std::vector<ExpertWeights>& experts() const { return experts_; }
    const std::vector<ExpertWeights>& draft_experts() const { return draft_experts_; }
    const TensorView& embedding() const { return embedding_; }
    const TensorView& head() const { return head_; }
    const TensorView& engram() const { return engram_; }
    TensorView target_tensor(const std::string& name) const { return view(target_, name); }
    TensorView draft_tensor(const std::string& name) const { return view(draft_, name); }
private:
    static TensorView view(const GgufModel& m, const std::string& name, std::vector<uint64_t> shape = {}) {
        size_t s = 0;
        const auto* t = m.find(name, &s);
        expect(t && m.in_bounds(*t, s), "missing or truncated tensor " + name);
        expect(shape.empty() || t->shape == shape, "tensor shape mismatch " + name);
        return {&m.shard(s), t, tensor_payload_bytes(*t)};
    }
    static ExpertWeights expert_views(const GgufModel& m, int l, int n, bool draft) {
        const auto p = "blk." + std::to_string(l) + ".ffn_";
        ExpertWeights e{view(m, p + "gate_exps.weight", {5120, 2304, uint64_t(n)}),
                        view(m, p + "up_exps.weight", {5120, 2304, uint64_t(n)}),
                        view(m, p + "down_exps.weight", {2304, 5120, uint64_t(n)})};
        expect(e.gate.tensor->type == e.up.tensor->type, "gate/up encoding mismatch");
        for (const auto& v : {e.gate, e.up, e.down})
            expect(draft ? v.tensor->type == 39 : v.tensor->type == 19 || v.tensor->type == 16,
                   "unsupported expert encoding");
        return e;
    }
    GgufModel target_, draft_;
    Contract contract_;
    std::vector<ExpertWeights> experts_, draft_experts_;
    TensorView embedding_, head_, engram_;
};

}  // namespace basalt::deepseek41
