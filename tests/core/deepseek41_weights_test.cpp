#include "basalt/deepseek41/weights.hpp"
#include <cstdio>
#include <functional>
#include <string>

namespace ds = basalt::deepseek41;
static int failures = 0;
static basalt::MetaValue number(double n) { basalt::MetaValue v; v.type = basalt::MetaType::F64; v.f = n; return v; }
static basalt::MetaValue text(const std::string& s) { basalt::MetaValue v; v.type = basalt::MetaType::STRING; v.s = s; return v; }
static basalt::MetaValue array(const std::vector<int>& a) {
    basalt::MetaValue v; v.type = basalt::MetaType::ARRAY; v.elem = basalt::MetaType::F64; v.count = a.size();
    for (int x : a) v.items.push_back(number(x));
    return v;
}
static ds::Metadata metadata(bool draft) {
    const std::string p = draft ? "dflash." : "deepseek41.";
    ds::Metadata m;
    m["general.architecture"] = text(draft ? "dflash" : "deepseek41");
    for (const auto& [key, value] : std::vector<std::pair<std::string, double>>{
             {"embedding_length",5120}, {"expert_feed_forward_length",2304}, {"block_count",draft ? 3 : 40},
             {"expert_count",draft ? 128 : 384}, {"expert_used_count",draft ? 3 : 6}, {"context_length",1048576},
             {"expert_gating_func",4}, {"expert_weights_norm",1}, {"expert_weights_scale",1.5},
             {"hyper_connection.count",4}, {"hyper_connection.sinkhorn_iterations",20}}) m[p + key] = number(value);
    m[p + "swiglu_clamp_exp"] = array(std::vector<int>(draft ? 3 : 40, 10));
    m[p + "swiglu_clamp_shexp"] = m[p + "swiglu_clamp_exp"];
    if (draft) {
        m[p + "target_layers"] = array({37,38,39});
        m[p + "block_size"] = number(5);
        m[p + "sample_from_anchor"] = number(1);
        m[p + "has_confidence_head"] = number(1);
    } else {
        m[p + "dspark.target_layers"] = array({37,38,39});
        m[p + "dspark.block_size"] = number(5);
        m[p + "dspark.markov_rank"] = number(256);
        m[p + "dspark.noise_token_id"] = number(128799);
        m[p + "dspark.expert_count"] = number(128);
        m[p + "dspark.expert_used_count"] = number(3);
        m[p + "attention.compress_ratios"] = array(std::vector<int>(43, 0));
    }
    return m;
}
static void refuse(const std::string& name, const std::function<void(ds::Metadata&,ds::Metadata&)>& change) {
    auto target = metadata(false), draft = metadata(true);
    change(target, draft);
    bool refused = false;
    try { (void) ds::Contract::read(target, draft); } catch (const std::runtime_error&) { refused = true; }
    if (!refused) { std::printf("FAIL: accepted %s\n", name.c_str()); ++failures; }
}
static int run(int argc, char** argv) {
    const auto c = ds::Contract::read(metadata(false), metadata(true));
    failures += c.hidden != 5120 || c.target_layers != std::vector<int>({37,38,39}) || c.compress_ratios.size() != 40;
    refuse("Qwen architecture", [](auto& a, auto&) { a["general.architecture"] = text("qwen4exp"); });
    refuse("missing field", [](auto& a, auto&) { a.erase("deepseek41.expert_count"); });
    refuse("wrong hidden size", [](auto&, auto& b) { b["dflash.embedding_length"] = number(2560); });
    refuse("fractional layer count", [](auto& a, auto&) { a["deepseek41.block_count"] = number(40.5); });
    refuse("wrong anchors", [](auto&, auto& b) { b["dflash.target_layers"] = array({36,38,39}); });
    refuse("wrong draft block", [](auto&, auto& b) { b["dflash.block_size"] = number(6); });
    refuse("wrong expert count", [](auto& a, auto&) { a["deepseek41.dspark.expert_count"] = number(384); });
    refuse("missing clamps", [](auto& a, auto&) { a["deepseek41.swiglu_clamp_exp"] = array({10}); });
    refuse("changed clamp", [](auto&, auto& b) { b["dflash.swiglu_clamp_exp"].items[1] = number(7); });
    refuse("truncated compression schedule", [](auto& a, auto&) { a["deepseek41.attention.compress_ratios"] = array({0,2}); });
    if (argc >= 3) {
        const basalt::GgufFile target(argv[1]), draft(argv[2]);
        const auto real = ds::Contract::read(target.metadata(), draft.metadata());
        std::printf("Unsloth contract: %d layers, %d experts, top-%d; DSpark %d layers top-%d block-%d\n",
                    real.layers, real.experts, real.top_k, real.draft_layers, real.draft_top_k, real.block_size);
        if (argc == 4 && std::string(argv[3]) == "--weights") {
            ds::Weights weights(argv[1], argv[2]);
            uint64_t bytes = 0;
            for (const auto& l : weights.experts()) bytes += l.gate.bytes + l.up.bytes + l.down.bytes;
            std::printf("Unmodified GGUF views: %zu target / %zu draft expert layers, %.6f GiB target experts\n",
                        weights.experts().size(), weights.draft_experts().size(), bytes / double(1ull << 30));
        }
    } else if (argc != 1) return 2;
    std::printf("deepseek41_weights_test: %d failures\n", failures);
    return failures ? 1 : 0;
}
int main(int argc, char** argv) {
    try { return run(argc, argv); }
    catch (const std::exception& e) { std::printf("deepseek41_weights_test: %s\n", e.what()); return 1; }
}
