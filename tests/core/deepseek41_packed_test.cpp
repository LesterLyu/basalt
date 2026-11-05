#include "basalt/deepseek41/packed_weights.hpp"
#include "basalt/deepseek41/engram.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace ds = basalt::deepseek41;
static int failures = 0;
static void check(bool ok, const char* message) {
    if (!ok) { ++failures; std::printf("FAIL: %s\n", message); }
}
template<class T> static void put(std::vector<uint8_t>& out, T value) {
    const auto* p = reinterpret_cast<const uint8_t*>(&value);
    out.insert(out.end(), p, p + sizeof(T));
}
static void str(std::vector<uint8_t>& out, const std::string& value) {
    put<uint64_t>(out, value.size()); out.insert(out.end(), value.begin(), value.end());
}
static void scalar(std::vector<uint8_t>& out, const std::string& key, uint32_t value) {
    str(out, key); put<uint32_t>(out, 4); put<uint32_t>(out, value);
}
static void array(std::vector<uint8_t>& out, const std::string& key, const std::vector<uint64_t>& value) {
    str(out, key); put<uint32_t>(out, 9); put<uint32_t>(out, 10); put<uint64_t>(out, value.size());
    for (auto x : value) put<uint64_t>(out, x);
}
static void fixture() {
    std::vector<uint8_t> out;
    put<uint32_t>(out, 0x46554747); put<uint32_t>(out, 3); put<uint64_t>(out, 0); put<uint64_t>(out, 11);
    str(out, "general.architecture"); put<uint32_t>(out, 8); str(out, "deepseek41");
    array(out, "deepseek41.ple.layers", {1, 14});
    scalar(out, "deepseek41.ple.ngram_size", 4);
    scalar(out, "deepseek41.ple.heads_per_ngram", 8);
    scalar(out, "deepseek41.embedding_length_per_layer_input", 256);
    scalar(out, "deepseek41.engram.compressed_vocab_size", 100);
    scalar(out, "deepseek41.engram.pad_token_id", 2);
    array(out, "deepseek41.ple.layer_multipliers", {3, 5, 7, 11, 13, 17, 19, 23});
    std::vector<uint64_t> offsets, moduli(48, 101), map;
    for (int i = 0; i < 48; ++i) offsets.push_back(i * 101);
    for (int i = 0; i < 100; ++i) map.push_back(i);
    array(out, "deepseek41.ple.head_offsets", offsets);
    array(out, "deepseek41.ple.head_vocab_sizes", moduli);
    array(out, "deepseek41.engram.token_map", map);
    const auto path = std::filesystem::temp_directory_path() / "deepseek41_engram_test.gguf";
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(out.data()), out.size());
    {
        basalt::GgufFile file(path.string());
        check(file.get("deepseek41.engram.token_map")->items.size() == 64, "normal metadata remains sampled");
        check(file.full_array("deepseek41.engram.token_map", 100).items[99].u == 99, "bounded full-array read");
        bool refused = false;
        try { (void)file.full_array("deepseek41.engram.token_map", 99); } catch (...) { refused = true; }
        check(refused, "full-array allocation limit");
        ds::EngramHasher hash(file);
        auto history = hash.initial_history();
        const auto first = hash.advance(7, history);
        // Python integer arithmetic: (7*3 xor 2*5), then xor 2*7, then xor 2*11; second layer starts 7*13.
        check(first[0] == 31 && first[8] == 808 + 17 && first[16] == 1616 + 7 &&
              first[24] == 2424 + 20 && first[32] == 3232 + 95 && first[40] == 4040 + 12, "Engram golden row ids");
        const auto checkpoint = history;
        const auto branch = hash.advance(9, history);
        history = checkpoint;
        check(hash.advance(9, history) == branch, "Engram speculative rollback");
        auto bad = history;
        refused = false;
        try { (void)hash.advance(100, history); } catch (...) { refused = true; }
        check(refused && history.previous == bad.previous, "invalid token preserves history");
    }
    std::filesystem::remove(path);
}
static void compare_view(const ds::TensorView& a, const ds::TensorView& b) {
    check(a.tensor->shape == b.tensor->shape && a.tensor->type == b.tensor->type && a.bytes == b.bytes,
          "packed tensor shape/type/size");
    const auto n = (std::min<uint64_t>)(a.bytes, 4096);
    check(std::memcmp(a.data(), b.data(), size_t(n)) == 0 &&
          std::memcmp(a.data() + a.bytes - n, b.data() + b.bytes - n, size_t(n)) == 0, "packed tensor boundary bytes");
}
static void real(const char* packed, const char* source, const char* draft) {
    ds::PackedWeights w(packed);
    ds::Weights raw(source, draft);
    check(w.contract().layers == 40 && w.contract().draft_layers == 3, "native packed contract");
    for (const auto& tensor : w.metadata_file().tensors()) compare_view(w.target_tensor(tensor.name), raw.target_tensor(tensor.name));
    for (const auto& tensor : w.draft_metadata_file().tensors()) compare_view(w.draft_tensor(tensor.name), raw.draft_tensor(tensor.name));
    check(w.draft_metadata_file().path() == packed, "DSpark alias maps physical container");
    auto check_experts = [&](const auto& layers, const auto& layout, const auto& file, int experts) {
        std::string error;
        for (size_t l = 0; l < layers.size(); ++l) {
            const auto& e = layers[l];
            std::vector<uint8_t> bytes(size_t(layout.blob_bytes(l)));
            for (int id : {0, experts / 2, experts - 1}) {
                check(basalt::core::read_range_at(file, layout.blob_offset(l, id), bytes.data(), bytes.size(), error), "read packed expert");
                size_t at = 0;
                for (const auto& role : {e.gate, e.up, e.down}) {
                    const size_t count = size_t(role.bytes / experts);
                    check(std::memcmp(bytes.data() + at, role.expert(id), count) == 0, "real packed expert byte identity");
                    at += count;
                }
            }
        }
    };
    check_experts(raw.experts(), w.expert_layout(), w.expert_file(), 384);
    check_experts(raw.draft_experts(), w.draft_expert_layout(), w.draft_expert_ranges().front().source, 128);
    ds::EngramHasher original(raw.metadata_file()), converted(w.metadata_file(), w.token_map());
    auto a = original.initial_history(), b = converted.initial_history();
    for (int token : {0, 42, 128799, 2, 129279, 2031, 1})
        check(original.advance(token, a) == converted.advance(token, b), "real Engram raw/packed row identity");
    std::printf("Packed model: %.6f GiB; %zu target / %zu draft layers, 129280 token-map entries\n",
                w.file().file_bytes() / double(1ull << 30), raw.experts().size(), raw.draft_experts().size());
}
static void xs2(const char* packed, const char* draft) {
    ds::PackedWeights w(packed, draft);
    int widened = 0;
    for (const auto& tensor : w.draft_metadata_file().tensors()) {
        const auto r = w.draft_tensor_range(tensor);
        check(r.source.path == draft, "XS2 dense tensor must use the original binary");
        check(r.source.bytes * (r.widen_bf16 ? 2 : 1) == basalt::tensor_payload_bytes(tensor), "XS2 dense tensor extent");
        widened += int(r.widen_bf16);
    }
    const auto& layout = w.draft_expert_layout();
    auto ranges = w.draft_expert_ranges();
    check(ranges.size() == size_t(layout.n_layers * layout.n_expert * 3), "XS2 expert tensor count");
    std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) { return a.destination < b.destination; });
    uint64_t end = 0;
    for (const auto& r : ranges) {
        check(r.source.path == draft && r.destination == end, "XS2 expert ranges must fill the device slab");
        end += r.source.bytes;
    }
    check(end == layout.total && widened > 0, "XS2 expert total and BF16 norms");
    std::string error;
    for (size_t i : {size_t(0), ranges.size()/2, ranges.size()-1}) {
        std::vector<uint8_t> sample(4096);
        check(basalt::core::read_range_at(ranges[i].source, 0, sample.data(), sample.size(), error), "read XS2 expert");
    }
    std::printf("XS2 direct: %zu dense tensors (%d BF16 widened), %zu expert ranges, %.3f GiB expert slab\n",
                w.draft_metadata_file().tensors().size(), widened, ranges.size(), end / double(1ull << 30));
}
int main(int argc, char** argv) {
    try {
        fixture();
        if (argc == 4) real(argv[1], argv[2], argv[3]);
        else if (argc == 3) xs2(argv[1], argv[2]);
        else if (argc != 1) return 2;
        std::printf("deepseek41_packed_test: %d failures\n", failures);
        return failures ? 1 : 0;
    } catch (const std::exception& e) { std::printf("deepseek41_packed_test: %s\n", e.what()); return 1; }
}
