// The real serving loop, with a CPU session: cancelled prompt chunks continue in place, including repeated STOP.
#include "../../src/program/serve/session_engine.hpp"
#include "basalt/program/engine_session.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <thread>

namespace {
using namespace basalt;
int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    failures += !ok;
}

class CpuSession final : public model::ModelSession {
    std::vector<int32_t> ids_;
    uint64_t hash_ = 1;
    int64_t tail_ = 0;
    bool pause_, first_ = false, second_ = false;
    void rewind(int64_t pos) {
        ids_.resize((size_t) pos);
        hash_ = 1;
        for (int32_t t : ids_) hash_ = hash_ * 33 + (uint32_t) t;
    }
    void feed(int32_t t) { ids_.push_back(t); hash_ = hash_ * 33 + (uint32_t) t; }
public:
    explicit CpuSession(bool pause) : pause_(pause) {}
    model::Result prefill_chunk(const model::PrefillRequest& r) override {
        if (r.pos == 0) rewind(0);
        if ((int64_t) ids_.size() != r.pos) return {model::Status::Error, "bad resume position"};
        tail_ = r.pos + r.n_tokens;
        for (int off = 0; off < r.n_tokens;) {
            const int end = std::min(off + 4, r.n_tokens);
            while (off < end) feed(r.tokens[off++]);
            if (r.progress) r.progress(off);
            // Estimates from inside a future chunk must never count as committed progress.
            if (r.live) r.live(std::min(off + 2, r.n_tokens));
            bool* waited = ids_.size() == 12 ? &first_ : ids_.size() == 20 ? &second_ : nullptr;
            if (pause_ && waited && !*waited) {
                *waited = true;
                const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                while (r.stop && !r.stop() && std::chrono::steady_clock::now() < limit)
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (r.stop && r.stop()) break;
        }
        return {};
    }
    bool can_continue_prefill(int64_t stop) const override {
        return (int64_t) ids_.size() >= tail_ || stop >= tail_;
    }
    model::Result decode_window(const model::DecodeRequest& r) override {
        if (!can_continue_prefill(r.pos)) return {model::Status::Error, "decoded incomplete layers"};
        rewind(r.pos);
        for (int i = 0; i < r.n_tokens; ++i) {
            feed(r.tokens[i]);
            if (r.next_token) *r.next_token = (int32_t) (hash_ % 997);
            if (r.row_tokens) r.row_tokens[i] = (int32_t) (hash_ % 997);
        }
        return {};
    }
    model::Result set_logit_bias(const float*) override { return {}; }
    size_t running_bytes() const override { return sizeof(hash_); }
    model::Result save_running(uint8_t* dst, int64_t pos) override {
        if (!can_continue_prefill(pos)) return {model::Status::Error, "saved incomplete layers"};
        std::memcpy(dst, &hash_, sizeof(hash_));
        return {};
    }
    model::Result restore_running(const uint8_t* src, int64_t pos) override {
        rewind(pos);
        uint64_t saved = 0;
        std::memcpy(&saved, src, sizeof(saved));
        if (saved != hash_) return {model::Status::Error, "checkpoint hash differs"};
        tail_ = 0;
        return {};
    }
};

class CpuModel final : public model::LoadedModel {
    model::ModelDesc desc_;
    bool pause_;
public:
    explicit CpuModel(bool pause) : pause_(pause) { desc_.arch = "test"; desc_.n_vocab = 1024; }
    const model::ModelDesc& desc() const override { return desc_; }
    model::Result make_session(std::unique_ptr<model::ModelSession>& out) override {
        out = std::make_unique<CpuSession>(pause_);
        return {};
    }
};

class CpuAdapter final : public model::ModelAdapter {
public:
    const char* arch() const override { return "test"; }
    std::string validate(const GgufFile&) const override { return {}; }
    model::ModelDesc describe(const GgufFile&) const override { return {}; }
    model::Result load(const model::LoadRequest&, std::unique_ptr<model::LoadedModel>&) const override { return {}; }
    model::TokenizerSpec tokenizer_spec() const override { return {}; }
    int max_expert_cards() const override { return 0; }
    model::TokenNames token_names() const override { return {}; }
    PreTokenizer pretokenizer() const override { return {}; }
    program::grammar::ToolForm tool_form() const override { return {}; }
};

std::string request(int slot, const std::vector<int64_t>& ids, bool admit = false) {
    std::string s = std::string(admit ? "BGEN " : "SGEN ") + std::to_string(slot) + " 1 ";
    for (int64_t id : ids) { if (s.back() != ' ') s += ','; s += std::to_string(id); }
    return s;
}
struct Reply { int64_t reused = -1, read = -1, token = -1; std::string finish; };
Reply parse(const std::vector<std::string>& lines) {
    Reply r;
    for (const auto& line : lines) {
        if (line.rfind("T ", 0) == 0) r.token = std::stoll(line.substr(2));
        if (line.rfind("DONE ", 0) != 0) continue;
        std::istringstream in(line);
        std::vector<std::string> fields;
        for (std::string f; in >> f;) fields.push_back(f);
        if (fields.size() >= 15) {
            r.finish = fields[5]; r.reused = std::stoll(fields[8]); r.read = std::stoll(fields[14]);
        }
    }
    return r;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--serve") == 0) {
        program::CommonOptions o;
        o.max_context = 128; o.batch = 2; o.spec = 0; o.prompt_cache = 6;
        o.turn_token = 2; o.prompt_cache_root = 4; o.conversation_cache_mib = 1;
        o.conversation_cache_min_tokens = 1;
        core::ModelFile file;
        CpuAdapter adapter;
        CpuModel model(argc > 2 && std::strcmp(argv[2], "pause") == 0);
        return program::serve::run_session(o, file, adapter, model);
    }
    const auto self = std::filesystem::absolute(argv[0]).string();
    const auto log = (std::filesystem::temp_directory_path() / "basalt_prefill_cancel_test.log").string();
    for (const std::string scenario : {"extend", "shorten", "edit"})
    for (bool admit : {false, true}) for (int slot : {0, 1}) {
        program::Engine engine;
        std::string error;
        check(engine.start(self, {"pause"}, log, error), "CPU serving loop starts");
        std::vector<int64_t> prompt;
        for (int i = 0; i < 25; ++i) prompt.push_back(i == 4 ? 2 : 100 + i);
        std::vector<std::string> lines;
        check(engine.command(request(slot, prompt, admit), "PP 12 ", lines, error), "first prefill reaches a complete chunk");
        check(engine.command("STOP", admit ? "BADM " : "DONE ", lines, error), "first prefill cancels");
        auto first = parse(lines);
        check(first.finish == "cancel" && first.reused == 0 && first.read == 12, "first cancel counts committed rows, not estimates");
        prompt.push_back(125);
        check(engine.command(request(slot, prompt, admit), "PP 20 ", lines, error), "second prefill continues to the next chunk");
        check(engine.command("STOP", admit ? "BADM " : "DONE ", lines, error), "second prefill cancels");
        auto second = parse(lines);
        check(second.finish == "cancel" && second.reused == 12 && second.read == 8, "second cancel accumulates reuse without rereading");
        prompt.push_back(126);
        if (scenario == "shorten") prompt.resize(23);
        if (scenario == "edit") prompt[8] = 900;
        check(engine.command(request(slot, prompt), "DONE ", lines, error), "third request completes");
        const auto resumed = parse(lines);
        const int64_t reuse = scenario == "extend" ? 20 : 4;
        check(resumed.reused == reuse && resumed.read == (int64_t) prompt.size() - reuse && resumed.token >= 0,
              scenario == "extend" ? "third request reuses both completed chunks" : "changed prefix or shortened tail uses a full checkpoint");
        uint64_t hash = 1;
        for (int64_t id : prompt) hash = hash * 33 + (uint32_t) id;
        check(resumed.token == (int64_t) (hash % 997), "resumed output equals uninterrupted output");
        engine.stop();
    }
    std::filesystem::remove(log);
    return failures ? 1 : 0;
}
