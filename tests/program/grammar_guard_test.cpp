// tests/program/grammar_guard_test.cpp - the pack vocabulary reader and the per-request window check (CPU only).
#include "basalt/program/grammar/guard.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace basalt::program::grammar;

static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

// ids: 0 { 1 } 2 "a" 3 : 4 1 5 True 6 <think> 7 </think> 8 <|im_end|> 9 " x" 10 "\n" 11 <tool_call> 12 hello
static const char* kVocab =
    "{\"{\": 0, \"}\": 1, \"\\\"a\\\"\": 2, \":\": 3, \"1\": 4, \"True\": 5, \"<think>\": 6, \"</think>\": 7, "
    "\"<|im_end|>\": 8, \"\xC4\xA0x\": 9, \"\xC4\x8A\": 10, \"<tool_call>\": 11, \"hello\": 12}";
static const char* kTypes = "[1, 1, 1, 1, 1, 1, 4, 4, 3, 1, 1, 4, 1]";
static const char* kTools =
    R"({"kind":"tools","tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object",)"
    R"("properties":{"flag":{"type":"boolean"}},"required":["flag"]}}}],"tool_choice":"auto"})";

static PackVocab load() {
    PackVocab pv;
    std::string err;
    CHECK(vocab_from_pack(kVocab, kTypes, 14, {8}, pv, err));
    return pv;
}

int main() {
    const PackVocab pv = load();
    CHECK(pv.vocab->bytes(0) == "{");
    CHECK(pv.vocab->bytes(2) == "\"a\"");
    CHECK(pv.vocab->bytes(9) == " x");
    CHECK(pv.vocab->bytes(10) == "\n");
    CHECK(pv.vocab->bytes(8).empty());
    CHECK(pv.vocab->bytes(11) == "<tool_call>");
    CHECK(pv.vocab->bytes(13).empty());
    CHECK(pv.think_open == 6 && pv.think_close == 7);
    {
        PackVocab bad;
        std::string err;
        CHECK(!vocab_from_pack("[1]", kTypes, 14, {}, bad, err));
        CHECK(!vocab_from_pack(kVocab, "{}", 14, {}, bad, err));
    }

    const std::string jo = R"({"kind":"json_object"})";
    std::string err;
    {   // a valid object passes; a refused row is reported and its mask allows the object's opener only
        Guard g(pv);
        CHECK(g.begin(jo, nullptr, 0, err) && g.on());
        const int32_t good[] = {0, 2, 3, 4, 1};
        CHECK(g.scan(good, 5) == -1);
    }
    {
        Guard g(pv);
        CHECK(g.begin(jo, nullptr, 0, err));
        const int32_t bad[] = {12};
        CHECK(g.scan(bad, 1) == 0);
        std::vector<float> bias;
        g.mask_bias({}, bias);
        CHECK(bias.size() == pv.vocab->size());
        CHECK(bias[0] == 0.0f && std::isinf(bias[12]) && std::isinf(bias[8]));
        std::vector<float> user(pv.vocab->size(), 0.0f);
        user[0] = 5.0f;
        g.mask_bias(user, bias);
        CHECK(bias[0] == 5.0f);
        CHECK(g.take(0));
        const int32_t rest[] = {2, 3, 4, 1};
        CHECK(g.scan(rest, 4) == -1);
        CHECK(g.misses() == 1);
    }
    {   // an end token only where the object is complete
        Guard g(pv);
        CHECK(g.begin(jo, nullptr, 0, err));
        const int32_t early[] = {0, 8};
        CHECK(g.scan(early, 2) == 1);
        Guard h(pv);
        CHECK(h.begin(jo, nullptr, 0, err));
        const int32_t done[] = {0, 1, 8};
        CHECK(h.scan(done, 3) == -1);
    }
    {   // a prompt that ends inside a thinking block: free text until the block closes
        Guard g(pv);
        const int64_t prompt[] = {9, 10, 6, 10};
        CHECK(g.begin(jo, prompt, 4, err));
        const int32_t out[] = {12, 12, 7, 0, 1};
        CHECK(g.scan(out, 5) == -1);
        Guard h(pv);
        const int64_t closed[] = {6, 10, 7, 10};
        CHECK(h.begin(jo, closed, 4, err));
        const int32_t first[] = {12};
        CHECK(h.scan(first, 1) == 0);
    }
    {   // tools: free text passes, the opener starts the grammar, then the layout is enforced
        Guard g(pv);
        CHECK(g.begin(kTools, nullptr, 0, err) && g.on());
        const int32_t text[] = {12, 9, 12};
        CHECK(g.scan(text, 3) == -1);
        const int32_t call[] = {11, 12};
        CHECK(g.scan(call, 2) == 1);
        std::vector<float> bias;
        g.mask_bias({}, bias);
        CHECK(bias[10] == 0.0f && std::isinf(bias[12]));
    }
    {   // tool_choice none: no grammar, the guard stays off
        Guard g(pv);
        CHECK(g.begin(R"({"kind":"tools","tools":[{"name":"f","parameters":{}}],"tool_choice":"none"})", nullptr, 0, err));
        CHECK(!g.on());
    }
    {   // interleaved slots share the vocabulary but never the grammar position or mask
        const std::string empty = R"({"kind":"json_schema","schema":{"type":"object","properties":{},"additionalProperties":false}})";
        Guard a(pv), b(pv);
        CHECK(a.begin(jo, nullptr, 0, err));
        CHECK(b.begin(empty, nullptr, 0, err));
        const int32_t open[] = {0};
        CHECK(a.scan(open, 1) == -1 && b.scan(open, 1) == -1);
        const int32_t key[] = {2, 3};
        CHECK(a.scan(key, 2) == -1);
        CHECK(b.scan(key, 2) == 0);
        std::vector<float> bias;
        b.mask_bias({}, bias);
        CHECK(bias[1] == 0.0f && std::isinf(bias[2]));
        CHECK(b.take(1));
        a.mask_bias({}, bias);
        CHECK(bias[4] == 0.0f && std::isinf(bias[1]));
        const int32_t rest[] = {4, 1, 8};
        CHECK(a.scan(rest, 3) == -1);
        CHECK(a.misses() == 0 && b.misses() == 1);
        CHECK(b.begin(jo, nullptr, 0, err));
        b.mask_bias({}, bias);
        CHECK(bias[0] == 0.0f && std::isinf(bias[1]));
    }
    {   // a handover replays only this reply, including the end of its initial thinking block
        Guard g(pv);
        const int64_t prompt[] = {6, 10, 12, 7, 0, 2, 3};
        CHECK(g.begin(jo, prompt, 7, err, 5));
        std::vector<float> bias;
        g.mask_bias({}, bias);
        CHECK(bias[4] == 0.0f && std::isinf(bias[1]) && std::isinf(bias[8]));
        const int32_t rest[] = {4, 1, 8};
        CHECK(g.scan(rest, 3) == -1);
        CHECK(g.windows() == 1 && g.misses() == 0);
        err.clear();
        CHECK(!g.begin(jo, prompt, 7, err, 8) && !g.on());
        err.clear();
        const int64_t bad[] = {6, 7, 0, 12};
        CHECK(!g.begin(jo, bad, 4, err, 2) && !g.on());
        err.clear();
    }
    {
        Guard g(pv);
        CHECK(!g.begin("{", nullptr, 0, err));
    }
    if (g_fail == 0) std::printf("grammar_guard_test: ok\n");
    return g_fail == 0 ? 0 : 1;
}
