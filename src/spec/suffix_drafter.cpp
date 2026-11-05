// src/spec/suffix_drafter.cpp - see include/basalt/spec/suffix_drafter.hpp.
#include "basalt/spec/suffix_drafter.hpp"

#include <algorithm>

namespace basalt::spec {

namespace {
uint64_t mix(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    return x ^ (x >> 33);
}
}  // namespace

SuffixDrafter::SuffixDrafter(int min_match, int max_match, size_t capacity_tokens,int bigram_width,bool trim_minority,bool skip_pairs)
    : min_match_(std::max(3, min_match)), max_match_(std::max(min_match, max_match)),
      bigram_width_(bigram_width),trim_minority_(trim_minority) {
    size_t cap = 1;
    while (cap < capacity_tokens * 2) cap <<= 1;          // load factor <= 0.5 at the nominal capacity
    table_.assign(cap, Slot{});
    mask_ = cap - 1;
    hist_.reserve(capacity_tokens);
    if(bigram_width_>0)for(size_t i=0;i<(skip_pairs?pairs_.size():1);++i)
        pairs_[i].resize(std::min<size_t>(cap,16384));
}

void SuffixDrafter::reset() {
    hist_.clear();
    std::fill(table_.begin(), table_.end(), Slot{});
    for(auto& pairs:pairs_)std::fill(pairs.begin(),pairs.end(),Slot{});
    last_match_ = 0;
}

uint64_t SuffixDrafter::key_at(size_t end) const {
    const uint64_t a = (uint32_t) hist_[end - 2], b = (uint32_t) hist_[end - 1], c = (uint32_t) hist_[end];
    return mix(a * 0x9E3779B97F4A7C15ull ^ mix(b + 0x632BE59BD9B4E019ull) ^ (c << 1)) | 1ull;   // never 0
}

uint64_t SuffixDrafter::pair_at(size_t end,size_t distance) const {
    return (uint64_t(uint32_t(hist_[end-distance]))<<32) | uint32_t(hist_[end]);
}

int SuffixDrafter::propose_pair(int max_k,int32_t* out) {
    for(size_t i=0;i<pairs_.size();++i) {
        const auto& pairs=pairs_[i];
        if(pairs.empty() || hist_.size()<=i+1)continue;
        const auto key=pair_at(hist_.size()-1,i+1);
        const auto& slot=pairs[mix(key)&(pairs.size()-1)];
        if(slot.key!=key || slot.n<3)continue;
        // Position zero is the current suffix, which has no observed continuation.
        int count=0;
        for(int k=1;k<=std::min(max_k,bigram_width_);++k) {
            int32_t token=0;
            int votes=0;
            for(int i=1;i<slot.n;++i) {
                const size_t pos=size_t(slot.pos[i])+k;
                if(pos>=hist_.size())continue;
                if(votes && hist_[pos]!=token) {votes=0;break;}
                token=hist_[pos];++votes;
            }
            if(votes<2)break;
            out[count++]=token;
        }
        if(count) {last_match_=2;return count;}
    }
    return 0;
}

SuffixDrafter::Slot* SuffixDrafter::find_slot(uint64_t key, bool insert) {
    for (size_t i = key & mask_, probes = 0; probes <= mask_; i = (i + 1) & mask_, ++probes) {
        Slot& s = table_[i];
        if (s.key == key) return &s;
        if (s.key == 0) {
            if (!insert) return nullptr;
            s.key = key;
            return &s;
        }
    }
    return nullptr;                                         // table full: the history outgrew its capacity
}

void SuffixDrafter::append(const int32_t* tokens, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        hist_.push_back(tokens[i]);
        const size_t end = hist_.size() - 1;
        for(size_t j=0;j<pairs_.size() && j<end;++j) {
            auto& pairs=pairs_[j];
            if(pairs.empty())break;
            const auto key=pair_at(end,j+1);
            auto& pair=pairs[mix(key)&(pairs.size()-1)];
            if(pair.key!=key) {pair.key=key;pair.n=0;}
            for(int w=WAYS-1;w>0;--w)pair.pos[w]=pair.pos[w-1];
            pair.pos[0]=uint32_t(end);
            if(pair.n<WAYS)++pair.n;
        }
        if (end < 2) continue;
        Slot* s = find_slot(key_at(end), true);
        if (s == nullptr) continue;
        for (int w = WAYS - 1; w > 0; --w) s->pos[w] = s->pos[w - 1];
        s->pos[0] = (uint32_t) end;
        if (s->n < WAYS) ++s->n;
    }
}

int SuffixDrafter::propose(int max_k, int32_t* out) {
    last_match_ = 0;
    const size_t n = hist_.size();
    if (n < 4 || max_k <= 0) return 0;
    const size_t cur = n - 1;
    const Slot* s = find_slot(key_at(cur), false);
    if (s == nullptr) return propose_pair(max_k,out);
    size_t best_end = 0;
    int best_len = 0;
    for (int w = 0; w < s->n; ++w) {
        const size_t p = s->pos[w];
        if (p >= cur) continue;                             // the current suffix itself
        int len = 0;
        while (len < max_match_ && len <= (int) p && hist_[p - len] == hist_[cur - len]) ++len;
        if (len > best_len) { best_len = len; best_end = p; }   // most recent first, so ties keep the newer
    }
    if (best_len < min_match_)return propose_pair(max_k,out);
    int k = 0;
    // The continuation may run into the current suffix (periodic text); reading history up to `cur` is valid.
    for (size_t q = best_end + 1; q <= cur && k < max_k; ++q) {
        if(trim_minority_) {
            int votes=0,agree=0;
            for(int w=0;w<s->n;++w) {
                const size_t follow=size_t(s->pos[w])+size_t(k)+1;
                if(follow>cur)continue;
                ++votes;
                agree+=hist_[follow]==hist_[q];
            }
            if(votes>=2 && agree*2<votes)break;
        }
        out[k++] = hist_[q];
    }
    if(!k)return propose_pair(max_k,out);
    last_match_=best_len;
    return k;
}

}  // namespace basalt::spec
