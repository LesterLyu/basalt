#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <stdexcept>

namespace basalt::deepseek41 {
// Confidence predicts conditional acceptance; observations after the first rejection are not target draws.
struct DraftWindow {
    std::array<double,16> seen{},matched{};
    static int bin(float probability) {
        if(!std::isfinite(probability) || probability<0 || probability>1)
            throw std::invalid_argument("invalid draft confidence");
        return std::min(15,int(probability*16));
    }
    int choose(std::span<const float> probabilities,int limit) const {
        if(limit<0 || limit>int(probabilities.size()))throw std::invalid_argument("invalid draft limit");
        double joint=1;
        int kept=0;
        for(int j=0;j<limit;++j) {
            const float p=probabilities[j];const int b=bin(p);
            const double calibrated=(matched[b]+4.*p)/(seen[b]+4.);
            joint*=calibrated;
            // Keep one proposal to observe whether a low-confidence first token improves.
            if(j && joint<0.35)break;
            ++kept;
        }
        return kept;
    }
    void observe(std::span<const float> probabilities,int offered,int accepted) {
        if(offered<0 || offered>int(probabilities.size()) || accepted<0 || accepted>offered)
            throw std::invalid_argument("invalid draft observation");
        for(std::size_t b=0;b<seen.size();++b) {seen[b]*=0.97;matched[b]*=0.97;}
        for(int j=0;j<offered && j<=accepted;++j) {
            const int b=bin(probabilities[j]);seen[b]+=1;matched[b]+=j<accepted?1:0;
        }
    }
};
} // namespace basalt::deepseek41
