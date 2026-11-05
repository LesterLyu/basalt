#pragma once
#include "basalt/deepseek41/device_model.hpp"
#include "basalt/deepseek41/model.hpp"
#include "basalt/core/expert_source.hpp"
#include <span>

namespace basalt::deepseek41 {
struct Profile;
// The prompt path's grouped MMQ products and finite copy/compute ring; the decode tiers retain ownership.
class PrefillExperts {
public:
    PrefillExperts(DeviceModel& device,const kernels::cpu::ExpertLayout& layout,int capacity);
    ~PrefillExperts();
    size_t pool_bytes() const;
    void run(int layer,const float* x,std::span<const int32_t> ids,const std::vector<int32_t>& kind,
             int tokens,int top,core::ExpertCache& cache,core::FileExpertSource& source,float* output,ExecutionStats& stats);
    void run_long(int layer,const float* x,const std::vector<int32_t>& ids,const float* weights,int tokens,int top,
                  core::ExpertCache& cache,core::FileExpertSource& source,core::PeerExperts& helper,
                  float* parts,double* accumulator,ExecutionStats& stats,int64_t primary_limit,Profile* profile=nullptr,
                  const std::function<void()>& between_groups={},void* background_stream=nullptr);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
