#pragma once
#include "basalt/kernels/cpu/native_expert.hpp"
#include <memory>
#include <vector>

namespace basalt::deepseek41 {

// The existing native row kernels, with model-sized scratch and row-parallel scheduling.
// Returned expert rows are unweighted; the GPU combines routing weights exactly once.
class CpuExperts {
public:
    struct Job { const uint8_t* blob = nullptr; int token = 0; float* destination = nullptr; };
    explicit CpuExperts(int workers = 0);
    ~CpuExperts();
    CpuExperts(const CpuExperts&) = delete;
    CpuExperts& operator=(const CpuExperts&) = delete;
    int workers() const;
    std::vector<float> run(const kernels::cpu::NativeFmt& format, const float* input,
                            const std::vector<const uint8_t*>& experts);
    // If every job supplies a distinct destination, write rows there and return an empty vector.
    std::vector<float> run_window(const kernels::cpu::NativeFmt& format, const float* input, int tokens,
                                   const std::vector<Job>& jobs);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace basalt::deepseek41
