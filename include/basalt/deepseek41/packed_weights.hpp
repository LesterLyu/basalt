#pragma once
#include "basalt/deepseek41/weights.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"

namespace basalt::deepseek41 {

struct DraftTensorRange {
    core::FileRange source;
    bool widen_bf16 = false;
};
struct DraftExpertRange {
    core::FileRange source;
    uint64_t destination = 0;
};

// The target owns the Engram table. A separate draft may provide its weights without changing the target.
class PackedWeights {
public:
    explicit PackedWeights(const std::string& path, const std::string& draft_path = {});
    const Contract& contract() const { return contract_; }
    const core::ModelFile& file() const { return file_; }
    const GgufFile& metadata_file() const { return *target_; }
    const GgufFile& draft_metadata_file() const { return *draft_; }
    const kernels::cpu::ExpertLayout& expert_layout() const { return target_layout_; }
    const kernels::cpu::ExpertLayout& draft_expert_layout() const { return draft_layout_; }
    const std::vector<uint32_t>& token_map() const { return token_map_; }
    core::FileRange expert_file() const { return file_.range("experts"); }
    const std::vector<DraftExpertRange>& draft_expert_ranges() const { return draft_expert_ranges_; }
    DraftTensorRange draft_tensor_range(const TensorInfo& tensor) const;
    TensorView target_tensor(const std::string& name) const;
    TensorView draft_tensor(const std::string& name) const;
private:
    core::ModelFile file_;
    std::unique_ptr<core::ModelFile> external_draft_;
    std::unique_ptr<GgufFile> target_, draft_;
    Contract contract_;
    kernels::cpu::ExpertLayout target_layout_, draft_layout_;
    std::map<std::string, DraftTensorRange> xs2_tensors_;
    std::vector<DraftExpertRange> draft_expert_ranges_;
    std::vector<uint32_t> token_map_;
    void open_xs2(const std::string& path);
};

}  // namespace basalt::deepseek41
