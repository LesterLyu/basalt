// src/models/deepseek41/prompt_host.hpp - the host memory prompts and images take at their peak, allocated once at start.
#pragma once

#include "basalt/vision/preprocess.hpp"
#include "basalt/core/memory_usage.hpp"
#include "models/deepseek41/pack.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::models::deepseek41 {

class DsModel;

/// Bytes between experts in a staging buffer: the layer's blob rounded up to whole blocks of its two formats.
uint64_t staged_stride(const ExpertLayer& e);

/// The prompt path's fixed host buffers and each slot's on-demand image buffers. The RAM tier leaves the image budget
/// available without keeping decoded images allocated between requests.
class PromptHost {
public:
    PromptHost() = default;
    ~PromptHost();
    PromptHost(const PromptHost&) = delete;
    PromptHost& operator=(const PromptHost&) = delete;

    /// `ring_pieces`: experts the pinned ring of file and second-card reads holds (0: none); `slots`: sessions;
    /// `vision`: budget each slot's largest image, allocating its buffers on use.
    bool open(const DsModel& m, int ring_pieces, int slots, bool vision, std::string& err);
    /// The ring at `pieces` experts (its contents are not kept).
    bool set_ring(int pieces, std::string& err);
    /// Page-locked bytes, which the driver keeps tracking memory for.
    uint64_t pinned_bytes() const { return pinned_; }
    uint64_t bytes() const { return bytes_; }
    uint64_t image_budget_bytes() const { return image_budget_; }
    void memory_usage(core::MemoryUsage& usage) const;

    size_t slot = 0;                    ///< bytes of one staged expert (staged_stride of the largest layer)
    uint8_t* ring = nullptr;            ///< [ring_pieces][slot], pinned
    int ring_pieces = 0;
    uint8_t* stage[2] = {};             ///< [kStageExperts][slot] each, pinned: a short read's two launches
    uint8_t* replacement_stage = nullptr; ///< one helper-owned replacement, copied through RAM without retaining it
    float* hx = nullptr;                ///< [share_rows][n_embd] a short read's activations, pinned
    float* hrows = nullptr;             ///< [kShareRows][n_embd] rows the CPU returns, pinned and mapped
    uint8_t* act = nullptr;             ///< [share_rows][kNativeActBytes] the CPU's quantized activations
    int share_rows = 0;                 ///< most tokens of a short read
    std::vector<float> embed;           ///< [kEmbedRows][4 streams][n_embd]
    std::vector<float> engram_rows;     ///< [kEngramRows * n_cols][row_values] dequantized
    std::vector<uint8_t> engram_raw[2]; ///< a chunk's Engram reads and the next one's: [layer][kMaxChunk * n_cols][row_bytes]
    size_t engram_layer = 0;            ///< bytes of one layer in engram_raw

    struct Image {
        vision::Image picture;          ///< the decoded image
        std::vector<float> patches;     ///< its patch rows
        void release() {
            std::vector<uint8_t>().swap(picture.rgb);
            std::vector<float>().swap(patches);
            picture.w = picture.h = 0;
        }
    };
    /// The next slot's image buffers (each session takes one), nullptr when there are none left.
    Image* take_image();

private:
    std::vector<Image> images_;
    int images_taken_ = 0;
    uint64_t pinned_ = 0, bytes_ = 0;
    uint64_t image_budget_ = 0;
    uint64_t expert_staging_bytes_ = 0, activation_bytes_ = 0;
};

}  // namespace basalt::models::deepseek41
