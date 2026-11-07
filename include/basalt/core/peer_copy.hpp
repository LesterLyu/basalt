#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

namespace basalt::core {

/// Bounded asynchronous device-to-device transport through portable pinned host buffers.
/// The source must stay unchanged until the destination stream finishes its copy.
/// Calls are serialized by the caller; the destination stream belongs to dst_device.
class PeerCopy {
public:
    PeerCopy() = default;
    ~PeerCopy() { close(); }
    PeerCopy(const PeerCopy&) = delete;
    PeerCopy& operator=(const PeerCopy&) = delete;

    bool open(int src_device, int dst_device, size_t bytes, size_t slots, std::string& err);
    bool copy(void* dst, const void* src, size_t bytes, cudaStream_t dst_stream, std::string& err);
    void close();
    size_t host_bytes() const { return bytes_ * slots_.size(); }

private:
    struct Slot {
        void* host = nullptr;
        cudaEvent_t read = nullptr, done = nullptr;
    };
    int src_ = -1, dst_ = -1;
    size_t bytes_ = 0, next_ = 0;
    cudaStream_t stream_ = nullptr;
    std::vector<Slot> slots_;
};

}  // namespace basalt::core
