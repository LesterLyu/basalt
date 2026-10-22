#include "basalt/core/peer_copy.hpp"

#include "basalt/core/on_device.hpp"

#include <limits>

namespace basalt::core {
namespace {
bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("peer copy: ") + what + ": " + cudaGetErrorString(e);
    return false;
}
}  // namespace

bool PeerCopy::open(int src_device, int dst_device, size_t bytes, size_t slots, std::string& err) {
    if (src_ == src_device && dst_ == dst_device && bytes_ >= bytes && slots_.size() == slots && stream_)
        return true;
    close();
    int count = 0;
    if (!ck(cudaGetDeviceCount(&count), "device count", err)) return false;
    if (src_device < 0 || dst_device < 0 || src_device == dst_device || !bytes || !slots ||
        src_device >= count || dst_device >= count ||
        slots > std::numeric_limits<size_t>::max() / bytes) {
        err = "peer copy: invalid devices or buffer size";
        return false;
    }
    src_ = src_device;
    dst_ = dst_device;
    bytes_ = bytes;
    slots_.resize(slots);
    bool ok = true;
    {
        const OnDevice on(dst_);
        for (Slot& s : slots_) {
            if (!ck(cudaHostAlloc(&s.host, bytes, cudaHostAllocPortable), "host allocation", err) ||
                !ck(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming), "destination event", err)) {
                ok = false;
                break;
            }
        }
    }
    if (ok) {
        const OnDevice on(src_);
        ok = ck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "source stream", err);
        for (Slot& s : slots_) {
            if (!ok) break;
            ok = ck(cudaEventCreateWithFlags(&s.read, cudaEventDisableTiming), "source event", err);
        }
    }
    if (!ok) close();
    return ok;
}

bool PeerCopy::copy(void* dst, const void* src, size_t bytes, cudaStream_t dst_stream, std::string& err) {
    if (!stream_ || !dst || !src || bytes > bytes_) {
        err = "peer copy: unopened transport, null pointer or oversized copy";
        return false;
    }
    Slot& s = slots_[next_];
    next_ = (next_ + 1) % slots_.size();
    {
        const OnDevice on(src_);
        if (!ck(cudaStreamWaitEvent(stream_, s.done, 0), "wait for buffer reuse", err) ||
            !ck(cudaMemcpyAsync(s.host, src, bytes, cudaMemcpyDeviceToHost, stream_), "read source", err) ||
            !ck(cudaEventRecord(s.read, stream_), "record source", err)) return false;
    }
    const OnDevice on(dst_);
    return ck(cudaStreamWaitEvent(dst_stream, s.read, 0), "wait for source", err) &&
           ck(cudaMemcpyAsync(dst, s.host, bytes, cudaMemcpyHostToDevice, dst_stream), "write destination", err) &&
           ck(cudaEventRecord(s.done, dst_stream), "record destination", err);
}

void PeerCopy::close() {
    if (src_ < 0) return;
    {
        const OnDevice on(src_);
        if (stream_) cudaStreamSynchronize(stream_);
    }
    {
        const OnDevice on(dst_);
        for (Slot& s : slots_) if (s.done) cudaEventSynchronize(s.done);
        for (Slot& s : slots_) if (s.done) cudaEventDestroy(s.done);
    }
    {
        const OnDevice on(src_);
        for (Slot& s : slots_) {
            if (s.read) cudaEventDestroy(s.read);
            if (s.host) cudaFreeHost(s.host);
        }
        if (stream_) cudaStreamDestroy(stream_);
    }
    slots_.clear();
    src_ = dst_ = -1;
    bytes_ = next_ = 0;
    stream_ = nullptr;
}

}  // namespace basalt::core
