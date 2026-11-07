#pragma once

#include "basalt/platform/cpu.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/models/qwen4exp/kernels/dec_moe.hpp"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace basalt::core {

// Keep the peer context on its own thread: switching away after launching a host-waiting kernel can block
// the caller in the driver before it launches the primary window that supplies the peer's doorbells.
class PeerWindowLauncher {
public:
    ~PeerWindowLauncher() {
        if (!worker_.joinable()) return;
        { std::lock_guard<std::mutex> lk(mu_); stop_ = true; }
        cv_.notify_one();
        worker_.join();
    }
    bool open(int device, std::string& err) {
        try {
            worker_ = std::thread([this, device] {
                const cudaError_t e = cudaSetDevice(device);
                ok_ = e == cudaSuccess;
                if (!ok_) error_ = cudaGetErrorString(e);
                ready_.store(true, std::memory_order_release);
                std::unique_lock<std::mutex> lk(mu_);
                for (;;) {
                    cv_.wait(lk, [this] { return pending_ || stop_; });
                    if (stop_) return;
                    pending_ = false;
                    lk.unlock();
                    const bool clear = basalt::kernels::dec_moe_zero_words(args_.arrive, 16, stream_);
                    ok_ = clear &&
                          basalt::kernels::dec_moe_peer_launch(args_, blocks_, stream_, q2_);
                    if (!ok_) error_ = clear ? "peer launch failed" : cudaGetErrorString(cudaGetLastError());
                    done_.store(true, std::memory_order_release);
                    lk.lock();
                }
            });
        } catch (const std::exception& e) {
            err = std::string("verify: peer launcher thread: ") + e.what();
            return false;
        }
        wait(ready_);
        if (!ok_) err = "verify: peer launcher context: " + error_;
        return ok_;
    }
    bool launch(const basalt::kernels::DmPeerArgs& args, int blocks, cudaStream_t stream, bool q2, std::string& err) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            args_ = args; blocks_ = blocks; stream_ = stream; q2_ = q2;
            done_.store(false, std::memory_order_relaxed);
            pending_ = true;
        }
        cv_.notify_one();
        wait(done_);
        if (!ok_) err = "verify: " + error_;
        return ok_;
    }
private:
    static void wait(const std::atomic<bool>& flag) {
        for (unsigned spins = 0; !flag.load(std::memory_order_acquire); ++spins) {
            if (spins < 20000) basalt::platform::cpu_relax();
            else std::this_thread::yield();
        }
    }
    std::thread worker_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::atomic<bool> ready_{false}, done_{true};
    bool pending_ = false, stop_ = false, ok_ = false, q2_ = false;
    int blocks_ = 0;
    cudaStream_t stream_ = nullptr;
    basalt::kernels::DmPeerArgs args_{};
    std::string error_;
};

}  // namespace basalt::core
