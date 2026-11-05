#include "basalt/deepseek41/cpu_experts.hpp"
#include "basalt/platform/cpu.hpp"
#include "basalt/log.hpp"
#include "basalt/core/diag.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <map>
#include <stdexcept>
#include <thread>

namespace basalt::deepseek41 {
struct CpuExperts::Impl {
    std::mutex mutex, invocation;
    std::condition_variable begin, done;
    std::vector<std::thread> threads;
    std::function<void(size_t)> task;
    std::vector<uint8_t> decode_activation,decode_hidden_q;
    std::vector<float> decode_hidden;
    std::array<uint64_t,7> iq2_groups{};
    std::atomic<size_t> next{0};
    std::atomic_flag pin_warning = ATOMIC_FLAG_INIT;
    size_t tasks = 0, generation = 0, remaining = 0;
    bool stopping = false;

    explicit Impl(int workers) {
        auto topology = platform::detect_cpu_topology(true);
        auto cpus = topology.worker_cores;
        if (workers <= 0) workers = std::max(1, int(cpus.size()));
        if (workers > 256) throw std::invalid_argument("deepseek41: excessive CPU worker count");
        for (int cpu : platform::logical_cpus())
            if (cpu != topology.host_core && std::find(cpus.begin(), cpus.end(), cpu) == cpus.end()) cpus.push_back(cpu);
        if (topology.host_core >= 0) cpus.push_back(topology.host_core);
        try {
            for (int i = 0; i < workers; ++i) threads.emplace_back([this, cpu = cpus.empty() ? -1 : cpus[size_t(i) % cpus.size()]] {
                std::string error;
                if (cpu >= 0 && !platform::set_thread_affinity(cpu, error) && !pin_warning.test_and_set())
                    basalt::log::warn("basalt deepseek41", "CPU worker affinity: %s", error.c_str());
                size_t seen = 0;
                std::unique_lock lock(mutex);
                for (;;) {
                    begin.wait(lock, [&] { return stopping || generation != seen; });
                    if (stopping) return;
                    seen = generation;
                    lock.unlock();
                    for (;;) {
                        const size_t index = next.fetch_add(1, std::memory_order_relaxed);
                        if (index >= tasks) break;
                        task(index);
                    }
                    lock.lock();
                    if (!--remaining) done.notify_one();
                }
            });
        } catch (...) { close(); throw; }
    }
    ~Impl() {
        close();
        if(core::exp_flag("ds-cpu-iq2-down-exact")!=0)
            basalt::log::info("basalt deepseek41","CPU IQ2_XXS groups width1=%llu width2=%llu width3=%llu width4=%llu width5=%llu width6=%llu",
                             (unsigned long long)iq2_groups[1],(unsigned long long)iq2_groups[2],
                             (unsigned long long)iq2_groups[3],(unsigned long long)iq2_groups[4],
                             (unsigned long long)iq2_groups[5],(unsigned long long)iq2_groups[6]);
    }
    void close() {
        { std::lock_guard lock(mutex); stopping = true; }
        begin.notify_all();
        for (auto& thread : threads) if (thread.joinable()) thread.join();
    }
    void parallel(size_t count, std::function<void(size_t)> fn) {
        if (!count) return;
        std::unique_lock lock(mutex);
        task = std::move(fn); tasks = count; next.store(0); remaining = threads.size(); ++generation;
        begin.notify_all();
        done.wait(lock, [&] { return !remaining; });
        task = {};
    }
};
CpuExperts::CpuExperts(int workers) : impl_(std::make_unique<Impl>(workers)) {}
CpuExperts::~CpuExperts() = default;
int CpuExperts::workers() const { return int(impl_->threads.size()); }
std::vector<float> CpuExperts::run(const kernels::cpu::NativeFmt& f, const float* x,
                                   const std::vector<const uint8_t*>& blobs) {
    std::vector<Job> jobs;
    for (auto* blob : blobs) jobs.push_back({blob, 0});
    return run_window(f, x, 1, jobs);
}
std::vector<float> CpuExperts::run_window(const kernels::cpu::NativeFmt& f, const float* x, int tokens,
                                          const std::vector<Job>& jobs) {
    std::lock_guard invocation(impl_->invocation);
    if (f.n_embd < 1 || f.n_ff < 1 || !f.act_bytes || !f.h_bytes || !x || tokens < 1 ||
        std::any_of(jobs.begin(), jobs.end(), [tokens](const auto& j) { return !j.blob || j.token < 0 || j.token >= tokens; }))
        throw std::invalid_argument("deepseek41: invalid CPU expert job");
    if (jobs.empty()) return {};
    const bool direct = jobs.front().destination != nullptr;
    if (std::any_of(jobs.begin(), jobs.end(), [direct](const Job& j) { return (j.destination != nullptr) != direct; }))
        throw std::invalid_argument("deepseek41: inconsistent CPU expert destinations");
    const bool reuse=tokens<=6 && core::exp_flag("ds-cpu-reuse-scratch")!=0;
    std::vector<uint8_t> local_activation,local_hidden_q;
    std::vector<float> local_hidden;
    auto& activation=reuse?impl_->decode_activation:local_activation;
    auto& hidden_q=reuse?impl_->decode_hidden_q:local_hidden_q;
    auto& hidden=reuse?impl_->decode_hidden:local_hidden;
    const auto grow=[](auto& storage,size_t size) {if(storage.size()<size)storage.resize(size);};
    grow(activation,tokens*f.act_bytes);
    grow(hidden_q,jobs.size()*f.h_bytes);
    grow(hidden,jobs.size()*size_t(f.n_ff));
    std::vector<float> output(direct ? 0 : jobs.size()*size_t(f.n_embd));
    unsigned needed=0;
    if(reuse)for(const auto& job:jobs)needed|=1u<<job.token;
    for(int t=0;t<tokens;++t)if(!reuse || ((needed>>t)&1u))
        kernels::cpu::native_quant_act(f,x+size_t(t)*f.n_embd,activation.data()+t*f.act_bytes);
    struct Group { const uint8_t* blob; std::vector<const void*> act, down; std::vector<float*> ff, out; };
    std::map<const uint8_t*, size_t> seen;
    std::vector<Group> groups;
    for (size_t job = 0; job < jobs.size(); ++job) {
        auto [it, inserted] = seen.emplace(jobs[job].blob, groups.size());
        if (inserted) groups.push_back({jobs[job].blob, {}, {}, {}, {}});
        auto& group = groups[it->second];
        group.act.push_back(activation.data() + jobs[job].token * f.act_bytes);
        group.down.push_back(hidden_q.data() + job * f.h_bytes);
        group.ff.push_back(hidden.data() + job * size_t(f.n_ff));
        group.out.push_back(direct ? jobs[job].destination : output.data() + job * size_t(f.n_embd));
    }
    const size_t gu_rows = (size_t(f.n_ff) + 63) / 64;
    if(f.d_type==16 && core::exp_flag("ds-cpu-iq2-down-exact")!=0)
        for(const auto& group:groups)if(group.act.size()<impl_->iq2_groups.size())++impl_->iq2_groups[group.act.size()];
    const auto gu = [&](size_t task) {
        const auto& group = groups[task / gu_rows];
        const int begin = int(task % gu_rows) * 64, end = std::min<int>(begin + 64, int(f.n_ff));
        kernels::cpu::native_gu_rows(f, group.blob, group.act.data(), int(group.act.size()), group.ff.data(), begin, end);
    };
    const size_t down_rows = (size_t(f.n_embd) + 127) / 128;
    const auto down = [&](size_t group_index,size_t chunk) {
        const auto& group = groups[group_index];
        const int begin = int(chunk) * 128, end = std::min<int>(begin + 128, int(f.n_embd));
        kernels::cpu::native_down_rows(f, group.blob, group.down.data(), int(group.down.size()), group.out.data(), begin, end);
    };
    const size_t gu_tasks=groups.size()*gu_rows,down_tasks=groups.size()*down_rows;
    if(core::exp_flag("ds-cpu-group-pipeline")!=0 && tokens<=6) {
        struct Phase {std::atomic<size_t> remaining{0};std::atomic<int> ready{-1};};
        std::vector<Phase> phases(groups.size());
        for(auto& phase:phases)phase.remaining.store(gu_rows,std::memory_order_relaxed);
        std::atomic<size_t> next_gu{0},next_ready{0},next_down{0};
        // One task per existing worker. The final gate task publishes each expert independently.
        impl_->parallel(impl_->threads.size(),[&](size_t) {
            for(size_t i;(i=next_gu.fetch_add(1,std::memory_order_relaxed))<gu_tasks;) {
                const size_t g=i/gu_rows;
                gu(i);
                if(phases[g].remaining.fetch_sub(1,std::memory_order_acq_rel)==1) {
                    const auto& group=groups[g];
                    for(size_t j=0;j<group.ff.size();++j)
                        kernels::cpu::native_quant_h(f,group.ff[j],const_cast<void*>(group.down[j]));
                    auto& ready=phases[next_ready.fetch_add(1,std::memory_order_relaxed)].ready;
                    ready.store(int(g),std::memory_order_release);ready.notify_all();
                }
            }
            for(size_t i;(i=next_down.fetch_add(1,std::memory_order_relaxed))<down_tasks;) {
                auto& ready=phases[i/down_rows].ready;
                int group=ready.load(std::memory_order_acquire);
                if(group<0) {
                    ready.wait(-1,std::memory_order_acquire);
                    group=ready.load(std::memory_order_acquire);
                }
                down(size_t(group),i%down_rows);
            }
        });
    } else {
        impl_->parallel(gu_tasks,gu);
        for(size_t j=0;j<jobs.size();++j)
            kernels::cpu::native_quant_h(f,hidden.data()+j*size_t(f.n_ff),hidden_q.data()+j*f.h_bytes);
        impl_->parallel(down_tasks,[&](size_t task) {down(task/down_rows,task%down_rows);});
    }
    return output;
}
} // namespace basalt::deepseek41
