#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace deeplearning {

// A fixed-size worker pool for short numeric kernels. Run divides [0, count)
// into contiguous ranges and waits for all workers. One process-wide pool is
// shared by CNN layers to avoid creating threads for every convolution.
class ThreadPool {
public:
  explicit ThreadPool(int worker_num);
  ~ThreadPool();

  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;

  void Run(int count, int worker_num,
           const std::function<void(int begin, int end)> &function);
  int worker_num() const { return static_cast<int>(workers_.size()); }

  static ThreadPool &Global();

private:
  void WorkerLoop(int worker_index);

  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable work_condition_;
  std::condition_variable done_condition_;
  std::function<void(int begin, int end)> function_;
  int count_ = 0;
  int active_workers_ = 0;
  int completed_workers_ = 0;
  unsigned long generation_ = 0;
  bool stopping_ = false;
  bool running_ = false;
};

} // namespace deeplearning
