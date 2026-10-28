#include "util/thread_pool.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace deeplearning {

ThreadPool::ThreadPool(int worker_num) {
  worker_num = std::max(1, worker_num);
  workers_.reserve(worker_num);
  for (int index = 0; index < worker_num; index++) {
    workers_.emplace_back([this, index]() { WorkerLoop(index); });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
    generation_++;
  }
  work_condition_.notify_all();
  for (auto &worker : workers_) {
    worker.join();
  }
}

void ThreadPool::Run(
    int count, int worker_num,
    const std::function<void(int begin, int end)> &function) {
  if (count <= 0) {
    return;
  }
  worker_num = std::max(1, std::min(
      {worker_num, count, static_cast<int>(workers_.size())}));
  if (worker_num == 1) {
    function(0, count);
    return;
  }
  {
    std::unique_lock<std::mutex> lock(mutex_);
    done_condition_.wait(lock, [this]() { return !running_; });
    running_ = true;
    count_ = count;
    active_workers_ = worker_num;
    completed_workers_ = 0;
    function_ = function;
    generation_++;
  }
  work_condition_.notify_all();
  std::unique_lock<std::mutex> lock(mutex_);
  done_condition_.wait(lock, [this]() {
    return completed_workers_ == active_workers_;
  });
  function_ = nullptr;
  running_ = false;
  done_condition_.notify_all();
}

void ThreadPool::WorkerLoop(int worker_index) {
  unsigned long observed_generation = 0;
  while (true) {
    std::function<void(int, int)> function;
    int begin = 0;
    int end = 0;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      work_condition_.wait(lock, [this, observed_generation]() {
        return stopping_ || generation_ != observed_generation;
      });
      if (stopping_) {
        return;
      }
      observed_generation = generation_;
      if (worker_index >= active_workers_) {
        continue;
      }
      begin = count_ * worker_index / active_workers_;
      end = count_ * (worker_index + 1) / active_workers_;
      function = function_;
    }
    function(begin, end);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      completed_workers_++;
      if (completed_workers_ == active_workers_) {
        done_condition_.notify_one();
      }
    }
  }
}

ThreadPool &ThreadPool::Global() {
  static ThreadPool pool([]() {
    const char *configured = std::getenv("DL_THREAD_POOL_SIZE");
    if (configured != nullptr) {
      try {
        return std::max(1, std::stoi(configured));
      } catch (...) {
        // Fall back to the hardware count for invalid environment values.
      }
    }
    return static_cast<int>(
        std::max(1u, std::thread::hardware_concurrency()));
  }());
  return pool;
}

} // namespace deeplearning
