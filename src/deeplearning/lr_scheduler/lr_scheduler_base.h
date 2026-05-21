#pragma once

namespace deeplearning {

enum LRSchedulerType {
  LR_SCHEDULER_CONSTANT,
  LR_SCHEDULER_STEP_DECAY,
  LR_SCHEDULER_EXPONENTIAL_DECAY,
  LR_SCHEDULER_COSINE_ANNEALING,
  LR_SCHEDULER_WARMUP_COSINE,
};

// 学习率调度器: 给定当前 step 返回该 step 应该用的学习率.
// 与 NeuralNetwork::Train 配合: 在每次 minibatch 更新前, 调用
// scheduler->GetLR(step) 设置 network 学习率.
//
// 也可以单独使用, 由调用方在 each_epoch_call 回调里手动调用:
//   network.set_learning_rate(scheduler->GetLR(epoch_num));
class LRScheduler {
public:
  explicit LRScheduler(double base_lr) : base_lr_(base_lr) {}
  virtual ~LRScheduler() = default;

  virtual double GetLR(int step) = 0;
  virtual LRSchedulerType GetType() = 0;

  double base_lr() const { return base_lr_; }
  void set_base_lr(double v) { base_lr_ = v; }

protected:
  double base_lr_;
};

} // namespace deeplearning
