#pragma once

#include "lr_scheduler_base.h"

namespace deeplearning {

// 指数衰减: lr = base_lr * gamma^step
// 适合简单训练循环, 每步都微调.
class ExponentialDecayLR : public LRScheduler {
public:
  ExponentialDecayLR(double base_lr, double gamma);

  double GetLR(int step) override;
  LRSchedulerType GetType() override;

  void set_gamma(double v) { gamma_ = v; }

private:
  double gamma_;
};

} // namespace deeplearning
