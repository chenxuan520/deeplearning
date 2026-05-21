#pragma once

#include "lr_scheduler_base.h"

namespace deeplearning {

// Step decay: 每 step_size 步乘以 gamma.
//   lr = base_lr * gamma^(step / step_size)
class StepDecayLR : public LRScheduler {
public:
  StepDecayLR(double base_lr, int step_size, double gamma = 0.1);

  double GetLR(int step) override;
  LRSchedulerType GetType() override;

  void set_step_size(int v) { step_size_ = v; }
  void set_gamma(double v) { gamma_ = v; }

private:
  int step_size_;
  double gamma_;
};

} // namespace deeplearning
