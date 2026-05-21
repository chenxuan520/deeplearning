#pragma once

#include "lr_scheduler_base.h"

namespace deeplearning {

// Linear warmup + cosine annealing (现代 Transformer 训练常用):
//   step < warmup_steps:
//     lr = base_lr * (step + 1) / warmup_steps
//   else:
//     progress = (step - warmup_steps) / (t_max - warmup_steps)
//     lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
class WarmupCosineLR : public LRScheduler {
public:
  WarmupCosineLR(double base_lr, int warmup_steps, int t_max,
                 double min_lr = 0.0);

  double GetLR(int step) override;
  LRSchedulerType GetType() override;

private:
  int warmup_steps_;
  int t_max_;
  double min_lr_;
};

} // namespace deeplearning
