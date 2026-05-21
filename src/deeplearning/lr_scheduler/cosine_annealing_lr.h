#pragma once

#include "lr_scheduler_base.h"

namespace deeplearning {

// Cosine annealing (Loshchilov & Hutter, SGDR):
//   lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * step / T_max))
//
// step >= T_max 时返回 min_lr. 适合长训练后期平滑收尾.
class CosineAnnealingLR : public LRScheduler {
public:
  CosineAnnealingLR(double base_lr, int t_max, double min_lr = 0.0);

  double GetLR(int step) override;
  LRSchedulerType GetType() override;

  void set_t_max(int v) { t_max_ = v; }
  void set_min_lr(double v) { min_lr_ = v; }

private:
  int t_max_;
  double min_lr_;
};

} // namespace deeplearning
