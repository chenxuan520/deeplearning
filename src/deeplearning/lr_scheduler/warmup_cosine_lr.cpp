#include "warmup_cosine_lr.h"

#include <cmath>

namespace deeplearning {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

WarmupCosineLR::WarmupCosineLR(double base_lr, int warmup_steps, int t_max,
                               double min_lr)
    : LRScheduler(base_lr), warmup_steps_(warmup_steps), t_max_(t_max),
      min_lr_(min_lr) {}

double WarmupCosineLR::GetLR(int step) {
  if (warmup_steps_ > 0 && step < warmup_steps_) {
    return base_lr_ * (double)(step + 1) / (double)warmup_steps_;
  }
  if (t_max_ <= warmup_steps_) {
    return base_lr_;
  }
  int rest = t_max_ - warmup_steps_;
  int s = step - warmup_steps_;
  if (s >= rest) {
    return min_lr_;
  }
  double progress = (double)s / (double)rest;
  double cos_val = std::cos(kPi * progress);
  return min_lr_ + 0.5 * (base_lr_ - min_lr_) * (1.0 + cos_val);
}

LRSchedulerType WarmupCosineLR::GetType() {
  return LR_SCHEDULER_WARMUP_COSINE;
}

} // namespace deeplearning
