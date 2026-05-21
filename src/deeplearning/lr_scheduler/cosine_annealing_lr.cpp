#include "cosine_annealing_lr.h"

#include <cmath>

namespace deeplearning {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

CosineAnnealingLR::CosineAnnealingLR(double base_lr, int t_max, double min_lr)
    : LRScheduler(base_lr), t_max_(t_max), min_lr_(min_lr) {}

double CosineAnnealingLR::GetLR(int step) {
  if (t_max_ <= 0) {
    return base_lr_;
  }
  if (step >= t_max_) {
    return min_lr_;
  }
  double cos_val = std::cos(kPi * (double)step / (double)t_max_);
  return min_lr_ + 0.5 * (base_lr_ - min_lr_) * (1.0 + cos_val);
}

LRSchedulerType CosineAnnealingLR::GetType() {
  return LR_SCHEDULER_COSINE_ANNEALING;
}

} // namespace deeplearning
