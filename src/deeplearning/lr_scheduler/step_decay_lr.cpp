#include "step_decay_lr.h"

#include <cmath>

namespace deeplearning {

StepDecayLR::StepDecayLR(double base_lr, int step_size, double gamma)
    : LRScheduler(base_lr), step_size_(step_size), gamma_(gamma) {}

double StepDecayLR::GetLR(int step) {
  if (step_size_ <= 0) {
    return base_lr_;
  }
  int k = step / step_size_;
  return base_lr_ * std::pow(gamma_, k);
}

LRSchedulerType StepDecayLR::GetType() { return LR_SCHEDULER_STEP_DECAY; }

} // namespace deeplearning
