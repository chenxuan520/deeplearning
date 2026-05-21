#include "exponential_decay_lr.h"

#include <cmath>

namespace deeplearning {

ExponentialDecayLR::ExponentialDecayLR(double base_lr, double gamma)
    : LRScheduler(base_lr), gamma_(gamma) {}

double ExponentialDecayLR::GetLR(int step) {
  return base_lr_ * std::pow(gamma_, step);
}

LRSchedulerType ExponentialDecayLR::GetType() {
  return LR_SCHEDULER_EXPONENTIAL_DECAY;
}

} // namespace deeplearning
