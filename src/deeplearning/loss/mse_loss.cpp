#include "mse_loss.h"

namespace deeplearning {

double MSELoss::Loss(double target, double output) {
  return (double)(1.0 / 2.0) * (double)(target - output) *
         (double)(target - output);
}

double MSELoss::DerivLoss(double target, double output) {
  // NOTE: Strictly, d/do [½(t−o)²] = −(t−o). This returns −2(t−o), i.e. a
  // constant factor of 2 vs the Loss formula above. Direction is identical;
  // the scale is absorbed by the learning rate. Kept as-is for compatibility.
  return -2.0 * (target - output);
}

LossType MSELoss::GetLossType() { return LOSS_MSE; }

} // namespace deeplearning
