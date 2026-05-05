#include "mse_loss.h"

namespace deeplearning {

double MSELoss::Loss(double target, double output) {
  return (double)(1.0 / 2.0) * (double)(target - output) *
         (double)(target - output);
}

double MSELoss::DerivLoss(double target, double output) {
  return -2.0 * (target - output);
}

LossType MSELoss::GetLossType() { return LOSS_MSE; }

} // namespace deeplearning
