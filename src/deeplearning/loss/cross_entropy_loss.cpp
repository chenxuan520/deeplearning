#include "cross_entropy_loss.h"

namespace deeplearning {

CrossEntropyLoss::CrossEntropyLoss() = default;

double CrossEntropyLoss::Loss(double target, double output) {
  return -target * log(output) - (1.0 - target) * log(1.0 - output);
}

double CrossEntropyLoss::DerivLoss(double target, double output) {
  return (output - target) / (output * (1.0 - output));
}

LossType CrossEntropyLoss::GetLossType() { return LOSS_CROSS_ENTROPY; }

} // namespace deeplearning
