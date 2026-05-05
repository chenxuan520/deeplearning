#pragma once

#include "loss_base.h"
#include <cmath>

namespace deeplearning {

class CrossEntropyLoss : public LossFunction {
public:
  CrossEntropyLoss();
  double Loss(double target, double output) override;
  double DerivLoss(double target, double output) override;
  LossType GetLossType() override;
};

} // namespace deeplearning
