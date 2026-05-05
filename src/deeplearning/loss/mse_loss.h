#pragma once
#include "loss_base.h"
namespace deeplearning {

class MSELoss : public LossFunction {
public:
  double Loss(double target, double output) override;
  double DerivLoss(double target, double output) override;
  LossType GetLossType() override;
};

} // namespace deeplearning
