#pragma once
#include "loss_base.h"
namespace deeplearning {

class MSELoss : public LossFunction {
public:
  double Loss(double target, double output) override {
    return (double)(1.0 / 2.0) * (double)(target - output) *
           (double)(target - output);
  }
  double DerivLoss(double target, double output) override {
    return -2.0 * (target - output);
  }
  LossType GetLossType() override { return LOSS_MSE; }
};

} // namespace deeplearning
