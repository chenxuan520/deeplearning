#pragma once

#include "loss_base.h"
#include <cmath>

namespace deeplearning {

class CrossEntropyLoss : public LossFunction {
public:
  CrossEntropyLoss() = default;
  virtual double Loss(double target, double output) override {
    return -target * log(output) - (1 - target) * log(1 - output);
  }

  virtual double DerivLoss(double target, double output) override {
    return (output - target) / (output * (1 - output));
  }

  virtual LossType GetLossType() override { return LOSS_CROSS_ENTROPY; }
};

} // namespace deeplearning
