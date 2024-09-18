#pragma once
#include "loss_base.h"
namespace deeplearning {

class MSELoss : public LossFunction {
public:
  double AverageLoss(const std::vector<double> &target,
                     const std::vector<double> &output) override {
    double result = 0;
    if (target.size() != output.size()) {
      return -1;
    }
    for (int i = 0; i < target.size(); i++) {
      result += Loss(target[i], output[i]);
    }
    result /= target.size();
    return result;
  }
  double Loss(double target, double output) override {
    return (target - output) * (target - output);
  }
  double DerivLoss(double target, double output) override {
    return -2 * (target - output);
  }
  LossType GetLossType() override { return LOSS_MSE; }
};

} // namespace deeplearning
