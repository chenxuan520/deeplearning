#pragma once

#include <vector>

namespace deeplearning {

enum LossType {
  LOSS_MSE,
  LOSS_CROSS_ENTROPY,
};

class LossFunction {
public:
  virtual double AverageLoss(const std::vector<double> &target,
                             const std::vector<double> &output) {
    double result = 0;
    if (target.size() != output.size() || target.size() == 0) {
      return -1;
    }
    for (int i = 0; i < target.size(); i++) {
      result += Loss(target[i], output[i]);
    }
    result /= target.size();
    return result;
  }

  virtual double Loss(double target, double output) = 0;
  virtual double DerivLoss(double target, double output) = 0;
  virtual LossType GetLossType() = 0;
};
} // namespace deeplearning
