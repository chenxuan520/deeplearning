#pragma once

#include <vector>

namespace deeplearning {

enum LossType {
  LOSS_MSE,
};

class LossFunction {
public:
  virtual double AverageLoss(const std::vector<double> &target,
                             const std::vector<double> &output) = 0;
  virtual double Loss(double target, double output) = 0;
  virtual double DerivLoss(double target, double output) = 0;
  virtual LossType GetLossType() = 0;
};
} // namespace deeplearning
