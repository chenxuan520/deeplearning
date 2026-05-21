#pragma once

#include <vector>

namespace deeplearning {

enum LossType {
  LOSS_MSE,
  LOSS_CROSS_ENTROPY,
};

class LossFunction {
public:
  virtual ~LossFunction() = default;

  virtual double AverageLoss(const std::vector<double> &target,
                             const std::vector<double> &output);

  virtual double Loss(double target, double output) = 0;
  virtual double DerivLoss(double target, double output) = 0;
  virtual LossType GetLossType() = 0;
};
} // namespace deeplearning
