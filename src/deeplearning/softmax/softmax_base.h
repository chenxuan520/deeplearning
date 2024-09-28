#pragma once

#include "loss/loss_base.h"
#include <memory>
#include <utility>
#include <vector>
namespace deeplearning {

enum SoftmaxType {
  SOFTMAX_NONE,
  SOFTMAX_STD,
};

class SoftmaxFunction {
public:
  virtual void Normalize(const std::vector<double> &input,
                         std::vector<double> &output) = 0;
  virtual double CalcDelta(double output, double target,
                           std::shared_ptr<LossFunction> loss_function) = 0;
  virtual SoftmaxType GetSoftmaxType() = 0;
};

} // namespace deeplearning
