#pragma once

#include "softmax_base.h"
#include <cmath>
#include <memory>
namespace deeplearning {

class StdSoftmax : public SoftmaxFunction {
public:
  void Normalize(const std::vector<double> &input,
                 std::vector<double> &output) override {
    long double sum = 0;
    for (int i = 0; i < input.size(); i++) {
      sum += std::exp(input[i]);
    }
    if (sum == 0 || input.size() != output.size()) {
      return;
    }
    for (int i = 0; i < input.size(); i++) {
      output[i] = std::exp(input[i]) / sum;
    }
  }
  double CalcDelta(double output, double target,
                   std::shared_ptr<LossFunction> loss_function) override {
    switch (loss_function->GetLossType()) {
    case LOSS_MSE:
      return output - target;
    case LOSS_CROSS_ENTROPY:
      return output - target;
    }
    return 0;
  }
  SoftmaxType GetSoftmaxType() override { return SOFTMAX_STD; }
};

} // namespace deeplearning
