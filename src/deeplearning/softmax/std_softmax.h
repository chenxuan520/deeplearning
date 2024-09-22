#pragma once

#include "softmax_base.h"
#include <cmath>
namespace deeplearning {

class StdSoftmax : public SoftmaxFunction {
public:
  void Normalize(const std::vector<double> &before,
                 std::vector<double> &after) override {
    long double sum = 0;
    for (int i = 0; i < before.size(); i++) {
      sum += std::exp(before[i]);
    }
    if (sum == 0 || before.size() != after.size()) {
      return;
    }
    for (int i = 0; i < before.size(); i++) {
      after[i] = std::exp(before[i]) / sum;
    }
  }
  SoftmaxType GetSoftmaxType() override { return SOFTMAX_STD; }
};

} // namespace deeplearning
