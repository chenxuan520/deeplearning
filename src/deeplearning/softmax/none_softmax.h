#pragma once
#include "softmax_base.h"

namespace deeplearning {
class NoneSoftmax : public SoftmaxFunction {
public:
  void Normalize(const std::vector<double> &before,
                 std::vector<double> &after) override {
    for (int i = 0; i < before.size(); i++) {
      after[i] = before[i];
    }
  }
  SoftmaxType GetSoftmaxType() override { return SOFTMAX_NONE; }
};
} // namespace deeplearning
