#pragma once

#include "softmax_base.h"
#include <memory>
namespace deeplearning {

class NoneSoftmax : public SoftmaxFunction {
public:
  void Normalize(const std::vector<double> &, std::vector<double> &) override {
    return;
  }
  double CalcDelta(double, double, std::shared_ptr<LossFunction>) override {
    return 0;
  }
  SoftmaxType GetSoftmaxType() override { return SOFTMAX_NONE; }

private:
};
} // namespace deeplearning
