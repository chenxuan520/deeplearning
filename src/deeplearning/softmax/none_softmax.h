#pragma once

#include "softmax_base.h"
#include <memory>
namespace deeplearning {

class NoneSoftmax : public SoftmaxFunction {
public:
  void Normalize(const std::vector<double> &input,
                 std::vector<double> &output) override;
  double CalcDelta(double output, double target,
                   std::shared_ptr<LossFunction> loss_function) override;
  SoftmaxType GetSoftmaxType() override;
};
} // namespace deeplearning
