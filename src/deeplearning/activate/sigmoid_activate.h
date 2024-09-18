#pragma once
#include "activate_base.h"
#include <cmath>

namespace deeplearning {

class SigmoidActivate : public ActivateFunction {
public:
  double Activate(const double &input) override {
    return 1 / (1 + exp(-input));
  }
  double DerivActivate(const double &output) override {
    return output * (1 - output);
  }
  ActivateType GetActivateType() override { return ACTIVATE_SIGMOID; }
};

} // namespace deeplearning
