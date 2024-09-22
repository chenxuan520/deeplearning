#pragma once

#include "activate_base.h"
#include <cmath>

namespace deeplearning {
class TanhActivate : public ActivateFunction {
public:
  double Activate(const double &input) override {
    return (1 - exp(-2 * input)) / (1 + exp(-2 * input));
  }
  double DerivActivate(const double &output) override {
    return 1 - output * output;
  }
  ActivateType GetActivateType() override { return ACTIVATE_TANH; }
};

} // namespace deeplearning
