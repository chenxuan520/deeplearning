#pragma once

#include "activate_base.h"

namespace deeplearning {

class ReluActivate : public ActivateFunction {
public:
  double Activate(const double &x) override { return x > 0 ? x : 0; }

  double DerivActivate(const double &output) override {
    return output > 0 ? 1 : 0;
  }

  ActivateType GetActivateType() override {
    return ActivateType::ACTIVATE_RELU;
  }
};

} // namespace deeplearning
