#pragma once

#include "activate_base.h"

namespace deeplearning {

class ReluActivate : public ActivateFunction {
public:
  double Activate(const double &x) override;
  double DerivActivate(const double &output) override;
  ActivateType GetActivateType() override;
};

} // namespace deeplearning
