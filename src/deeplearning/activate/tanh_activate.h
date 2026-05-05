#pragma once

#include "activate_base.h"
#include <cmath>

namespace deeplearning {
class TanhActivate : public ActivateFunction {
public:
  double Activate(const double &input) override;
  double DerivActivate(const double &output) override;
  ActivateType GetActivateType() override;
};

} // namespace deeplearning
