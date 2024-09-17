#pragma once
#include "activate_base.h"
#include <cmath>

namespace deeplearning {

class SigmoidActivate : public ActivateFunction {
public:
  double Activate(const double &input) { return 1 / (1 + exp(-input)); }
  double DerivActivate(const double &output) { return output * (1 - output); }
};

} // namespace deeplearning
