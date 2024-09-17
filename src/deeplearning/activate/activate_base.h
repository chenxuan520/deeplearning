#pragma once

namespace deeplearning {

class ActivateFunction {
public:
  virtual double Activate(const double &input) = 0;
  virtual double DerivActivate(const double &output) = 0;
};

} // namespace deeplearning
