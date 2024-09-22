#pragma once

namespace deeplearning {

enum ActivateType {
  ACTIVATE_SIGMOID,
  ACTIVATE_RELU,
  ACTIVATE_TANH,
};

class ActivateFunction {
public:
  virtual double Activate(const double &input) = 0;
  virtual double DerivActivate(const double &output) = 0;
  virtual ActivateType GetActivateType() = 0;
};

} // namespace deeplearning
