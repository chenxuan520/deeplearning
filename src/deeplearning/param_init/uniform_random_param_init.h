#pragma once

#include "param_init_base.h"
#include <random>

namespace deeplearning {

class UniformRandomParamInitFunction : public ParamInitFunction {
public:
  UniformRandomParamInitFunction();
  UniformRandomParamInitFunction(double min, double max);

  void InitParam(std::vector<std::vector<std::vector<double>>> &weight,
                 std::vector<std::vector<double>> &bias) override;

  ParamInitType GetParamInitType() override;

private:
  double min_;
  double max_;
};
} // namespace deeplearning
