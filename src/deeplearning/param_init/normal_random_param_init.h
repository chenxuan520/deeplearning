#pragma once

#include "param_init_base.h"
#include <random>

namespace deeplearning {

class NormalRandomParamInitFunction : public ParamInitFunction {
public:
  NormalRandomParamInitFunction();
  NormalRandomParamInitFunction(double mean, double stddev);

  void InitParam(std::vector<std::vector<std::vector<double>>> &weight,
                 std::vector<std::vector<double>> &bias) override;

  ParamInitType GetParamInitType() override;

private:
  double mean_;
  double stddev_;
};

} // namespace deeplearning
