#pragma once

#include <vector>
namespace deeplearning {

enum ParamInitType {
  PARAM_INIT_ZERO,
  PARAM_INIT_UNIFORM_RANDOM,
  PARAM_INIT_NORMAL_RANDOM,
  PARAM_INIT_XAVIER,
  PARAM_INIT_HE,
};

class ParamInitFunction {
public:
  virtual void InitParam(std::vector<std::vector<std::vector<double>>> &weight,
                         std::vector<std::vector<double>> &bias) = 0;
  virtual ParamInitType GetParamInitType() = 0;
};

} // namespace deeplearning
