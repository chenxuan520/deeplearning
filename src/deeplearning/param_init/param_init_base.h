#pragma once

#include <vector>
namespace deeplearning {

enum ParamInitType {
  PARAM_INIT_ZERO,
};

class ParamInitFunction {
public:
  virtual void InitParam(std::vector<std::vector<std::vector<double>>> &weight,
                          std::vector<std::vector<double>> &bias) = 0;
  virtual ParamInitType GetParamInitType() = 0;
};

} // namespace deeplearning
