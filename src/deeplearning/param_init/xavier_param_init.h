#pragma once

#include "param_init_base.h"
#include <random>

namespace deeplearning {

class XavierParamInitFunction : public ParamInitFunction {
public:
  void InitParam(std::vector<std::vector<std::vector<double>>> &weight,
                 std::vector<std::vector<double>> &bias) override;

  ParamInitType GetParamInitType() override;
};

} // namespace deeplearning
