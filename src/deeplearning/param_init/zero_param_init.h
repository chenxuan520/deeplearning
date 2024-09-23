#pragma once

#include "param_init_base.h"
namespace deeplearning {

class ZeroParamInitFunction : public ParamInitFunction {
public:
  void InitParam(std::vector<std::vector<std::vector<double>>> &weight,
                  std::vector<std::vector<double>> &bias) override {
    for (auto &w : weight) {
      for (auto &w_ : w) {
        std::fill(w_.begin(), w_.end(), 0.0);
      }
    }
    for (auto &b : bias) {
      std::fill(b.begin(), b.end(), 0.0);
    }
  }

  ParamInitType GetParamInitType() override {
    return ParamInitType::PARAM_INIT_ZERO;
  }
};

} // namespace deeplearning
