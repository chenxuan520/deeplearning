#include "uniform_random_param_init.h"

namespace deeplearning {

UniformRandomParamInitFunction::UniformRandomParamInitFunction()
    : min_(-0.5), max_(0.5) {}

UniformRandomParamInitFunction::UniformRandomParamInitFunction(double min,
                                                               double max)
    : min_(min), max_(max) {}

void UniformRandomParamInitFunction::InitParam(
    std::vector<std::vector<std::vector<double>>> &weight,
    std::vector<std::vector<double>> &bias) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distr(min_, max_);

  for (auto &w : weight) {
    for (auto &w_ : w) {
      for (auto &w__ : w_) {
        w__ = distr(gen);
      }
    }
  }
  for (auto &b : bias) {
    for (auto &b_ : b) {
      b_ = distr(gen);
    }
  }
}

ParamInitType UniformRandomParamInitFunction::GetParamInitType() {
  return ParamInitType::PARAM_INIT_UNIFORM_RANDOM;
}

} // namespace deeplearning
