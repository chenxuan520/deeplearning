#include "normal_random_param_init.h"

namespace deeplearning {

NormalRandomParamInitFunction::NormalRandomParamInitFunction()
    : mean_(0.0), stddev_(1.0) {}

NormalRandomParamInitFunction::NormalRandomParamInitFunction(double mean,
                                                             double stddev)
    : mean_(mean), stddev_(stddev) {}

void NormalRandomParamInitFunction::InitParam(
    std::vector<std::vector<std::vector<double>>> &weight,
    std::vector<std::vector<double>> &bias) {
  std::mt19937 gen;
  if (seed_ < 0) {
    std::random_device rd;
    gen.seed(rd());
  } else {
    gen.seed(static_cast<uint32_t>(seed_) + 0x517CC1B7u);
  }
  std::normal_distribution<double> distr(mean_, stddev_);

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

ParamInitType NormalRandomParamInitFunction::GetParamInitType() {
  return ParamInitType::PARAM_INIT_NORMAL_RANDOM;
}

} // namespace deeplearning
