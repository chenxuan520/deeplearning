#pragma once

#include <vector>
namespace deeplearning {

enum SoftmaxType {
  SOFTMAX_NONE,
  SOFTMAX_STD,
};

class SoftmaxFunction {
public:
  virtual void Normalize(const std::vector<double> &before,
                         std::vector<double> &after) = 0;
  virtual SoftmaxType GetSoftmaxType() = 0;
};

} // namespace deeplearning
