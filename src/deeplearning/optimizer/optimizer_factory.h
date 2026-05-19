#pragma once

#include "optimizer/optimizer_base.h"
#include <memory>
#include <vector>
namespace deeplearning {

class OptimizerFactory {
public:
  static std::shared_ptr<OptimizerFunction>
  Create(OptimizerType optimizer_type, const std::vector<int> &layer);
};

} // namespace deeplearning
