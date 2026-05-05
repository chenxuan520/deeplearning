#pragma once

#include <vector>

namespace deeplearning {

class PositionalEncoding {
public:
  static std::vector<std::vector<double>> Create(int sequence_length,
                                                 int model_dim);
  static bool Apply(std::vector<std::vector<double>> &sequence);
};

} // namespace deeplearning
