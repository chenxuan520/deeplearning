#include "positional_encoding.h"

#include <cmath>

namespace deeplearning {

std::vector<std::vector<double>>
PositionalEncoding::Create(int sequence_length, int model_dim) {
  if (sequence_length <= 0 || model_dim <= 0) {
    return {};
  }

  std::vector<std::vector<double>> encoding(
      sequence_length, std::vector<double>(model_dim, 0));
  for (int pos = 0; pos < sequence_length; pos++) {
    for (int i = 0; i < model_dim; i += 2) {
      double angle =
          pos / std::pow(10000.0, static_cast<double>(i) / model_dim);
      encoding[pos][i] = std::sin(angle);
      if (i + 1 < model_dim) {
        encoding[pos][i + 1] = std::cos(angle);
      }
    }
  }
  return encoding;
}

bool PositionalEncoding::Apply(std::vector<std::vector<double>> &sequence) {
  if (sequence.empty() || sequence[0].empty()) {
    return false;
  }

  const int model_dim = sequence[0].size();
  for (const auto &token : sequence) {
    if (token.size() != model_dim) {
      return false;
    }
  }

  auto encoding = Create(sequence.size(), model_dim);
  for (int i = 0; i < sequence.size(); i++) {
    for (int j = 0; j < model_dim; j++) {
      sequence[i][j] += encoding[i][j];
    }
  }
  return true;
}

} // namespace deeplearning
