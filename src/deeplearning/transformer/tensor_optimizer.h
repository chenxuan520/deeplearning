#pragma once

#include "optimizer/adam_optimizer.h"

#include <optional>
#include <vector>

namespace deeplearning {

// In-place Adam updater for a single parameter tensor (a weight matrix or a
// bias / LayerNorm-scale vector).
//
// It wraps the shared AdamOptimizer (Kingma & Ba) used by the MLP path so the
// Transformer trains with the exact same optimizer implementation. The layer
// trick maps a [rows x cols] weight matrix onto AdamOptimizer's layout by
// building it with layer = {cols, rows} (so weight state[1] has shape
// [rows][cols]); a length-D vector is mapped with layer = {D} and updated
// through the bias slot.
//
// The optimizer state is created lazily on first use and the type stays
// copyable (std::optional<AdamOptimizer>) so modules that hold it, such as
// TransformerBlock stored inside std::vector, remain copyable.
class TensorOptimizer {
public:
  // Updates a [rows x cols] weight matrix in place: param -= Adam(grad).
  void Apply(std::vector<std::vector<double>> &param,
             const std::vector<std::vector<double>> &grad,
             double learning_rate) {
    Apply(param, grad, learning_rate, 1.0);
  }

  // Updates a [rows x cols] weight matrix after scaling the supplied gradient.
  void Apply(std::vector<std::vector<double>> &param,
             const std::vector<std::vector<double>> &grad,
             double learning_rate, double gradient_scale) {
    if (param.empty() || param[0].empty()) {
      return;
    }
    const int rows = static_cast<int>(param.size());
    const int cols = static_cast<int>(param[0].size());
    if (!matrix_adam_) {
      matrix_adam_.emplace(std::vector<int>{cols, rows});
    }
    matrix_adam_->BeforeStep();
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        param[row][col] -= matrix_adam_->CalcChangeValue(
            grad[row][col] * gradient_scale, learning_rate, {1, row}, col,
            param[row][col]);
      }
    }
  }

  // Updates a length-D vector (bias / LayerNorm scale) in place.
  void Apply(std::vector<double> &param, const std::vector<double> &grad,
             double learning_rate) {
    Apply(param, grad, learning_rate, 1.0);
  }

  // Updates a length-D vector after scaling the supplied gradient.
  void Apply(std::vector<double> &param, const std::vector<double> &grad,
             double learning_rate, double gradient_scale) {
    if (param.empty()) {
      return;
    }
    const int size = static_cast<int>(param.size());
    if (!vector_adam_) {
      vector_adam_.emplace(std::vector<int>{size});
    }
    vector_adam_->BeforeStep();
    for (int i = 0; i < size; i++) {
      param[i] -= vector_adam_->CalcChangeValue(grad[i] * gradient_scale,
                                                learning_rate, {0, i}, -1,
                                                param[i]);
    }
  }

private:
  std::optional<AdamOptimizer> matrix_adam_;
  std::optional<AdamOptimizer> vector_adam_;
};

} // namespace deeplearning
