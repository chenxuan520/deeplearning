#pragma once

// Finite-difference gradient checks for the hand-written Transformer backward
// passes. For a fixed random projection R, the scalar loss is
//   L(x) = sum_ij forward(x)_ij * R_ij
// so dL/dy = R. We feed R into BackwardAccumulate and compare the analytic
// dL/dx against the central difference (L(x + eps) - L(x - eps)) / (2 eps)
// for every input element. Any mistake in a backward formula (softmax,
// masking, head split/merge, LayerNorm, residual, FFN) shows up as a large
// relative error here, while a correct implementation lands around 1e-9.

#include "test.h"
#include "transformer/layer_norm.h"
#include "transformer/self_attention.h"
#include "transformer/transformer_block.h"
#include "transformer/transformer_decoder.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

namespace {

using GradCheckMatrix = std::vector<std::vector<double>>;

// Analytic and numerical gradients must agree to this relative error.
constexpr double kGradCheckTolerance = 1e-5;
constexpr double kGradCheckEpsilon = 1e-5;

GradCheckMatrix GradCheckRandomMatrix(int rows, int cols, unsigned seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<double> dist(0.0, 1.0);
  GradCheckMatrix matrix(rows, std::vector<double>(cols));
  for (auto &row : matrix) {
    for (auto &value : row) {
      value = dist(gen);
    }
  }
  return matrix;
}

GradCheckMatrix GradCheckCausalMask(int seq_len) {
  GradCheckMatrix mask(seq_len, std::vector<double>(seq_len, 1.0));
  for (int row = 0; row < seq_len; ++row) {
    for (int col = row + 1; col < seq_len; ++col) {
      mask[row][col] = 0.0;
    }
  }
  return mask;
}

// forward(x, y) must run a forward pass; backward(R, grad_x) must run
// BackwardAccumulate on the state left by the most recent forward(x).
// Returns the max relative error over all input elements, or a negative
// value if any call reports failure.
double MaxInputGradRelError(
    const std::function<bool(const GradCheckMatrix &, GradCheckMatrix &)>
        &forward,
    const std::function<bool(const GradCheckMatrix &, GradCheckMatrix &)>
        &backward,
    int seq_len, int dim) {
  const auto input = GradCheckRandomMatrix(seq_len, dim, 1);
  const auto projection = GradCheckRandomMatrix(seq_len, dim, 2);

  auto loss = [&](const GradCheckMatrix &x, bool &ok) {
    GradCheckMatrix y;
    ok = forward(x, y);
    double sum = 0.0;
    for (size_t i = 0; ok && i < y.size(); ++i) {
      for (size_t j = 0; j < y[i].size(); ++j) {
        sum += y[i][j] * projection[i][j];
      }
    }
    return sum;
  };

  GradCheckMatrix output, grad_input;
  if (!forward(input, output) || !backward(projection, grad_input)) {
    return -1.0;
  }
  if (grad_input.size() != input.size()) {
    return -1.0;
  }

  double max_rel_error = 0.0;
  for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < dim; ++j) {
      auto plus = input;
      auto minus = input;
      plus[i][j] += kGradCheckEpsilon;
      minus[i][j] -= kGradCheckEpsilon;
      bool ok_plus = false, ok_minus = false;
      const double numerical =
          (loss(plus, ok_plus) - loss(minus, ok_minus)) /
          (2.0 * kGradCheckEpsilon);
      if (!ok_plus || !ok_minus) {
        return -1.0;
      }
      const double analytic = grad_input[i][j];
      const double denom =
          std::max(1e-8, std::fabs(numerical) + std::fabs(analytic));
      max_rel_error =
          std::max(max_rel_error, std::fabs(numerical - analytic) / denom);
    }
  }
  return max_rel_error;
}

} // namespace

TEST(GradientCheck, SelfAttentionInput) {
  const int seq_len = 4, dim = 8;
  const auto mask = GradCheckCausalMask(seq_len);
  for (int head_num : {1, 2}) {
    for (bool use_mask : {false, true}) {
      deeplearning::SelfAttention attention;
      attention.set_random_seed(7);
      MUST_TRUE(attention.Init(dim, head_num) ==
                    deeplearning::SelfAttention::SUCCESS,
                attention.err_msg());
      const GradCheckMatrix *mask_ptr = use_mask ? &mask : nullptr;
      const double err = MaxInputGradRelError(
          [&](const GradCheckMatrix &x, GradCheckMatrix &y) {
            return attention.Forward(x, y, mask_ptr) ==
                   deeplearning::SelfAttention::SUCCESS;
          },
          [&](const GradCheckMatrix &grad_y, GradCheckMatrix &grad_x) {
            attention.ClearGradients();
            return attention.BackwardAccumulate(grad_y, grad_x) ==
                   deeplearning::SelfAttention::SUCCESS;
          },
          seq_len, dim);
      DEBUG("self attention heads=" << head_num << " mask=" << use_mask
                                    << " max rel err=" << err);
      MUST_TRUE(err >= 0.0 && err < kGradCheckTolerance,
                "self attention input gradient mismatch, heads=" +
                    std::to_string(head_num) +
                    " mask=" + std::to_string(use_mask) +
                    " err=" + std::to_string(err));
    }
  }
}

TEST(GradientCheck, LayerNormInput) {
  const int seq_len = 4, dim = 8;
  deeplearning::LayerNorm layer_norm;
  MUST_TRUE(layer_norm.Init(dim) == deeplearning::LayerNorm::SUCCESS,
            "layer norm init failed");
  const double err = MaxInputGradRelError(
      [&](const GradCheckMatrix &x, GradCheckMatrix &y) {
        return layer_norm.Forward(x, y) == deeplearning::LayerNorm::SUCCESS;
      },
      [&](const GradCheckMatrix &grad_y, GradCheckMatrix &grad_x) {
        layer_norm.ClearGradients();
        return layer_norm.BackwardAccumulate(grad_y, grad_x) ==
               deeplearning::LayerNorm::SUCCESS;
      },
      seq_len, dim);
  DEBUG("layer norm max rel err=" << err);
  MUST_TRUE(err >= 0.0 && err < kGradCheckTolerance,
            "layer norm input gradient mismatch, err=" + std::to_string(err));
}

TEST(GradientCheck, TransformerBlockInput) {
  const int seq_len = 4, dim = 8, feed_forward_dim = 16;
  const auto mask = GradCheckCausalMask(seq_len);
  for (int head_num : {1, 2}) {
    for (bool use_mask : {false, true}) {
      deeplearning::TransformerBlock block;
      block.set_random_seed(7);
      MUST_TRUE(block.Init(dim, head_num, feed_forward_dim) ==
                    deeplearning::TransformerBlock::SUCCESS,
                "transformer block init failed");
      const GradCheckMatrix *mask_ptr = use_mask ? &mask : nullptr;
      const double err = MaxInputGradRelError(
          [&](const GradCheckMatrix &x, GradCheckMatrix &y) {
            return block.Forward(x, y, mask_ptr) ==
                   deeplearning::TransformerBlock::SUCCESS;
          },
          [&](const GradCheckMatrix &grad_y, GradCheckMatrix &grad_x) {
            block.ClearGradients();
            return block.BackwardAccumulate(grad_y, grad_x) ==
                   deeplearning::TransformerBlock::SUCCESS;
          },
          seq_len, dim);
      DEBUG("transformer block heads=" << head_num << " mask=" << use_mask
                                       << " max rel err=" << err);
      MUST_TRUE(err >= 0.0 && err < kGradCheckTolerance,
                "transformer block input gradient mismatch, heads=" +
                    std::to_string(head_num) +
                    " mask=" + std::to_string(use_mask) +
                    " err=" + std::to_string(err));
    }
  }
}

TEST(GradientCheck, TransformerDecoderInput) {
  const int block_num = 2, seq_len = 4, dim = 8, head_num = 2,
            feed_forward_dim = 16;
  deeplearning::TransformerDecoder decoder;
  decoder.set_random_seed(7);
  MUST_TRUE(decoder.Init(block_num, dim, head_num, feed_forward_dim) ==
                deeplearning::TransformerDecoder::SUCCESS,
            "transformer decoder init failed");
  const double err = MaxInputGradRelError(
      [&](const GradCheckMatrix &x, GradCheckMatrix &y) {
        return decoder.Forward(x, y) ==
               deeplearning::TransformerDecoder::SUCCESS;
      },
      [&](const GradCheckMatrix &grad_y, GradCheckMatrix &grad_x) {
        decoder.ClearGradients();
        return decoder.BackwardAccumulate(grad_y, grad_x) ==
               deeplearning::TransformerDecoder::SUCCESS;
      },
      seq_len, dim);
  DEBUG("transformer decoder max rel err=" << err);
  MUST_TRUE(err >= 0.0 && err < kGradCheckTolerance,
            "transformer decoder input gradient mismatch, err=" +
                std::to_string(err));
}
