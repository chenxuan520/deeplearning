#pragma once

#include "cnn/batch_norm2d.h"
#include "cnn/batched_conv2d.h"
#include "cnn/conv2d.h"
#include "cnn/float_linear.h"
#include "cnn/max_pool2d.h"
#include "cnn/mini_cnn_classifier.h"
#include "cnn/policy_value_resnet.h"
#include "cnn/residual_block2d.h"
#include "loss/policy_value_loss.h"
#include "loss/policy_value_wdl_loss.h"
#include "optimizer/float_adamw.h"
#include "optimizer/float_sgd.h"
#include "test.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {

using namespace deeplearning;

bool CnnNearlyEqual(double lhs, double rhs, double eps = 1e-6) {
  return std::fabs(lhs - rhs) <= eps;
}

Conv2D::Tensor3D MakeImage(const std::vector<std::string> &rows) {
  Conv2D::Tensor3D image(1,
                         std::vector<std::vector<double>>(
                             rows.size(), std::vector<double>(rows[0].size(), 0.0)));
  for (int row = 0; row < static_cast<int>(rows.size()); row++) {
    for (int col = 0; col < static_cast<int>(rows[row].size()); col++) {
      image[0][row][col] = rows[row][col] == '1' ? 1.0 : 0.0;
    }
  }
  return image;
}

double CNNAccuracy(MiniCNNClassifier &model,
                   const std::vector<Conv2D::Tensor3D> &images,
                   const std::vector<int> &labels) {
  int correct = 0;
  for (int i = 0; i < static_cast<int>(images.size()); i++) {
    int pred = -1;
    if (model.Predict(images[i], pred) != MiniCNNClassifier::SUCCESS) {
      return -1.0;
    }
    if (pred == labels[i]) {
      correct++;
    }
  }
  return correct * 1.0 / images.size();
}

void MakeCNNToyData(std::vector<Conv2D::Tensor3D> &images,
                    std::vector<int> &labels) {
  images = {
      MakeImage({"0100", "0100", "0100", "0100"}),
      MakeImage({"0010", "0010", "0010", "0010"}),
      MakeImage({"0100", "0100", "0110", "0100"}),
      MakeImage({"0010", "0010", "0011", "0010"}),
      MakeImage({"0000", "1111", "0000", "0000"}),
      MakeImage({"0000", "0000", "1111", "0000"}),
      MakeImage({"0000", "1111", "0010", "0000"}),
      MakeImage({"0000", "0010", "1111", "0000"}),
  };
  labels = {0, 0, 0, 0, 1, 1, 1, 1};
}

MiniCNNClassifier::Config MakeCNNToyConfig() {
  MiniCNNClassifier::Config config;
  config.input_channels_ = 1;
  config.input_height_ = 4;
  config.input_width_ = 4;
  config.conv_channels_ = 3;
  config.kernel_height_ = 2;
  config.kernel_width_ = 2;
  config.pool_height_ = 2;
  config.pool_width_ = 2;
  config.pool_stride_ = 2;
  config.class_num_ = 2;
  config.rand_seed_ = 7;
  return config;
}

bool SameCNNParams(MiniCNNClassifier &lhs, MiniCNNClassifier &rhs,
                   double eps) {
  const auto &lfw = lhs.fc_weight();
  const auto &rfw = rhs.fc_weight();
  const auto &lfb = lhs.fc_bias();
  const auto &rfb = rhs.fc_bias();
  if (lfw.size() != rfw.size() || lfb.size() != rfb.size()) {
    return false;
  }
  for (int cls = 0; cls < static_cast<int>(lfw.size()); cls++) {
    if (lfw[cls].size() != rfw[cls].size()) {
      return false;
    }
    if (!CnnNearlyEqual(lfb[cls], rfb[cls], eps)) {
      return false;
    }
    for (int dim = 0; dim < static_cast<int>(lfw[cls].size()); dim++) {
      if (!CnnNearlyEqual(lfw[cls][dim], rfw[cls][dim], eps)) {
        return false;
      }
    }
  }

  const auto &lw = lhs.conv().weight();
  const auto &rw = rhs.conv().weight();
  const auto &lb = lhs.conv().bias();
  const auto &rb = rhs.conv().bias();
  if (lw.size() != rw.size() || lb.size() != rb.size()) {
    return false;
  }
  for (int oc = 0; oc < static_cast<int>(lw.size()); oc++) {
    if (!CnnNearlyEqual(lb[oc], rb[oc], eps)) {
      return false;
    }
    for (int ic = 0; ic < static_cast<int>(lw[oc].size()); ic++) {
      for (int kh = 0; kh < static_cast<int>(lw[oc][ic].size()); kh++) {
        for (int kw = 0; kw < static_cast<int>(lw[oc][ic][kh].size()); kw++) {
          if (!CnnNearlyEqual(lw[oc][ic][kh][kw], rw[oc][ic][kh][kw], eps)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

} // namespace

TEST(Conv2D, ForwardSingleKernel) {
  Conv2D conv;
  conv.set_random_seed(0);
  MUST_EQUAL(conv.Init(1, 1, 2, 2), Conv2D::SUCCESS);
  MUST_EQUAL(conv.set_weight({{{{1.0, 0.0}, {0.0, -1.0}}}}), Conv2D::SUCCESS);
  MUST_EQUAL(conv.set_bias({0.0}), Conv2D::SUCCESS);

  Conv2D::Tensor3D input = {{{1.0, 2.0, 3.0},
                             {4.0, 5.0, 6.0},
                             {7.0, 8.0, 9.0}}};
  Conv2D::Tensor3D output;
  MUST_EQUAL(conv.Forward(input, output), Conv2D::SUCCESS);
  MUST_EQUAL(output.size(), 1);
  MUST_TRUE(CnnNearlyEqual(output[0][0][0], -4.0), "top-left mismatch");
  MUST_TRUE(CnnNearlyEqual(output[0][0][1], -4.0), "top-right mismatch");
  MUST_TRUE(CnnNearlyEqual(output[0][1][0], -4.0), "bottom-left mismatch");
  MUST_TRUE(CnnNearlyEqual(output[0][1][1], -4.0), "bottom-right mismatch");
}

TEST(Conv2D, BackwardRejectsNegativeLearningRate) {
  Conv2D conv;
  conv.set_random_seed(0);
  MUST_EQUAL(conv.Init(1, 1, 2, 2), Conv2D::SUCCESS);
  Conv2D::Tensor3D input = {{{1.0, 2.0}, {3.0, 4.0}}};
  Conv2D::Tensor3D output;
  MUST_EQUAL(conv.Forward(input, output), Conv2D::SUCCESS);

  Conv2D::Tensor3D grad_output = {{{1.0}}};
  Conv2D::Tensor3D grad_input;
  MUST_EQUAL(conv.Backward(grad_output, grad_input, -0.1),
             Conv2D::INVALID_DATA);
}

TEST(Conv2D, BackwardKeepsLegacyErrorPriority) {
  Conv2D::Tensor3D grad_output = {{{1.0}}};
  Conv2D::Tensor3D grad_input;

  Conv2D conv;
  MUST_EQUAL(conv.Backward(grad_output, grad_input, -0.1), Conv2D::NOT_INIT);
  MUST_TRUE(conv.err_msg() == "[Conv2D::Backward] Conv2D not init",
            "not-init error should keep old Backward message");

  conv.set_random_seed(0);
  MUST_EQUAL(conv.Init(1, 1, 2, 2), Conv2D::SUCCESS);
  MUST_EQUAL(conv.Backward(grad_output, grad_input, -0.1),
             Conv2D::INVALID_DATA);
  MUST_TRUE(conv.err_msg() == "[Conv2D::Backward] Missing forward cache",
            "missing-cache error should keep priority over learning rate");

  Conv2D::Tensor3D input = {{{1.0, 2.0}, {3.0, 4.0}}};
  Conv2D::Tensor3D output;
  MUST_EQUAL(conv.Forward(input, output), Conv2D::SUCCESS);
  Conv2D::Tensor3D invalid_grad_output = {{{1.0, 2.0}}};
  MUST_EQUAL(conv.Backward(invalid_grad_output, grad_input, 0.1),
             Conv2D::INVALID_DATA);
  MUST_TRUE(conv.err_msg() == "[Conv2D::Backward] Invalid grad_output shape",
            "shape error should keep old Backward message");
}

TEST(BatchedConv2D, KnownBatchForward) {
  BatchedConv2D conv;
  BatchedConv2D::Config config;
  config.input_channels_ = 1;
  config.output_channels_ = 1;
  config.kernel_height_ = 2;
  config.kernel_width_ = 2;
  config.thread_num_ = 2;
  MUST_EQUAL(conv.Init(config), BatchedConv2D::SUCCESS);
  MUST_EQUAL(conv.set_weight({1.0f, 0.0f, 0.0f, -1.0f}),
             BatchedConv2D::SUCCESS);
  MUST_EQUAL(conv.set_bias({0.5f}), BatchedConv2D::SUCCESS);

  FloatTensor4D input(2, 1, 3, 3);
  for (int row = 0; row < 3; row++) {
    for (int column = 0; column < 3; column++) {
      input(0, 0, row, column) = static_cast<float>(row * 3 + column + 1);
      input(1, 0, row, column) =
          static_cast<float>((row * 3 + column + 1) * 2);
    }
  }
  FloatTensor4D output;
  MUST_EQUAL(conv.Forward(input, output), BatchedConv2D::SUCCESS);
  MUST_EQUAL(output.batch(), 2);
  MUST_EQUAL(output.channels(), 1);
  MUST_EQUAL(output.height(), 2);
  MUST_EQUAL(output.width(), 2);
  MUST_TRUE(CnnNearlyEqual(output(0, 0, 0, 0), -3.5),
            "batch 0 forward mismatch");
  MUST_TRUE(CnnNearlyEqual(output(1, 0, 1, 1), -7.5),
            "batch 1 forward mismatch");
}

TEST(BatchedConv2D, BatchMatchesSingleInference) {
  BatchedConv2D batch_conv;
  BatchedConv2D single_conv;
  BatchedConv2D::Config config;
  config.input_channels_ = 2;
  config.output_channels_ = 3;
  config.kernel_height_ = 3;
  config.kernel_width_ = 3;
  config.padding_ = 1;
  config.thread_num_ = 3;
  MUST_EQUAL(batch_conv.Init(config), BatchedConv2D::SUCCESS);
  MUST_EQUAL(single_conv.Init(config), BatchedConv2D::SUCCESS);
  MUST_EQUAL(single_conv.set_weight(batch_conv.weight()),
             BatchedConv2D::SUCCESS);
  MUST_EQUAL(single_conv.set_bias(batch_conv.bias()),
             BatchedConv2D::SUCCESS);

  FloatTensor4D input(2, 2, 4, 5);
  for (int batch = 0; batch < 2; batch++) {
    for (int channel = 0; channel < 2; channel++) {
      for (int row = 0; row < 4; row++) {
        for (int column = 0; column < 5; column++) {
          input(batch, channel, row, column) =
              static_cast<float>((batch + 1) * 0.1 + channel * 0.2 +
                                 row * 0.03 - column * 0.01);
        }
      }
    }
  }
  FloatTensor4D batch_output;
  MUST_EQUAL(batch_conv.Forward(input, batch_output),
             BatchedConv2D::SUCCESS);
  for (int batch = 0; batch < 2; batch++) {
    FloatTensor4D single_input(1, 2, 4, 5);
    for (int channel = 0; channel < 2; channel++) {
      for (int row = 0; row < 4; row++) {
        for (int column = 0; column < 5; column++) {
          single_input(0, channel, row, column) =
              input(batch, channel, row, column);
        }
      }
    }
    FloatTensor4D single_output;
    MUST_EQUAL(single_conv.Forward(single_input, single_output),
               BatchedConv2D::SUCCESS);
    for (int channel = 0; channel < 3; channel++) {
      for (int row = 0; row < 4; row++) {
        for (int column = 0; column < 5; column++) {
          MUST_TRUE(
              CnnNearlyEqual(batch_output(batch, channel, row, column),
                             single_output(0, channel, row, column), 1e-5),
              "batch and single inference differ");
        }
      }
    }
  }
}

TEST(BatchedConv2D, FiniteDifferenceGradient) {
  BatchedConv2D conv;
  BatchedConv2D::Config config;
  config.input_channels_ = 1;
  config.output_channels_ = 1;
  config.kernel_height_ = 2;
  config.kernel_width_ = 2;
  config.use_bias_ = true;
  MUST_EQUAL(conv.Init(config), BatchedConv2D::SUCCESS);
  MUST_EQUAL(conv.set_weight({0.2f, -0.1f, 0.05f, 0.3f}),
             BatchedConv2D::SUCCESS);
  MUST_EQUAL(conv.set_bias({-0.02f}), BatchedConv2D::SUCCESS);

  FloatTensor4D input(1, 1, 3, 3);
  for (int row = 0; row < 3; row++) {
    for (int column = 0; column < 3; column++) {
      input(0, 0, row, column) =
          static_cast<float>(0.1 + row * 0.07 + column * 0.03);
    }
  }
  FloatTensor4D output;
  MUST_EQUAL(conv.Forward(input, output, true), BatchedConv2D::SUCCESS);
  FloatTensor4D grad_output(1, 1, 2, 2);
  grad_output(0, 0, 0, 0) = 0.4f;
  grad_output(0, 0, 0, 1) = -0.2f;
  grad_output(0, 0, 1, 0) = 0.1f;
  grad_output(0, 0, 1, 1) = 0.3f;
  conv.ZeroGrad();
  FloatTensor4D grad_input;
  MUST_EQUAL(conv.Backward(grad_output, grad_input),
             BatchedConv2D::SUCCESS);

  auto objective = [&](const FloatTensor4D &value) {
    FloatTensor4D prediction;
    if (conv.Forward(value, prediction) != BatchedConv2D::SUCCESS) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    double sum = 0.0;
    for (int row = 0; row < 2; row++) {
      for (int column = 0; column < 2; column++) {
        sum += prediction(0, 0, row, column) *
               grad_output(0, 0, row, column);
      }
    }
    return sum;
  };

  const float epsilon = 1e-3f;
  const float original_weight = conv.mutable_weight()[2];
  conv.mutable_weight()[2] = original_weight + epsilon;
  const double weight_plus = objective(input);
  conv.mutable_weight()[2] = original_weight - epsilon;
  const double weight_minus = objective(input);
  conv.mutable_weight()[2] = original_weight;
  const double numeric_weight =
      (weight_plus - weight_minus) / (2.0 * epsilon);
  MUST_TRUE(CnnNearlyEqual(conv.grad_weight()[2], numeric_weight, 2e-4),
            "weight finite difference mismatch");

  FloatTensor4D input_plus = input;
  FloatTensor4D input_minus = input;
  input_plus(0, 0, 1, 1) += epsilon;
  input_minus(0, 0, 1, 1) -= epsilon;
  const double numeric_input =
      (objective(input_plus) - objective(input_minus)) / (2.0 * epsilon);
  MUST_TRUE(CnnNearlyEqual(grad_input(0, 0, 1, 1), numeric_input, 2e-4),
            "input finite difference mismatch");
}

TEST(BatchNorm2D, NormalizeAndUseRunningStatistics) {
  BatchNorm2D normalization;
  BatchNorm2D::Config config;
  config.channels_ = 1;
  config.momentum_ = 1.0f;
  config.thread_num_ = 2;
  MUST_EQUAL(normalization.Init(config), BatchNorm2D::SUCCESS);
  MUST_EQUAL(normalization.set_scale({2.0f}), BatchNorm2D::SUCCESS);
  MUST_EQUAL(normalization.set_bias({0.5f}), BatchNorm2D::SUCCESS);

  FloatTensor4D input(1, 1, 2, 2);
  input.values() = {1.0f, 2.0f, 3.0f, 4.0f};
  FloatTensor4D training_output;
  MUST_EQUAL(normalization.Forward(input, training_output, true),
             BatchNorm2D::SUCCESS);
  const double mean = 2.5;
  const double inverse_std = 1.0 / std::sqrt(1.25 + config.epsilon_);
  MUST_TRUE(
      CnnNearlyEqual(training_output(0, 0, 0, 0),
                     2.0 * (1.0 - mean) * inverse_std + 0.5, 1e-5),
      "batch norm training output mismatch");
  MUST_TRUE(CnnNearlyEqual(normalization.running_mean()[0], 2.5, 1e-6),
            "running mean mismatch");
  MUST_TRUE(CnnNearlyEqual(normalization.running_variance()[0], 1.25, 1e-6),
            "running variance mismatch");

  FloatTensor4D inference_output;
  MUST_EQUAL(normalization.Forward(input, inference_output, false),
             BatchNorm2D::SUCCESS);
  for (std::size_t index = 0; index < training_output.size(); index++) {
    MUST_TRUE(CnnNearlyEqual(training_output.values()[index],
                             inference_output.values()[index], 1e-5),
              "training and inference output differ");
  }
}

TEST(BatchNorm2D, FrozenRunningStatisticsForwardAndBackward) {
  BatchNorm2D normalization;
  BatchNorm2D::Config config;
  config.channels_ = 1;
  config.momentum_ = 0.7f;
  MUST_EQUAL(normalization.Init(config), BatchNorm2D::SUCCESS);
  MUST_EQUAL(normalization.set_scale({1.5f}), BatchNorm2D::SUCCESS);
  MUST_EQUAL(normalization.set_bias({-0.25f}), BatchNorm2D::SUCCESS);
  MUST_EQUAL(normalization.set_running_mean({2.0f}), BatchNorm2D::SUCCESS);
  MUST_EQUAL(normalization.set_running_variance({4.0f}),
             BatchNorm2D::SUCCESS);

  FloatTensor4D input(1, 1, 1, 3);
  input.values() = {1.0f, 2.0f, 5.0f};
  FloatTensor4D frozen_output;
  MUST_EQUAL(normalization.Forward(input, frozen_output, true, true),
             BatchNorm2D::SUCCESS);
  MUST_TRUE(normalization.running_mean() == std::vector<float>({2.0f}),
            "frozen running mean changed");
  MUST_TRUE(normalization.running_variance() == std::vector<float>({4.0f}),
            "frozen running variance changed");

  FloatTensor4D inference_output;
  MUST_EQUAL(normalization.Forward(input, inference_output, false),
             BatchNorm2D::SUCCESS);
  for (std::size_t index = 0; index < frozen_output.size(); index++) {
    MUST_TRUE(CnnNearlyEqual(frozen_output.values()[index],
                             inference_output.values()[index], 1e-6),
              "frozen and inference outputs differ");
  }

  // Recreate the training cache after the inference call.
  MUST_EQUAL(normalization.Forward(input, frozen_output, true, true),
             BatchNorm2D::SUCCESS);
  FloatTensor4D grad_output(1, 1, 1, 3);
  grad_output.values() = {0.2f, -0.4f, 0.7f};
  normalization.ZeroGrad();
  FloatTensor4D grad_input;
  MUST_EQUAL(normalization.Backward(grad_output, grad_input),
             BatchNorm2D::SUCCESS);
  const double inverse_std = 1.0 / std::sqrt(4.0 + config.epsilon_);
  for (std::size_t index = 0; index < grad_input.size(); index++) {
    MUST_TRUE(CnnNearlyEqual(
                  grad_input.values()[index],
                  grad_output.values()[index] * 1.5 * inverse_std, 1e-6),
              "frozen input gradient mismatch");
  }
  const double expected_bias_gradient = 0.2 - 0.4 + 0.7;
  const double expected_scale_gradient =
      0.2 * (1.0 - 2.0) * inverse_std +
      -0.4 * (2.0 - 2.0) * inverse_std +
      0.7 * (5.0 - 2.0) * inverse_std;
  MUST_TRUE(CnnNearlyEqual(normalization.grad_bias()[0],
                           expected_bias_gradient, 1e-6),
            "frozen bias gradient mismatch");
  MUST_TRUE(CnnNearlyEqual(normalization.grad_scale()[0],
                           expected_scale_gradient, 1e-6),
            "frozen scale gradient mismatch");
}

TEST(BatchNorm2D, FiniteDifferenceGradient) {
  BatchNorm2D normalization;
  BatchNorm2D::Config config;
  config.channels_ = 2;
  config.momentum_ = 0.0f;
  MUST_EQUAL(normalization.Init(config), BatchNorm2D::SUCCESS);
  MUST_EQUAL(normalization.set_scale({1.2f, 0.7f}), BatchNorm2D::SUCCESS);
  MUST_EQUAL(normalization.set_bias({0.1f, -0.2f}), BatchNorm2D::SUCCESS);

  FloatTensor4D input(2, 2, 2, 2);
  for (int batch = 0; batch < 2; batch++) {
    for (int channel = 0; channel < 2; channel++) {
      for (int row = 0; row < 2; row++) {
        for (int column = 0; column < 2; column++) {
          input(batch, channel, row, column) =
              static_cast<float>(0.2 + batch * 0.3 + channel * 0.4 +
                                 row * 0.11 - column * 0.07);
        }
      }
    }
  }
  FloatTensor4D grad_output(2, 2, 2, 2);
  for (std::size_t index = 0; index < grad_output.size(); index++) {
    grad_output.values()[index] =
        static_cast<float>((static_cast<int>(index) % 5 - 2) * 0.13);
  }

  FloatTensor4D output;
  MUST_EQUAL(normalization.Forward(input, output, true),
             BatchNorm2D::SUCCESS);
  normalization.ZeroGrad();
  FloatTensor4D grad_input;
  MUST_EQUAL(normalization.Backward(grad_output, grad_input),
             BatchNorm2D::SUCCESS);

  auto objective = [&](const FloatTensor4D &value) {
    FloatTensor4D prediction;
    if (normalization.Forward(value, prediction, true) !=
        BatchNorm2D::SUCCESS) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    double sum = 0.0;
    for (std::size_t index = 0; index < prediction.size(); index++) {
      sum += prediction.values()[index] * grad_output.values()[index];
    }
    return sum;
  };
  const float epsilon = 1e-3f;
  FloatTensor4D input_plus = input;
  FloatTensor4D input_minus = input;
  input_plus(1, 0, 1, 0) += epsilon;
  input_minus(1, 0, 1, 0) -= epsilon;
  const double numeric_input =
      (objective(input_plus) - objective(input_minus)) / (2.0 * epsilon);
  MUST_TRUE(CnnNearlyEqual(grad_input(1, 0, 1, 0), numeric_input, 5e-4),
            "batch norm input gradient mismatch");

  const float original_scale = normalization.mutable_scale()[1];
  normalization.mutable_scale()[1] = original_scale + epsilon;
  const double scale_plus = objective(input);
  normalization.mutable_scale()[1] = original_scale - epsilon;
  const double scale_minus = objective(input);
  normalization.mutable_scale()[1] = original_scale;
  const double numeric_scale =
      (scale_plus - scale_minus) / (2.0 * epsilon);
  MUST_TRUE(CnnNearlyEqual(normalization.grad_scale()[1], numeric_scale, 5e-4),
            "batch norm scale gradient mismatch");
}

TEST(ResidualBlock2D, ZeroBranchIsReluIdentity) {
  ResidualBlock2D block;
  ResidualBlock2D::Config config;
  config.channels_ = 2;
  config.height_ = 2;
  config.width_ = 3;
  config.thread_num_ = 2;
  MUST_EQUAL(block.Init(config), ResidualBlock2D::SUCCESS);
  MUST_EQUAL(
      block.conv1().set_weight(
          std::vector<float>(block.conv1().weight().size(), 0.0f)),
      BatchedConv2D::SUCCESS);
  MUST_EQUAL(
      block.conv2().set_weight(
          std::vector<float>(block.conv2().weight().size(), 0.0f)),
      BatchedConv2D::SUCCESS);

  FloatTensor4D input(2, 2, 2, 3);
  for (std::size_t index = 0; index < input.size(); index++) {
    input.values()[index] =
        static_cast<float>((static_cast<int>(index) % 7 - 3) * 0.2);
  }
  FloatTensor4D output;
  MUST_EQUAL(block.Forward(input, output, true),
             ResidualBlock2D::SUCCESS);
  for (std::size_t index = 0; index < input.size(); index++) {
    MUST_TRUE(CnnNearlyEqual(output.values()[index],
                             std::max(0.0f, input.values()[index]), 1e-6),
              "zero residual branch is not ReLU identity");
  }

  FloatTensor4D grad_output(2, 2, 2, 3, 1.0f);
  FloatTensor4D grad_input;
  block.ZeroGrad();
  MUST_EQUAL(block.Backward(grad_output, grad_input),
             ResidualBlock2D::SUCCESS);
  for (std::size_t index = 0; index < input.size(); index++) {
    const float expected = input.values()[index] > 0.0f ? 1.0f : 0.0f;
    MUST_TRUE(CnnNearlyEqual(grad_input.values()[index], expected, 1e-6),
              "skip gradient mismatch");
  }
}

TEST(ResidualBlock2D, FiniteDifferenceGradient) {
  ResidualBlock2D block;
  ResidualBlock2D::Config config;
  config.channels_ = 1;
  config.height_ = 2;
  config.width_ = 2;
  config.thread_num_ = 1;
  config.batch_norm_momentum_ = 0.0f;
  MUST_EQUAL(block.Init(config), ResidualBlock2D::SUCCESS);
  std::vector<float> conv1_weight(block.conv1().weight().size(), 0.03f);
  std::vector<float> conv2_weight(block.conv2().weight().size(), 0.02f);
  conv1_weight[4] = 0.4f;
  conv2_weight[4] = 0.3f;
  MUST_EQUAL(block.conv1().set_weight(conv1_weight),
             BatchedConv2D::SUCCESS);
  MUST_EQUAL(block.conv2().set_weight(conv2_weight),
             BatchedConv2D::SUCCESS);
  MUST_EQUAL(block.norm1().set_bias({1.0f}), BatchNorm2D::SUCCESS);
  MUST_EQUAL(block.norm2().set_bias({0.5f}), BatchNorm2D::SUCCESS);

  FloatTensor4D input(2, 1, 2, 2);
  input.values() = {0.2f, 0.4f, 0.7f, 0.9f,
                    0.3f, 0.6f, 0.8f, 1.1f};
  FloatTensor4D grad_output(2, 1, 2, 2);
  grad_output.values() = {0.1f, -0.2f, 0.3f, 0.4f,
                          -0.1f, 0.25f, -0.15f, 0.2f};
  FloatTensor4D output;
  MUST_EQUAL(block.Forward(input, output, true),
             ResidualBlock2D::SUCCESS);
  block.ZeroGrad();
  FloatTensor4D grad_input;
  MUST_EQUAL(block.Backward(grad_output, grad_input),
             ResidualBlock2D::SUCCESS);

  auto objective = [&](const FloatTensor4D &value) {
    FloatTensor4D prediction;
    if (block.Forward(value, prediction, true) !=
        ResidualBlock2D::SUCCESS) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    double sum = 0.0;
    for (std::size_t index = 0; index < prediction.size(); index++) {
      sum += prediction.values()[index] * grad_output.values()[index];
    }
    return sum;
  };
  const float epsilon = 1e-3f;
  const float original_weight = block.conv2().mutable_weight()[4];
  block.conv2().mutable_weight()[4] = original_weight + epsilon;
  const double plus = objective(input);
  block.conv2().mutable_weight()[4] = original_weight - epsilon;
  const double minus = objective(input);
  block.conv2().mutable_weight()[4] = original_weight;
  const double numeric_weight = (plus - minus) / (2.0 * epsilon);
  MUST_TRUE(CnnNearlyEqual(block.conv2().grad_weight()[4],
                           numeric_weight, 2e-3),
            "residual block weight gradient mismatch");

  FloatTensor4D input_plus = input;
  FloatTensor4D input_minus = input;
  input_plus(1, 0, 0, 1) += epsilon;
  input_minus(1, 0, 0, 1) -= epsilon;
  const double numeric_input =
      (objective(input_plus) - objective(input_minus)) / (2.0 * epsilon);
  MUST_TRUE(CnnNearlyEqual(grad_input(1, 0, 0, 1), numeric_input, 2e-3),
            "residual block input gradient mismatch");
}

TEST(FloatLinear, BatchForwardAndFiniteDifferenceGradient) {
  FloatLinear linear;
  FloatLinear::Config config;
  config.input_dim_ = 3;
  config.output_dim_ = 2;
  config.thread_num_ = 2;
  MUST_EQUAL(linear.Init(config), FloatLinear::SUCCESS);
  MUST_EQUAL(linear.set_weight({0.2f, -0.1f, 0.3f,
                                -0.4f, 0.5f, 0.1f}),
             FloatLinear::SUCCESS);
  MUST_EQUAL(linear.set_bias({0.05f, -0.02f}), FloatLinear::SUCCESS);
  std::vector<float> input = {0.2f, 0.4f, -0.1f,
                              0.7f, -0.3f, 0.5f};
  std::vector<float> output;
  MUST_EQUAL(linear.Forward(input, 2, output, true), FloatLinear::SUCCESS);
  MUST_TRUE(CnnNearlyEqual(output[0], 0.02, 1e-6),
            "linear batch 0 output 0 mismatch");
  MUST_TRUE(CnnNearlyEqual(output[1], 0.09, 1e-6),
            "linear batch 0 output 1 mismatch");
  MUST_TRUE(CnnNearlyEqual(output[2], 0.37, 1e-6),
            "linear batch 1 output 0 mismatch");
  MUST_TRUE(CnnNearlyEqual(output[3], -0.4, 1e-6),
            "linear batch 1 output 1 mismatch");

  std::vector<float> grad_output = {0.3f, -0.2f, 0.1f, 0.4f};
  linear.ZeroGrad();
  std::vector<float> grad_input;
  MUST_EQUAL(linear.Backward(grad_output, grad_input), FloatLinear::SUCCESS);
  auto objective = [&](const std::vector<float> &value) {
    std::vector<float> prediction;
    if (linear.Forward(value, 2, prediction) != FloatLinear::SUCCESS) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    double sum = 0.0;
    for (std::size_t index = 0; index < prediction.size(); index++) {
      sum += prediction[index] * grad_output[index];
    }
    return sum;
  };
  const float epsilon = 1e-3f;
  const float original_weight = linear.mutable_weight()[4];
  linear.mutable_weight()[4] = original_weight + epsilon;
  const double plus = objective(input);
  linear.mutable_weight()[4] = original_weight - epsilon;
  const double minus = objective(input);
  linear.mutable_weight()[4] = original_weight;
  MUST_TRUE(CnnNearlyEqual(linear.grad_weight()[4],
                           (plus - minus) / (2.0 * epsilon), 2e-4),
            "linear weight gradient mismatch");

  std::vector<float> input_plus = input;
  std::vector<float> input_minus = input;
  input_plus[2] += epsilon;
  input_minus[2] -= epsilon;
  MUST_TRUE(CnnNearlyEqual(grad_input[2],
                           (objective(input_plus) - objective(input_minus)) /
                               (2.0 * epsilon),
                           2e-4),
            "linear input gradient mismatch");
}

TEST(PolicyValueResNet, SharedHeadsForwardAndBackward) {
  PolicyValueResNet model;
  PolicyValueResNet::Config config;
  config.input_channels_ = 2;
  config.board_height_ = 3;
  config.board_width_ = 3;
  config.trunk_channels_ = 3;
  config.residual_block_num_ = 1;
  config.policy_channels_ = 2;
  config.policy_size_ = 7;
  config.value_channels_ = 1;
  config.value_hidden_dim_ = 4;
  config.thread_num_ = 2;
  config.rand_seed_ = 19;
  config.batch_norm_momentum_ = 0.0f;
  MUST_EQUAL(model.Init(config), PolicyValueResNet::SUCCESS);
  MUST_TRUE(model.parameter_count() > 0, "parameter count must be positive");

  FloatTensor4D input(2, 2, 3, 3);
  for (std::size_t index = 0; index < input.size(); index++) {
    input.values()[index] =
        static_cast<float>(0.1 + (static_cast<int>(index) % 11) * 0.04);
  }
  PolicyValueResNet::Output output;
  MUST_EQUAL(model.Forward(input, output, true),
             PolicyValueResNet::SUCCESS);
  MUST_EQUAL(output.batch_, 2);
  MUST_EQUAL(output.policy_logits_.size(), 14);
  MUST_EQUAL(output.values_.size(), 2);
  MUST_TRUE(std::isfinite(output.values_[0]), "value is not finite");
  MUST_TRUE(std::fabs(output.values_[0]) <= 1.0,
            "value must be bounded by tanh");

  std::vector<float> policy_gradient(14, 0.0f);
  std::vector<float> value_gradient(2, 0.0f);
  for (std::size_t index = 0; index < policy_gradient.size(); index++) {
    policy_gradient[index] =
        static_cast<float>((static_cast<int>(index) % 5 - 2) * 0.03);
  }
  value_gradient[0] = 0.2f;
  value_gradient[1] = -0.15f;

  // Both heads must contribute to the same stem gradient. Re-run forward
  // before each backward because the model stores one training cache.
  model.ZeroGrad();
  MUST_EQUAL(model.Forward(input, output, true),
             PolicyValueResNet::SUCCESS);
  FloatTensor4D policy_input_gradient;
  MUST_EQUAL(model.Backward(policy_gradient, {0.0f, 0.0f},
                            policy_input_gradient),
             PolicyValueResNet::SUCCESS);
  const std::vector<float> policy_stem_gradient =
      model.stem_conv().grad_weight();

  model.ZeroGrad();
  MUST_EQUAL(model.Forward(input, output, true),
             PolicyValueResNet::SUCCESS);
  FloatTensor4D value_input_gradient;
  MUST_EQUAL(model.Backward(std::vector<float>(14, 0.0f), value_gradient,
                            value_input_gradient),
             PolicyValueResNet::SUCCESS);
  const std::vector<float> value_stem_gradient =
      model.stem_conv().grad_weight();

  model.ZeroGrad();
  MUST_EQUAL(model.Forward(input, output, true),
             PolicyValueResNet::SUCCESS);
  FloatTensor4D joint_input_gradient;
  MUST_EQUAL(model.Backward(policy_gradient, value_gradient,
                            joint_input_gradient),
             PolicyValueResNet::SUCCESS);
  const std::vector<float> &joint_stem_gradient =
      model.stem_conv().grad_weight();
  double policy_norm = 0.0;
  double value_norm = 0.0;
  for (std::size_t index = 0; index < joint_stem_gradient.size(); index++) {
    policy_norm += std::fabs(policy_stem_gradient[index]);
    value_norm += std::fabs(value_stem_gradient[index]);
    MUST_TRUE(
        CnnNearlyEqual(joint_stem_gradient[index],
                       policy_stem_gradient[index] +
                           value_stem_gradient[index],
                       2e-4),
        "joint stem gradient is not the sum of both heads");
  }
  MUST_TRUE(policy_norm > 0.0, "policy head did not reach shared trunk");
  MUST_TRUE(value_norm > 0.0, "value head did not reach shared trunk");
}

TEST(PolicyValueLoss, MaskedPolicyAndValueGradients) {
  PolicyValueLoss loss;
  PolicyValueLoss::Config config;
  config.policy_size_ = 4;
  config.policy_weight_ = 1.0f;
  config.value_weight_ = 0.5f;
  MUST_EQUAL(loss.Init(config), PolicyValueLoss::SUCCESS);
  std::vector<float> logits = {0.0f, 1.0f, 10.0f, -1.0f};
  std::vector<std::uint8_t> legal = {1, 1, 0, 0};
  std::vector<float> policy_target = {0.25f, 0.75f, 0.0f, 0.0f};
  std::vector<float> values = {0.2f};
  std::vector<float> value_target = {1.0f};
  PolicyValueLoss::Result result;
  MUST_EQUAL(loss.ForwardBackward(logits, values, legal, policy_target,
                                  value_target, 1, result),
             PolicyValueLoss::SUCCESS);
  const double p0 = 1.0 / (1.0 + std::exp(1.0));
  const double p1 = 1.0 - p0;
  MUST_TRUE(CnnNearlyEqual(result.grad_policy_logits_[0], p0 - 0.25, 1e-6),
            "legal action 0 gradient mismatch");
  MUST_TRUE(CnnNearlyEqual(result.grad_policy_logits_[1], p1 - 0.75, 1e-6),
            "legal action 1 gradient mismatch");
  MUST_TRUE(CnnNearlyEqual(result.grad_policy_logits_[2], 0.0, 1e-9),
            "illegal action gradient must be zero");
  MUST_TRUE(CnnNearlyEqual(result.grad_policy_logits_[3], 0.0, 1e-9),
            "illegal action gradient must be zero");
  MUST_TRUE(CnnNearlyEqual(result.grad_policy_logits_[0] +
                               result.grad_policy_logits_[1],
                           0.0, 1e-6),
            "legal policy gradients must sum to zero");
  MUST_TRUE(CnnNearlyEqual(result.grad_values_[0], -0.8, 1e-6),
            "weighted value gradient mismatch");
}

TEST(PolicyValueLoss, UnknownValueTargetsAreMasked) {
  PolicyValueLoss loss;
  PolicyValueLoss::Config config;
  config.policy_size_ = 2;
  config.policy_weight_ = 1.0f;
  config.value_weight_ = 0.5f;
  MUST_EQUAL(loss.Init(config), PolicyValueLoss::SUCCESS);
  const std::vector<float> logits = {0.0f, 0.0f, 1.0f, -1.0f};
  const std::vector<std::uint8_t> legal = {1, 1, 1, 1};
  const std::vector<float> policy_target = {1.0f, 0.0f, 0.0f, 1.0f};
  const std::vector<float> values = {0.25f, -0.75f};
  const std::vector<float> value_target = {
      1.0f, std::numeric_limits<float>::quiet_NaN()};
  const std::vector<std::uint8_t> value_mask = {1, 0};
  PolicyValueLoss::Result result;
  MUST_EQUAL(
      loss.ForwardBackward(logits, values, legal, policy_target, value_target,
                           value_mask, 2, result),
      PolicyValueLoss::SUCCESS);
  MUST_TRUE(CnnNearlyEqual(result.value_loss_, 0.28125, 1e-6),
            "masked value loss must normalize over known targets");
  MUST_TRUE(CnnNearlyEqual(result.grad_values_[0], -0.75, 1e-6),
            "known value gradient mismatch");
  MUST_TRUE(CnnNearlyEqual(result.grad_values_[1], 0.0, 1e-9),
            "unknown value gradient must be zero");

  const std::vector<std::uint8_t> unknown_mask = {0, 0};
  MUST_EQUAL(
      loss.ForwardBackward(logits, values, legal, policy_target, value_target,
                           unknown_mask, 2, result),
      PolicyValueLoss::SUCCESS);
  MUST_TRUE(CnnNearlyEqual(result.value_loss_, 0.0, 1e-9),
            "all-unknown value loss must be zero");
  MUST_TRUE(CnnNearlyEqual(result.grad_values_[0], 0.0, 1e-9) &&
                CnnNearlyEqual(result.grad_values_[1], 0.0, 1e-9),
            "all-unknown value gradients must be zero");

  std::vector<std::uint8_t> invalid_mask = {1, 2};
  MUST_EQUAL(
      loss.ForwardBackward(logits, values, legal, policy_target, value_target,
                           invalid_mask, 2, result),
      PolicyValueLoss::INVALID_DATA);
}

TEST(PolicyValueWdlLoss, SoftTargetsAndFiniteDifferenceGradient) {
  PolicyValueWdlLoss loss;
  PolicyValueWdlLoss::Config config;
  config.policy_size_ = 2;
  config.policy_weight_ = 0.0f;
  config.value_weight_ = 0.75f;
  MUST_EQUAL(loss.Init(config), PolicyValueWdlLoss::SUCCESS);

  const std::vector<float> policy_logits = {0.0f, 0.0f};
  const std::vector<std::uint8_t> legal = {1, 1};
  const std::vector<float> policy_target = {0.5f, 0.5f};
  std::vector<float> value_probabilities = {0.6f, 0.25f, 0.15f};
  const std::vector<float> value_target = {0.7f, 0.2f, 0.1f};
  const std::vector<std::uint8_t> value_mask = {1};
  PolicyValueWdlLoss::Result result;
  MUST_EQUAL(loss.ForwardBackward(
                 policy_logits, value_probabilities, legal, policy_target,
                 value_target, value_mask, 1, result),
             PolicyValueWdlLoss::SUCCESS);
  MUST_TRUE(result.value_loss_ > 0.0f,
            "WDL cross-entropy must be positive");
  MUST_TRUE(CnnNearlyEqual(
                result.grad_value_probabilities_[0],
                -0.75 * 0.7 / 0.6, 1e-6),
            "WDL win-probability gradient mismatch");

  const float epsilon = 1e-3f;
  for (int outcome = 0; outcome < 2; outcome++) {
    std::vector<float> plus = value_probabilities;
    std::vector<float> minus = value_probabilities;
    plus[outcome] += epsilon;
    plus[2] -= epsilon;
    minus[outcome] -= epsilon;
    minus[2] += epsilon;
    PolicyValueWdlLoss::Result plus_result;
    PolicyValueWdlLoss::Result minus_result;
    MUST_EQUAL(loss.ForwardBackward(
                   policy_logits, plus, legal, policy_target, value_target,
                   value_mask, 1, plus_result),
               PolicyValueWdlLoss::SUCCESS);
    MUST_EQUAL(loss.ForwardBackward(
                   policy_logits, minus, legal, policy_target, value_target,
                   value_mask, 1, minus_result),
               PolicyValueWdlLoss::SUCCESS);
    const double numerical =
        (plus_result.value_loss_ - minus_result.value_loss_) /
        (2.0 * epsilon);
    MUST_TRUE(CnnNearlyEqual(
                  result.grad_value_probabilities_[outcome] -
                      result.grad_value_probabilities_[2],
                  numerical, 2e-4),
              "WDL finite-difference gradient mismatch");
  }

  const std::vector<std::uint8_t> unknown = {0};
  const std::vector<float> unknown_target = {
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN()};
  MUST_EQUAL(loss.ForwardBackward(
                 policy_logits, value_probabilities, legal, policy_target,
                 unknown_target, unknown, 1, result),
             PolicyValueWdlLoss::SUCCESS);
  MUST_TRUE(CnnNearlyEqual(result.value_loss_, 0.0, 1e-9),
            "unknown WDL target must be masked");
}

TEST(PolicyValueResNet, ScalarToWdlMigrationPreservesPolicyAndEv) {
  PolicyValueResNet scalar;
  PolicyValueResNet::Config config;
  config.input_channels_ = 2;
  config.board_height_ = 2;
  config.board_width_ = 3;
  config.trunk_channels_ = 4;
  config.residual_block_num_ = 1;
  config.policy_channels_ = 2;
  config.policy_size_ = 6;
  config.value_channels_ = 1;
  config.value_hidden_dim_ = 5;
  config.thread_num_ = 2;
  config.rand_seed_ = 71;
  MUST_EQUAL(scalar.Init(config), PolicyValueResNet::SUCCESS);

  FloatTensor4D input(7, 2, 2, 3);
  for (std::size_t index = 0; index < input.size(); index++) {
    input.values()[index] =
        static_cast<float>((static_cast<int>(index) % 13 - 6) * 0.11);
  }
  PolicyValueResNet::Output scalar_output;
  MUST_EQUAL(scalar.Forward(input, scalar_output, false),
             PolicyValueResNet::SUCCESS);

  PolicyValueResNet wdl;
  MUST_EQUAL(wdl.InitWdlFromScalar(scalar, -1.6f),
             PolicyValueResNet::SUCCESS);
  MUST_EQUAL(wdl.config().value_class_num_, 3);
  PolicyValueResNet::Output wdl_output;
  MUST_EQUAL(wdl.Forward(input, wdl_output, false),
             PolicyValueResNet::SUCCESS);
  MUST_EQUAL(wdl_output.value_logits_.size(), 14);
  MUST_EQUAL(wdl_output.value_probabilities_.size(), 21);
  for (std::size_t index = 0; index < scalar_output.policy_logits_.size();
       index++) {
    MUST_TRUE(scalar_output.policy_logits_[index] ==
                  wdl_output.policy_logits_[index],
              "WDL migration changed policy logits");
  }
  for (std::size_t sample = 0; sample < scalar_output.values_.size();
       sample++) {
    MUST_TRUE(CnnNearlyEqual(
                  scalar_output.values_[sample], wdl_output.values_[sample],
                  5e-7),
              "WDL migration changed scalar EV");
    const std::size_t base = sample * 3;
    MUST_TRUE(CnnNearlyEqual(
                  wdl_output.value_probabilities_[base] +
                      wdl_output.value_probabilities_[base + 1] +
                      wdl_output.value_probabilities_[base + 2],
                  1.0, 1e-6),
              "WDL probabilities do not sum to one");
  }

  const std::string file = "/tmp/policy_value_resnet_wdl_test.bin";
  MUST_EQUAL(wdl.Save(file), PolicyValueResNet::SUCCESS);
  PolicyValueResNet loaded;
  MUST_EQUAL(loaded.Load(file), PolicyValueResNet::SUCCESS);
  MUST_EQUAL(loaded.config().value_class_num_, 3);
  PolicyValueResNet::Output loaded_output;
  MUST_EQUAL(loaded.Forward(input, loaded_output, false),
             PolicyValueResNet::SUCCESS);
  MUST_TRUE(loaded_output.policy_logits_ == wdl_output.policy_logits_,
            "WDL round trip changed policy logits");
  MUST_TRUE(loaded_output.value_logits_ == wdl_output.value_logits_,
            "WDL round trip changed value logits");
  MUST_TRUE(loaded_output.values_ == wdl_output.values_,
            "WDL round trip changed scalar EV");
  std::remove(file.c_str());
}

TEST(PolicyValueResNet, WdlDirectLogitBackwardMatchesFiniteDifference) {
  PolicyValueResNet model;
  PolicyValueResNet::Config config;
  config.input_channels_ = 1;
  config.board_height_ = 1;
  config.board_width_ = 1;
  config.trunk_channels_ = 2;
  config.residual_block_num_ = 0;
  config.policy_channels_ = 1;
  config.policy_size_ = 2;
  config.value_channels_ = 1;
  config.value_hidden_dim_ = 3;
  config.value_class_num_ = 3;
  config.thread_num_ = 1;
  config.rand_seed_ = 79;
  MUST_EQUAL(model.Init(config), PolicyValueResNet::SUCCESS);
  FloatTensor4D input(1, 1, 1, 1, 0.7f);
  PolicyValueResNet::Output output;
  MUST_EQUAL(model.Forward(input, output, true),
             PolicyValueResNet::SUCCESS);
  const std::vector<float> value_gradient = {0.2f, -0.3f, 0.1f};
  model.ZeroGrad();
  MUST_EQUAL(model.BackwardWdlOutputOnly(value_gradient),
             PolicyValueResNet::SUCCESS);
  const std::vector<float> analytic = model.value_output().grad_weight();
  const float epsilon = 1e-3f;
  for (std::size_t index = 0; index < analytic.size(); index++) {
    const float original = model.value_output().mutable_weight()[index];
    model.value_output().mutable_weight()[index] = original + epsilon;
    PolicyValueResNet::Output plus;
    MUST_EQUAL(model.Forward(input, plus, false),
               PolicyValueResNet::SUCCESS);
    model.value_output().mutable_weight()[index] = original - epsilon;
    PolicyValueResNet::Output minus;
    MUST_EQUAL(model.Forward(input, minus, false),
               PolicyValueResNet::SUCCESS);
    model.value_output().mutable_weight()[index] = original;
    double plus_objective = 0.0;
    double minus_objective = 0.0;
    for (int outcome = 0; outcome < 3; outcome++) {
      plus_objective += plus.value_probabilities_[outcome] *
                        value_gradient[outcome];
      minus_objective += minus.value_probabilities_[outcome] *
                         value_gradient[outcome];
    }
    MUST_TRUE(CnnNearlyEqual(
                  analytic[index],
                  (plus_objective - minus_objective) / (2.0 * epsilon),
                  2e-4),
              "WDL direct logit backward mismatch");
  }
}

TEST(FloatAdamW, TwoStepUpdateAndDecayMask) {
  FloatAdamW optimizer;
  FloatAdamW::Config config;
  config.beta1_ = 0.9f;
  config.beta2_ = 0.999f;
  config.epsilon_ = 1e-8f;
  config.weight_decay_ = 0.1f;
  MUST_EQUAL(optimizer.Init({2, 1}, config), FloatAdamW::SUCCESS);
  std::vector<float> weight = {1.0f, -2.0f};
  std::vector<float> weight_gradient = {0.5f, -0.25f};
  std::vector<float> bias = {1.0f};
  std::vector<float> bias_gradient = {0.5f};
  std::vector<FloatAdamW::Parameter> parameters = {
      {&weight, &weight_gradient, true},
      {&bias, &bias_gradient, false},
  };
  MUST_EQUAL(optimizer.Step(parameters, 0.01f), FloatAdamW::SUCCESS);
  MUST_EQUAL(optimizer.step(), 1);
  MUST_TRUE(CnnNearlyEqual(weight[0], 0.989, 1e-5),
            "weight AdamW update mismatch");
  MUST_TRUE(CnnNearlyEqual(weight[1], -1.988, 1e-5),
            "negative weight AdamW update mismatch");
  MUST_TRUE(CnnNearlyEqual(bias[0], 0.99, 1e-5),
            "bias must not receive weight decay");
  MUST_EQUAL(optimizer.Step(parameters, 0.01f), FloatAdamW::SUCCESS);
  MUST_EQUAL(optimizer.step(), 2);
  MUST_TRUE(weight[0] < 0.989f && bias[0] < 0.99f,
            "second optimizer step did not update parameters");
}

TEST(FloatSGD, ScaleDecayMaskAndCheckpointRoundTrip) {
  FloatSGD optimizer;
  FloatSGD::Config config;
  config.weight_decay_ = 0.1f;
  MUST_EQUAL(optimizer.Init({2, 1}, config), FloatSGD::SUCCESS);
  std::vector<float> weight = {1.0f, -2.0f};
  std::vector<float> weight_gradient = {0.5f, -0.25f};
  std::vector<float> bias = {1.0f};
  std::vector<float> bias_gradient = {0.5f};
  std::vector<FloatSGD::Parameter> parameters = {
      {&weight, &weight_gradient, true},
      {&bias, &bias_gradient, false},
  };
  MUST_EQUAL(optimizer.Step(parameters, 0.01f, 2.0f),
             FloatSGD::SUCCESS);
  MUST_EQUAL(optimizer.step(), 1);
  MUST_TRUE(CnnNearlyEqual(weight[0], 0.989, 1e-7),
            "scaled SGD weight update mismatch");
  MUST_TRUE(CnnNearlyEqual(weight[1], -1.993, 1e-7),
            "scaled SGD negative weight update mismatch");
  MUST_TRUE(CnnNearlyEqual(bias[0], 0.99, 1e-7),
            "SGD bias must not receive weight decay");

  const std::string file = "/tmp/float_sgd_test.bin";
  MUST_EQUAL(optimizer.Save(file), FloatSGD::SUCCESS);
  FloatSGD loaded;
  MUST_EQUAL(loaded.Load(file), FloatSGD::SUCCESS);
  MUST_EQUAL(loaded.step(), 1);
  MUST_TRUE(CnnNearlyEqual(
                loaded.config().weight_decay_, config.weight_decay_, 1e-9),
            "SGD checkpoint config mismatch");
  MUST_TRUE(loaded.parameter_sizes() == std::vector<std::size_t>({2, 1}),
            "SGD checkpoint tensor sizes mismatch");

  std::vector<float> loaded_weight = weight;
  std::vector<float> loaded_bias = bias;
  std::vector<FloatSGD::Parameter> loaded_parameters = {
      {&loaded_weight, &weight_gradient, true},
      {&loaded_bias, &bias_gradient, false},
  };
  MUST_EQUAL(optimizer.Step(parameters, 0.005f, 0.5f),
             FloatSGD::SUCCESS);
  MUST_EQUAL(loaded.Step(loaded_parameters, 0.005f, 0.5f),
             FloatSGD::SUCCESS);
  MUST_TRUE(weight == loaded_weight && bias == loaded_bias,
            "SGD checkpoint continuation mismatch");
  std::remove(file.c_str());
}

TEST(FloatSGD, MaxUpdateNormScale) {
  MUST_TRUE(CnnNearlyEqual(
                FloatSGD::GradientScaleForMaxUpdateNorm(
                    2.0, 0.1f, 0.5f),
                1.0, 1e-9),
            "SGD update below cap must not scale");
  const float scale = FloatSGD::GradientScaleForMaxUpdateNorm(
      10.0, 0.1f, 0.25f);
  MUST_TRUE(CnnNearlyEqual(scale, 0.25, 1e-7),
            "SGD update cap scale mismatch");
  MUST_TRUE(CnnNearlyEqual(0.1 * 10.0 * scale, 0.25, 1e-7),
            "SGD capped update norm mismatch");
  MUST_TRUE(CnnNearlyEqual(
                FloatSGD::GradientScaleForMaxUpdateNorm(
                    10.0, 0.1f, 0.0f),
                1.0, 1e-9),
            "disabled SGD cap must not scale");
  MUST_TRUE(CnnNearlyEqual(
                FloatSGD::GradientScaleForMaxUpdateNorm(
                    0.0, 0.1f, 0.25f),
                1.0, 1e-9),
            "zero SGD gradient must not scale");
  MUST_TRUE(std::isnan(
                FloatSGD::GradientScaleForMaxUpdateNorm(
                    -1.0, 0.1f, 0.25f)),
            "invalid SGD gradient norm must be rejected");
}

TEST(PolicyValueResNet, JointTrainingFitsToyBatch) {
  PolicyValueResNet model;
  PolicyValueResNet::Config config;
  config.input_channels_ = 2;
  config.board_height_ = 3;
  config.board_width_ = 3;
  config.trunk_channels_ = 6;
  config.residual_block_num_ = 1;
  config.policy_channels_ = 2;
  config.policy_size_ = 5;
  config.value_channels_ = 1;
  config.value_hidden_dim_ = 8;
  config.thread_num_ = 4;
  config.rand_seed_ = 31;
  config.batch_norm_momentum_ = 0.1f;
  MUST_EQUAL(model.Init(config), PolicyValueResNet::SUCCESS);

  FloatTensor4D input(4, 2, 3, 3, 0.0f);
  for (int sample = 0; sample < 4; sample++) {
    input(sample, 0, sample / 2, sample % 2) = 1.0f;
    input(sample, 1, 2 - sample / 2, 2 - sample % 2) = 1.0f;
  }
  std::vector<std::uint8_t> legal(4 * 5, 1);
  std::vector<float> policy_target(4 * 5, 0.0f);
  std::vector<int> target_action = {0, 1, 3, 4};
  for (int sample = 0; sample < 4; sample++) {
    policy_target[sample * 5 + target_action[sample]] = 1.0f;
  }
  std::vector<float> value_target = {1.0f, -1.0f, 1.0f, -1.0f};
  PolicyValueLoss loss;
  PolicyValueLoss::Config loss_config;
  loss_config.policy_size_ = 5;
  loss_config.policy_weight_ = 1.0f;
  loss_config.value_weight_ = 1.0f;
  MUST_EQUAL(loss.Init(loss_config), PolicyValueLoss::SUCCESS);

  auto trainable = model.TrainableParameters();
  std::vector<std::size_t> sizes;
  std::vector<FloatAdamW::Parameter> optimizer_parameters;
  for (const auto &parameter : trainable) {
    sizes.push_back(parameter.value_->size());
    optimizer_parameters.push_back({
        parameter.value_,
        parameter.gradient_,
        parameter.apply_weight_decay_,
    });
  }
  FloatAdamW optimizer;
  FloatAdamW::Config optimizer_config;
  optimizer_config.weight_decay_ = 1e-5f;
  MUST_EQUAL(optimizer.Init(sizes, optimizer_config), FloatAdamW::SUCCESS);

  float first_loss = 0.0f;
  float final_loss = 0.0f;
  for (int step = 0; step < 250; step++) {
    PolicyValueResNet::Output output;
    MUST_EQUAL(model.Forward(input, output, true),
               PolicyValueResNet::SUCCESS);
    PolicyValueLoss::Result result;
    MUST_EQUAL(loss.ForwardBackward(
                   output.policy_logits_, output.values_, legal,
                   policy_target, value_target, 4, result),
               PolicyValueLoss::SUCCESS);
    if (step == 0) {
      first_loss = result.total_loss_;
    }
    final_loss = result.total_loss_;
    model.ZeroGrad();
    FloatTensor4D grad_input;
    MUST_EQUAL(model.Backward(result.grad_policy_logits_, result.grad_values_,
                              grad_input),
               PolicyValueResNet::SUCCESS);
    MUST_EQUAL(optimizer.Step(optimizer_parameters, 0.003f),
               FloatAdamW::SUCCESS);
  }
  DEBUG("toy shared resnet loss " << first_loss << " -> " << final_loss);
  MUST_TRUE(final_loss < first_loss * 0.25f,
            "joint model did not reduce toy loss enough");

  PolicyValueResNet::Output prediction;
  MUST_EQUAL(model.Forward(input, prediction, false),
             PolicyValueResNet::SUCCESS);
  int correct_policy = 0;
  int correct_value = 0;
  for (int sample = 0; sample < 4; sample++) {
    int best = 0;
    for (int action = 1; action < 5; action++) {
      if (prediction.policy_logits_[sample * 5 + action] >
          prediction.policy_logits_[sample * 5 + best]) {
        best = action;
      }
    }
    if (best == target_action[sample]) {
      correct_policy++;
    }
    if (prediction.values_[sample] * value_target[sample] > 0.0f) {
      correct_value++;
    }
  }
  MUST_EQUAL(correct_policy, 4);
  MUST_EQUAL(correct_value, 4);
}

TEST(PolicyValueResNet, ModelAndOptimizerCheckpointRoundTrip) {
  PolicyValueResNet model;
  PolicyValueResNet::Config config;
  config.input_channels_ = 2;
  config.board_height_ = 2;
  config.board_width_ = 3;
  config.trunk_channels_ = 4;
  config.residual_block_num_ = 1;
  config.policy_channels_ = 2;
  config.policy_size_ = 6;
  config.value_channels_ = 1;
  config.value_hidden_dim_ = 5;
  config.thread_num_ = 2;
  config.rand_seed_ = 43;
  MUST_EQUAL(model.Init(config), PolicyValueResNet::SUCCESS);
  FloatTensor4D input(2, 2, 2, 3);
  for (std::size_t index = 0; index < input.size(); index++) {
    input.values()[index] =
        static_cast<float>(0.05 + (static_cast<int>(index) % 9) * 0.06);
  }

  auto trainable = model.TrainableParameters();
  std::vector<std::size_t> sizes;
  std::vector<FloatAdamW::Parameter> parameters;
  for (const auto &parameter : trainable) {
    sizes.push_back(parameter.value_->size());
    parameters.push_back({
        parameter.value_, parameter.gradient_, parameter.apply_weight_decay_});
  }
  FloatAdamW optimizer;
  MUST_EQUAL(optimizer.Init(sizes), FloatAdamW::SUCCESS);
  PolicyValueResNet::Output training_output;
  MUST_EQUAL(model.Forward(input, training_output, true),
             PolicyValueResNet::SUCCESS);
  std::vector<float> policy_gradient(training_output.policy_logits_.size(),
                                     0.01f);
  std::vector<float> value_gradient = {0.2f, -0.1f};
  model.ZeroGrad();
  FloatTensor4D grad_input;
  MUST_EQUAL(model.Backward(policy_gradient, value_gradient, grad_input),
             PolicyValueResNet::SUCCESS);
  MUST_EQUAL(optimizer.Step(parameters, 0.001f), FloatAdamW::SUCCESS);

  const std::string model_file = "/tmp/policy_value_resnet_test.bin";
  const std::string optimizer_file = "/tmp/float_adamw_test.bin";
  MUST_EQUAL(model.Save(model_file), PolicyValueResNet::SUCCESS);
  MUST_EQUAL(optimizer.Save(optimizer_file), FloatAdamW::SUCCESS);
  PolicyValueResNet::Output expected;
  MUST_EQUAL(model.Forward(input, expected, false),
             PolicyValueResNet::SUCCESS);

  PolicyValueResNet loaded_model;
  MUST_EQUAL(loaded_model.Load(model_file), PolicyValueResNet::SUCCESS);
  PolicyValueResNet::Output actual;
  MUST_EQUAL(loaded_model.Forward(input, actual, false),
             PolicyValueResNet::SUCCESS);
  MUST_EQUAL(actual.policy_logits_.size(), expected.policy_logits_.size());
  for (std::size_t index = 0; index < actual.policy_logits_.size(); index++) {
    MUST_TRUE(CnnNearlyEqual(actual.policy_logits_[index],
                             expected.policy_logits_[index], 1e-6),
              "loaded policy output mismatch");
  }
  for (std::size_t index = 0; index < actual.values_.size(); index++) {
    MUST_TRUE(CnnNearlyEqual(actual.values_[index], expected.values_[index],
                             1e-6),
              "loaded value output mismatch");
  }

  auto loaded_trainable = loaded_model.TrainableParameters();
  std::vector<FloatAdamW::Parameter> loaded_parameters;
  for (const auto &parameter : loaded_trainable) {
    loaded_parameters.push_back({
        parameter.value_, parameter.gradient_, parameter.apply_weight_decay_});
  }
  FloatAdamW loaded_optimizer;
  MUST_EQUAL(loaded_optimizer.Load(optimizer_file), FloatAdamW::SUCCESS);
  MUST_EQUAL(loaded_optimizer.step(), optimizer.step());

  // Apply the same synthetic gradient to both uninterrupted and restored
  // paths. Their second AdamW update must remain identical.
  for (std::size_t tensor = 0; tensor < parameters.size(); tensor++) {
    std::vector<float> &original_gradient =
        const_cast<std::vector<float> &>(*parameters[tensor].gradient_);
    std::vector<float> &loaded_gradient =
        const_cast<std::vector<float> &>(*loaded_parameters[tensor].gradient_);
    for (std::size_t index = 0; index < original_gradient.size(); index++) {
      const float gradient =
          static_cast<float>((static_cast<int>(index + tensor) % 7 - 3) *
                             0.001);
      original_gradient[index] = gradient;
      loaded_gradient[index] = gradient;
    }
  }
  MUST_EQUAL(optimizer.Step(parameters, 0.001f), FloatAdamW::SUCCESS);
  MUST_EQUAL(loaded_optimizer.Step(loaded_parameters, 0.001f),
             FloatAdamW::SUCCESS);
  for (std::size_t tensor = 0; tensor < parameters.size(); tensor++) {
    for (std::size_t index = 0; index < parameters[tensor].value_->size();
         index++) {
      MUST_TRUE(
          CnnNearlyEqual((*parameters[tensor].value_)[index],
                         (*loaded_parameters[tensor].value_)[index], 1e-7),
          "optimizer checkpoint continuation mismatch");
    }
  }
  std::remove(model_file.c_str());
  std::remove(optimizer_file.c_str());
}

TEST(MaxPool2D, ForwardAndBackward) {
  MaxPool2D pool;
  MUST_EQUAL(pool.Init(2, 2, 2), MaxPool2D::SUCCESS);

  MaxPool2D::Tensor3D input = {{{1.0, 5.0, 2.0, 4.0},
                                {3.0, 2.0, 7.0, 1.0},
                                {0.0, 6.0, 8.0, 2.0},
                                {9.0, 1.0, 3.0, 4.0}}};
  MaxPool2D::Tensor3D output;
  MUST_EQUAL(pool.Forward(input, output), MaxPool2D::SUCCESS);
  MUST_TRUE(CnnNearlyEqual(output[0][0][0], 5.0), "pooled(0,0) mismatch");
  MUST_TRUE(CnnNearlyEqual(output[0][0][1], 7.0), "pooled(0,1) mismatch");
  MUST_TRUE(CnnNearlyEqual(output[0][1][0], 9.0), "pooled(1,0) mismatch");
  MUST_TRUE(CnnNearlyEqual(output[0][1][1], 8.0), "pooled(1,1) mismatch");

  MaxPool2D::Tensor3D grad_output = {{{1.0, 2.0}, {3.0, 4.0}}};
  MaxPool2D::Tensor3D grad_input;
  MUST_EQUAL(pool.Backward(grad_output, grad_input), MaxPool2D::SUCCESS);
  MUST_TRUE(CnnNearlyEqual(grad_input[0][0][1], 1.0), "grad for max=5 mismatch");
  MUST_TRUE(CnnNearlyEqual(grad_input[0][1][2], 2.0), "grad for max=7 mismatch");
  MUST_TRUE(CnnNearlyEqual(grad_input[0][3][0], 3.0), "grad for max=9 mismatch");
  MUST_TRUE(CnnNearlyEqual(grad_input[0][2][2], 4.0), "grad for max=8 mismatch");
}

TEST(MiniCNNClassifier, TrainToyImageClassifier) {
  std::vector<Conv2D::Tensor3D> images;
  std::vector<int> labels;
  MakeCNNToyData(images, labels);

  MiniCNNClassifier model;
  auto config = MakeCNNToyConfig();
  MUST_EQUAL(model.Init(config), MiniCNNClassifier::SUCCESS);

  auto rc = model.Train(images, labels, nullptr, 160, 0.03);
  MUST_TRUE(rc == MiniCNNClassifier::SUCCESS, model.err_msg());

  double acc = CNNAccuracy(model, images, labels);
  DEBUG("mini cnn acc=" << acc);
  MUST_TRUE(acc > 0.99, "mini cnn should fit toy images");
}

TEST(MiniCNNClassifier, ForwardInvalidInputKeepsLegacyMessage) {
  MiniCNNClassifier model;
  MUST_EQUAL(model.Init(MakeCNNToyConfig()), MiniCNNClassifier::SUCCESS);

  Conv2D::Tensor3D invalid_image = MakeImage({"010", "010", "010"});
  std::vector<double> logits;
  MUST_EQUAL(model.Forward(invalid_image, logits), MiniCNNClassifier::INVALID_DATA);
  MUST_TRUE(model.err_msg() ==
                "[MiniCNNClassifier::ForwardFeature] Invalid input",
            "invalid input should keep old ForwardFeature message");
}

TEST(MiniCNNClassifier, TrainThreadNumConfig) {
  MiniCNNClassifier model;
  MUST_EQUAL(model.train_thread_num(), 1);
  model.set_train_thread_num(0);
  MUST_EQUAL(model.train_thread_num(), 1);
  model.set_train_thread_num(4);
  MUST_EQUAL(model.train_thread_num(), 4);
}

TEST(MiniCNNClassifier, TrainBatchParallelMatchesSingleThreadOneStep) {
  std::vector<Conv2D::Tensor3D> images;
  std::vector<int> labels;
  MakeCNNToyData(images, labels);
  auto config = MakeCNNToyConfig();

  MiniCNNClassifier serial_model;
  MiniCNNClassifier parallel_model;
  MUST_EQUAL(serial_model.Init(config), MiniCNNClassifier::SUCCESS);
  MUST_EQUAL(parallel_model.Init(config), MiniCNNClassifier::SUCCESS);
  parallel_model.set_train_thread_num(4);

  auto serial_rc = serial_model.TrainBatch(images, labels, 4, nullptr, 1, 0.03);
  auto parallel_rc =
      parallel_model.TrainBatch(images, labels, 4, nullptr, 1, 0.03);
  MUST_TRUE(serial_rc == MiniCNNClassifier::SUCCESS, serial_model.err_msg());
  MUST_TRUE(parallel_rc == MiniCNNClassifier::SUCCESS,
            parallel_model.err_msg());
  MUST_TRUE(SameCNNParams(serial_model, parallel_model, 1e-12),
            "parallel CNN batch params should match single-thread");
}

TEST(MiniCNNClassifier, TrainBatchParallelToyImageClassifier) {
  std::vector<Conv2D::Tensor3D> images;
  std::vector<int> labels;
  MakeCNNToyData(images, labels);

  MiniCNNClassifier model;
  MUST_EQUAL(model.Init(MakeCNNToyConfig()), MiniCNNClassifier::SUCCESS);
  model.set_train_thread_num(4);

  auto rc = model.TrainBatch(images, labels, 4, nullptr, 240, 0.08);
  MUST_TRUE(rc == MiniCNNClassifier::SUCCESS, model.err_msg());

  double acc = CNNAccuracy(model, images, labels);
  DEBUG("mini cnn parallel batch acc=" << acc);
  MUST_TRUE(acc > 0.99, "parallel mini cnn should fit toy images");
}
