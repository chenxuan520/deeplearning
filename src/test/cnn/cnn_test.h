#pragma once

#include "cnn/conv2d.h"
#include "cnn/max_pool2d.h"
#include "cnn/mini_cnn_classifier.h"
#include "test.h"

#include <cmath>
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
  std::vector<Conv2D::Tensor3D> images = {
      MakeImage({"0100", "0100", "0100", "0100"}),
      MakeImage({"0010", "0010", "0010", "0010"}),
      MakeImage({"0100", "0100", "0110", "0100"}),
      MakeImage({"0010", "0010", "0011", "0010"}),
      MakeImage({"0000", "1111", "0000", "0000"}),
      MakeImage({"0000", "0000", "1111", "0000"}),
      MakeImage({"0000", "1111", "0010", "0000"}),
      MakeImage({"0000", "0010", "1111", "0000"}),
  };
  std::vector<int> labels = {0, 0, 0, 0, 1, 1, 1, 1};

  MiniCNNClassifier model;
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
  MUST_EQUAL(model.Init(config), MiniCNNClassifier::SUCCESS);

  auto rc = model.Train(images, labels, nullptr, 160, 0.03);
  MUST_TRUE(rc == MiniCNNClassifier::SUCCESS, model.err_msg());

  double acc = CNNAccuracy(model, images, labels);
  DEBUG("mini cnn acc=" << acc);
  MUST_TRUE(acc > 0.99, "mini cnn should fit toy images");
}
