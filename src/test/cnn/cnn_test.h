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
