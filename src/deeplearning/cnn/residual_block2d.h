#pragma once

#include "cnn/batch_norm2d.h"
#include "cnn/batched_conv2d.h"

#include <string>

namespace deeplearning {

class ResidualBlock2D {
public:
  struct Config {
    int channels_ = 0;
    int height_ = 0;
    int width_ = 0;
    int thread_num_ = 1;
    int rand_seed_ = 0;
    float batch_norm_epsilon_ = 1e-5f;
    float batch_norm_momentum_ = 0.1f;
  };

  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
    MISSING_CACHE,
  };

  RC Init(const Config &config);
  RC Forward(const FloatTensor4D &input, FloatTensor4D &output,
             bool training, bool use_running_statistics = false);
  RC Backward(const FloatTensor4D &grad_output, FloatTensor4D &grad_input);
  void ZeroGrad();

  BatchedConv2D &conv1() { return conv1_; }
  BatchedConv2D &conv2() { return conv2_; }
  BatchNorm2D &norm1() { return norm1_; }
  BatchNorm2D &norm2() { return norm2_; }
  const BatchedConv2D &conv1() const { return conv1_; }
  const BatchedConv2D &conv2() const { return conv2_; }
  const BatchNorm2D &norm1() const { return norm1_; }
  const BatchNorm2D &norm2() const { return norm2_; }
  const Config &config() const { return config_; }
  std::string err_msg() const { return err_msg_; }

private:
  bool ValidateInput(const FloatTensor4D &input, const char *function);

  Config config_;
  BatchedConv2D conv1_;
  BatchedConv2D conv2_;
  BatchNorm2D norm1_;
  BatchNorm2D norm2_;
  FloatTensor4D relu1_cache_;
  FloatTensor4D pre_relu2_cache_;
  bool is_init_ = false;
  bool has_cache_ = false;
  std::string err_msg_;
};

} // namespace deeplearning
