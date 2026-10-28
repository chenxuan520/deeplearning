#pragma once

#include "cnn/batched_conv2d.h"

#include <string>

namespace deeplearning {

// A zero-output bottleneck residual branch. The up projection starts at zero,
// so the first backward reaches it and subsequent backwards reach both layers.
class SingleResidualAdapter2D {
public:
  struct Config {
    int channels_ = 0;
    int bottleneck_channels_ = 0;
    int height_ = 0;
    int width_ = 0;
    int thread_num_ = 1;
    int rand_seed_ = 0;
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
             bool training);
  RC Backward(const FloatTensor4D &grad_output, FloatTensor4D &grad_input);
  void ZeroGrad();

  BatchedConv2D &down() { return down_; }
  BatchedConv2D &up() { return up_; }
  const BatchedConv2D &down() const { return down_; }
  const BatchedConv2D &up() const { return up_; }
  const Config &config() const { return config_; }
  std::string err_msg() const { return err_msg_; }

private:
  bool ValidateInput(const FloatTensor4D &input, const char *function);
  RC ForwardInference(const FloatTensor4D &input, FloatTensor4D &output);

  Config config_;
  BatchedConv2D down_;
  BatchedConv2D up_;
  FloatTensor4D relu_cache_;
  bool is_init_ = false;
  bool has_cache_ = false;
  std::string err_msg_;
};

} // namespace deeplearning
