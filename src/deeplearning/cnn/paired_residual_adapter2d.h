#pragma once

#include "cnn/batched_conv2d.h"

#include <string>

namespace deeplearning {

// Two identical bottleneck branches cancel exactly at initialization while
// keeping every branch weight reachable by the first backward pass.
class PairedResidualAdapter2D {
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

  BatchedConv2D &branch_a_down() { return branch_a_down_; }
  BatchedConv2D &branch_a_up() { return branch_a_up_; }
  BatchedConv2D &branch_b_down() { return branch_b_down_; }
  BatchedConv2D &branch_b_up() { return branch_b_up_; }
  const BatchedConv2D &branch_a_down() const { return branch_a_down_; }
  const BatchedConv2D &branch_a_up() const { return branch_a_up_; }
  const BatchedConv2D &branch_b_down() const { return branch_b_down_; }
  const BatchedConv2D &branch_b_up() const { return branch_b_up_; }
  const Config &config() const { return config_; }
  std::string err_msg() const { return err_msg_; }

private:
  bool ValidateInput(const FloatTensor4D &input, const char *function);
  RC ForwardInference(const FloatTensor4D &input, FloatTensor4D &output);

  Config config_;
  BatchedConv2D branch_a_down_;
  BatchedConv2D branch_a_up_;
  BatchedConv2D branch_b_down_;
  BatchedConv2D branch_b_up_;
  FloatTensor4D branch_a_relu_cache_;
  FloatTensor4D branch_b_relu_cache_;
  bool is_init_ = false;
  bool has_cache_ = false;
  std::string err_msg_;
};

} // namespace deeplearning
