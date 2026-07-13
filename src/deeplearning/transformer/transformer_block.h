#pragma once

#include "layer_norm.h"
#include "self_attention.h"
#include "tensor_optimizer.h"

#include <random>
#include <string>
#include <vector>

namespace deeplearning {

class TransformerBlock {
public:
  using Matrix = std::vector<std::vector<double>>;

  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
  };

public:
  RC Init(int model_dim, int head_num, int feed_forward_dim);
  RC Forward(const Matrix &input, Matrix &output,
             const Matrix *mask = nullptr);
  RC Backward(const Matrix &grad_output, Matrix &grad_input,
              double learning_rate);
  RC BackwardAccumulate(const Matrix &grad_output, Matrix &grad_input);
  void ApplyGradient(double learning_rate, double gradient_scale = 1.0);
  void ClearGradients();

  void set_random_seed(int seed);
  RC set_feed_forward_weight_1(const Matrix &weight);
  RC set_feed_forward_bias_1(const std::vector<double> &bias);
  RC set_feed_forward_weight_2(const Matrix &weight);
  RC set_feed_forward_bias_2(const std::vector<double> &bias);

  SelfAttention &self_attention();
  const SelfAttention &self_attention() const;
  LayerNorm &attention_norm();
  const LayerNorm &attention_norm() const;
  LayerNorm &feed_forward_norm();
  const LayerNorm &feed_forward_norm() const;
  const Matrix &feed_forward_weight_1() const;
  const std::vector<double> &feed_forward_bias_1() const;
  const Matrix &feed_forward_weight_2() const;
  const std::vector<double> &feed_forward_bias_2() const;
  std::string err_msg();

private:
  RC ValidateInput(const Matrix &input);
  RC ValidateWeightShape(const Matrix &weight, int row, int col,
                         const char *func_name);
  void InitWeight(Matrix &weight, std::mt19937 &gen);

private:
  int model_dim_ = 0;
  int head_num_ = 0;
  int feed_forward_dim_ = 0;
  int rand_seed_ = 0;
  Matrix feed_forward_weight_1_;
  std::vector<double> feed_forward_bias_1_;
  Matrix feed_forward_weight_2_;
  std::vector<double> feed_forward_bias_2_;
  Matrix last_input_;
  Matrix last_attention_output_;
  Matrix last_residual_1_;
  Matrix last_norm_1_;
  Matrix last_feed_forward_hidden_linear_;
  Matrix last_feed_forward_hidden_;
  Matrix last_feed_forward_output_;
  Matrix last_residual_2_;
  Matrix grad_feed_forward_weight_1_;
  std::vector<double> grad_feed_forward_bias_1_;
  Matrix grad_feed_forward_weight_2_;
  std::vector<double> grad_feed_forward_bias_2_;
  SelfAttention self_attention_;
  LayerNorm attention_norm_;
  LayerNorm feed_forward_norm_;
  TensorOptimizer feed_forward_weight_1_optimizer_;
  TensorOptimizer feed_forward_bias_1_optimizer_;
  TensorOptimizer feed_forward_weight_2_optimizer_;
  TensorOptimizer feed_forward_bias_2_optimizer_;
  std::string err_msg_;
  bool is_init_ = false;
};

} // namespace deeplearning
