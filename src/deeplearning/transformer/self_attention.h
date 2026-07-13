#pragma once

#include "tensor_optimizer.h"

#include <random>
#include <string>
#include <vector>

namespace deeplearning {

class SelfAttention {
public:
  using Matrix = std::vector<std::vector<double>>;
  using Tensor3D = std::vector<Matrix>;

  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
  };

public:
  RC Init(int model_dim, int head_num = 1);
  RC Forward(const Matrix &input, Matrix &output,
             const Matrix *mask = nullptr);
  RC Backward(const Matrix &grad_output, Matrix &grad_input,
              double learning_rate);
  RC BackwardAccumulate(const Matrix &grad_output, Matrix &grad_input);
  void ApplyGradient(double learning_rate, double gradient_scale = 1.0);
  void ClearGradients();

  void set_random_seed(int seed);
  RC set_query_weight(const Matrix &weight);
  RC set_key_weight(const Matrix &weight);
  RC set_value_weight(const Matrix &weight);
  RC set_output_weight(const Matrix &weight);

  std::string err_msg();
  const Matrix &query_weight() const;
  const Matrix &key_weight() const;
  const Matrix &value_weight() const;
  const Matrix &output_weight() const;
  const Tensor3D &last_attention_weight() const;

private:
  RC ValidateWeight(const Matrix &weight, const char *func_name);
  void InitProjectionWeight(Matrix &weight, std::mt19937 &gen);

private:
  int model_dim_ = 0;
  int head_num_ = 0;
  int head_dim_ = 0;
  int rand_seed_ = 0;
  Matrix query_weight_;
  Matrix key_weight_;
  Matrix value_weight_;
  Matrix output_weight_;
  Matrix last_input_;
  Matrix last_query_;
  Matrix last_key_;
  Matrix last_value_;
  Matrix last_merged_output_;
  Matrix last_mask_;
  bool has_last_mask_ = false;
  Tensor3D last_attention_weight_;
  Matrix grad_query_weight_;
  Matrix grad_key_weight_;
  Matrix grad_value_weight_;
  Matrix grad_output_weight_;
  TensorOptimizer query_optimizer_;
  TensorOptimizer key_optimizer_;
  TensorOptimizer value_optimizer_;
  TensorOptimizer output_optimizer_;
  std::string err_msg_;
  bool is_init_ = false;
};

} // namespace deeplearning
