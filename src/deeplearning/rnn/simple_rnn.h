#pragma once

#include "transformer/tensor_optimizer.h"

#include <string>
#include <vector>

namespace deeplearning {

class SimpleRNN {
public:
  using Matrix = std::vector<std::vector<double>>;

  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
  };

public:
  RC Init(int input_dim, int hidden_dim);
  RC Forward(const Matrix &input_sequence, Matrix &hidden_sequence);
  RC Backward(const Matrix &grad_hidden_sequence, Matrix &grad_input_sequence);
  void ApplyGradient(double learning_rate);

  void set_random_seed(int seed);
  RC set_input_weight(const Matrix &weight);
  RC set_hidden_weight(const Matrix &weight);
  RC set_bias(const std::vector<double> &bias);

  std::string err_msg();
  int input_dim() const;
  int hidden_dim() const;
  const Matrix &input_weight() const;
  const Matrix &hidden_weight() const;
  const std::vector<double> &bias() const;
  const Matrix &grad_input_weight() const;
  const Matrix &grad_hidden_weight() const;
  const std::vector<double> &grad_bias() const;
  double GradSquaredNorm() const;
  void ScaleGradients(double scale);

private:
  void ResetGradients();

private:
  int input_dim_ = 0;
  int hidden_dim_ = 0;
  int rand_seed_ = 0;
  Matrix input_weight_;
  Matrix hidden_weight_;
  std::vector<double> bias_;
  Matrix grad_input_weight_;
  Matrix grad_hidden_weight_;
  std::vector<double> grad_bias_;
  Matrix last_input_sequence_;
  Matrix last_hidden_sequence_;
  TensorOptimizer input_weight_optimizer_;
  TensorOptimizer hidden_weight_optimizer_;
  TensorOptimizer bias_optimizer_;
  std::string err_msg_;
  bool is_init_ = false;
  bool has_forward_cache_ = false;
};

} // namespace deeplearning
