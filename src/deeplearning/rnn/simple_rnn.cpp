#include "rnn/simple_rnn.h"

#include <cmath>
#include <random>

namespace deeplearning {

SimpleRNN::RC SimpleRNN::Init(int input_dim, int hidden_dim) {
  if (is_init_) {
    err_msg_ = "[SimpleRNN::Init] SimpleRNN has init";
    return ALREADY_INIT;
  }
  if (input_dim <= 0 || hidden_dim <= 0) {
    err_msg_ = "[SimpleRNN::Init] Invalid config";
    return INVALID_DATA;
  }

  input_dim_ = input_dim;
  hidden_dim_ = hidden_dim;
  input_weight_.assign(hidden_dim_, std::vector<double>(input_dim_, 0.0));
  hidden_weight_.assign(hidden_dim_, std::vector<double>(hidden_dim_, 0.0));
  bias_.assign(hidden_dim_, 0.0);
  grad_input_weight_.assign(hidden_dim_, std::vector<double>(input_dim_, 0.0));
  grad_hidden_weight_.assign(hidden_dim_, std::vector<double>(hidden_dim_, 0.0));
  grad_bias_.assign(hidden_dim_, 0.0);

  std::mt19937 gen(static_cast<std::mt19937::result_type>(rand_seed_));
  const double input_limit =
      std::sqrt(6.0 / static_cast<double>(input_dim_ + hidden_dim_));
  const double hidden_limit =
      std::sqrt(6.0 / static_cast<double>(hidden_dim_ + hidden_dim_));
  std::uniform_real_distribution<double> input_dist(-input_limit, input_limit);
  std::uniform_real_distribution<double> hidden_dist(-hidden_limit, hidden_limit);
  for (auto &row : input_weight_) {
    for (double &value : row) {
      value = input_dist(gen);
    }
  }
  for (auto &row : hidden_weight_) {
    for (double &value : row) {
      value = hidden_dist(gen);
    }
  }

  is_init_ = true;
  return SUCCESS;
}

SimpleRNN::RC SimpleRNN::Forward(const Matrix &input_sequence,
                                 Matrix &hidden_sequence) {
  if (!is_init_) {
    err_msg_ = "[SimpleRNN::Forward] SimpleRNN not init";
    return NOT_INIT;
  }
  if (input_sequence.empty()) {
    err_msg_ = "[SimpleRNN::Forward] Invalid input sequence";
    return INVALID_DATA;
  }
  for (const auto &input : input_sequence) {
    if (static_cast<int>(input.size()) != input_dim_) {
      err_msg_ = "[SimpleRNN::Forward] Invalid input sequence";
      return INVALID_DATA;
    }
  }

  last_input_sequence_ = input_sequence;
  last_hidden_sequence_.assign(input_sequence.size() + 1,
                               std::vector<double>(hidden_dim_, 0.0));
  hidden_sequence.assign(input_sequence.size(), std::vector<double>(hidden_dim_, 0.0));

  for (int t = 0; t < static_cast<int>(input_sequence.size()); t++) {
    const auto &input = input_sequence[t];
    const auto &prev_hidden = last_hidden_sequence_[t];
    auto &hidden = hidden_sequence[t];
    for (int h = 0; h < hidden_dim_; h++) {
      double sum = bias_[h];
      for (int i = 0; i < input_dim_; i++) {
        sum += input_weight_[h][i] * input[i];
      }
      for (int i = 0; i < hidden_dim_; i++) {
        sum += hidden_weight_[h][i] * prev_hidden[i];
      }
      hidden[h] = std::tanh(sum);
    }
    last_hidden_sequence_[t + 1] = hidden;
  }

  has_forward_cache_ = true;
  return SUCCESS;
}

SimpleRNN::RC SimpleRNN::Backward(const Matrix &grad_hidden_sequence,
                                  Matrix &grad_input_sequence) {
  if (!is_init_) {
    err_msg_ = "[SimpleRNN::Backward] SimpleRNN not init";
    return NOT_INIT;
  }
  if (!has_forward_cache_) {
    err_msg_ = "[SimpleRNN::Backward] Missing forward cache";
    return INVALID_DATA;
  }
  if (grad_hidden_sequence.size() + 1 != last_hidden_sequence_.size()) {
    err_msg_ = "[SimpleRNN::Backward] Invalid grad sequence";
    return INVALID_DATA;
  }
  for (const auto &grad_hidden : grad_hidden_sequence) {
    if (static_cast<int>(grad_hidden.size()) != hidden_dim_) {
      err_msg_ = "[SimpleRNN::Backward] Invalid grad sequence";
      return INVALID_DATA;
    }
  }

  ResetGradients();
  grad_input_sequence.assign(grad_hidden_sequence.size(),
                             std::vector<double>(input_dim_, 0.0));
  std::vector<double> grad_from_future(hidden_dim_, 0.0);

  for (int t = static_cast<int>(grad_hidden_sequence.size()) - 1; t >= 0; t--) {
    const auto &input = last_input_sequence_[t];
    const auto &prev_hidden = last_hidden_sequence_[t];
    const auto &hidden = last_hidden_sequence_[t + 1];
    std::vector<double> grad_hidden_total = grad_hidden_sequence[t];
    for (int h = 0; h < hidden_dim_; h++) {
      grad_hidden_total[h] += grad_from_future[h];
    }

    std::vector<double> grad_from_prev(hidden_dim_, 0.0);
    for (int h = 0; h < hidden_dim_; h++) {
      const double dz = grad_hidden_total[h] * (1.0 - hidden[h] * hidden[h]);
      grad_bias_[h] += dz;
      for (int i = 0; i < input_dim_; i++) {
        grad_input_weight_[h][i] += dz * input[i];
        grad_input_sequence[t][i] += input_weight_[h][i] * dz;
      }
      for (int i = 0; i < hidden_dim_; i++) {
        grad_hidden_weight_[h][i] += dz * prev_hidden[i];
        grad_from_prev[i] += hidden_weight_[h][i] * dz;
      }
    }
    grad_from_future = grad_from_prev;
  }
  return SUCCESS;
}

void SimpleRNN::ApplyGradient(double learning_rate) {
  if (learning_rate <= 0.0) {
    return;
  }
  input_weight_optimizer_.Apply(input_weight_, grad_input_weight_, learning_rate);
  hidden_weight_optimizer_.Apply(hidden_weight_, grad_hidden_weight_, learning_rate);
  bias_optimizer_.Apply(bias_, grad_bias_, learning_rate);
}

void SimpleRNN::set_random_seed(int seed) { rand_seed_ = seed; }

SimpleRNN::RC SimpleRNN::set_input_weight(const Matrix &weight) {
  if (!is_init_) {
    err_msg_ = "[SimpleRNN::set_input_weight] SimpleRNN not init";
    return NOT_INIT;
  }
  if (static_cast<int>(weight.size()) != hidden_dim_) {
    err_msg_ = "[SimpleRNN::set_input_weight] Invalid weight size";
    return INVALID_DATA;
  }
  for (const auto &row : weight) {
    if (static_cast<int>(row.size()) != input_dim_) {
      err_msg_ = "[SimpleRNN::set_input_weight] Invalid weight size";
      return INVALID_DATA;
    }
  }
  input_weight_ = weight;
  return SUCCESS;
}

SimpleRNN::RC SimpleRNN::set_hidden_weight(const Matrix &weight) {
  if (!is_init_) {
    err_msg_ = "[SimpleRNN::set_hidden_weight] SimpleRNN not init";
    return NOT_INIT;
  }
  if (static_cast<int>(weight.size()) != hidden_dim_) {
    err_msg_ = "[SimpleRNN::set_hidden_weight] Invalid weight size";
    return INVALID_DATA;
  }
  for (const auto &row : weight) {
    if (static_cast<int>(row.size()) != hidden_dim_) {
      err_msg_ = "[SimpleRNN::set_hidden_weight] Invalid weight size";
      return INVALID_DATA;
    }
  }
  hidden_weight_ = weight;
  return SUCCESS;
}

SimpleRNN::RC SimpleRNN::set_bias(const std::vector<double> &bias) {
  if (!is_init_) {
    err_msg_ = "[SimpleRNN::set_bias] SimpleRNN not init";
    return NOT_INIT;
  }
  if (static_cast<int>(bias.size()) != hidden_dim_) {
    err_msg_ = "[SimpleRNN::set_bias] Invalid bias size";
    return INVALID_DATA;
  }
  bias_ = bias;
  return SUCCESS;
}

std::string SimpleRNN::err_msg() { return err_msg_; }

int SimpleRNN::input_dim() const { return input_dim_; }

int SimpleRNN::hidden_dim() const { return hidden_dim_; }

const SimpleRNN::Matrix &SimpleRNN::input_weight() const { return input_weight_; }

const SimpleRNN::Matrix &SimpleRNN::hidden_weight() const { return hidden_weight_; }

const std::vector<double> &SimpleRNN::bias() const { return bias_; }

const SimpleRNN::Matrix &SimpleRNN::grad_input_weight() const {
  return grad_input_weight_;
}

const SimpleRNN::Matrix &SimpleRNN::grad_hidden_weight() const {
  return grad_hidden_weight_;
}

const std::vector<double> &SimpleRNN::grad_bias() const { return grad_bias_; }

double SimpleRNN::GradSquaredNorm() const {
  double sq_sum = 0.0;
  for (const auto &row : grad_input_weight_) {
    for (double value : row) {
      sq_sum += value * value;
    }
  }
  for (const auto &row : grad_hidden_weight_) {
    for (double value : row) {
      sq_sum += value * value;
    }
  }
  for (double value : grad_bias_) {
    sq_sum += value * value;
  }
  return sq_sum;
}

void SimpleRNN::ScaleGradients(double scale) {
  for (auto &row : grad_input_weight_) {
    for (double &value : row) {
      value *= scale;
    }
  }
  for (auto &row : grad_hidden_weight_) {
    for (double &value : row) {
      value *= scale;
    }
  }
  for (double &value : grad_bias_) {
    value *= scale;
  }
}

void SimpleRNN::ResetGradients() {
  for (auto &row : grad_input_weight_) {
    std::fill(row.begin(), row.end(), 0.0);
  }
  for (auto &row : grad_hidden_weight_) {
    std::fill(row.begin(), row.end(), 0.0);
  }
  std::fill(grad_bias_.begin(), grad_bias_.end(), 0.0);
}

} // namespace deeplearning
