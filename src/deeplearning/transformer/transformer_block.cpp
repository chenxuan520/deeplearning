#include "transformer_block.h"

#include <cmath>
#include <random>

namespace deeplearning {
namespace {

using Matrix = TransformerBlock::Matrix;

std::vector<double> ApplyLinear(const std::vector<double> &input,
                                const Matrix &weight,
                                const std::vector<double> &bias) {
  std::vector<double> result(weight.size(), 0);
  for (int row = 0; row < static_cast<int>(weight.size()); row++) {
    result[row] = bias[row];
    for (int col = 0; col < static_cast<int>(input.size()); col++) {
      result[row] += weight[row][col] * input[col];
    }
  }
  return result;
}

double Relu(double value) { return value > 0 ? value : 0; }

void AddMatrix(Matrix &target, const Matrix &source) {
  for (int row = 0; row < static_cast<int>(target.size()); row++) {
    for (int col = 0; col < static_cast<int>(target[row].size()); col++) {
      target[row][col] += source[row][col];
    }
  }
}

void AddVector(std::vector<double> &target,
               const std::vector<double> &source) {
  for (int i = 0; i < static_cast<int>(target.size()); i++) {
    target[i] += source[i];
  }
}

} // namespace

TransformerBlock::RC TransformerBlock::Init(int model_dim, int head_num,
                                            int feed_forward_dim) {
  if (is_init_) {
    err_msg_ = "[TransformerBlock::Init] TransformerBlock has init";
    return ALREADY_INIT;
  }
  if (model_dim <= 0 || head_num <= 0 || feed_forward_dim <= 0 ||
      model_dim % head_num != 0) {
    err_msg_ = "[TransformerBlock::Init] Invalid config";
    return INVALID_DATA;
  }

  model_dim_ = model_dim;
  head_num_ = head_num;
  feed_forward_dim_ = feed_forward_dim;
  self_attention_.set_random_seed(rand_seed_);
  if (self_attention_.Init(model_dim_, head_num_) != SelfAttention::SUCCESS) {
    err_msg_ = self_attention_.err_msg();
    return INVALID_DATA;
  }
  if (attention_norm_.Init(model_dim_) != LayerNorm::SUCCESS) {
    err_msg_ = attention_norm_.err_msg();
    return INVALID_DATA;
  }
  if (feed_forward_norm_.Init(model_dim_) != LayerNorm::SUCCESS) {
    err_msg_ = feed_forward_norm_.err_msg();
    return INVALID_DATA;
  }

  std::mt19937 gen(rand_seed_ + 1);
  feed_forward_weight_1_.assign(feed_forward_dim_,
                                std::vector<double>(model_dim_, 0));
  feed_forward_weight_2_.assign(model_dim_,
                                std::vector<double>(feed_forward_dim_, 0));
  feed_forward_bias_1_.assign(feed_forward_dim_, 0);
  feed_forward_bias_2_.assign(model_dim_, 0);
  grad_feed_forward_weight_1_.assign(feed_forward_dim_,
                                     std::vector<double>(model_dim_, 0));
  grad_feed_forward_weight_2_.assign(model_dim_,
                                     std::vector<double>(feed_forward_dim_, 0));
  grad_feed_forward_bias_1_.assign(feed_forward_dim_, 0);
  grad_feed_forward_bias_2_.assign(model_dim_, 0);
  InitWeight(feed_forward_weight_1_, gen);
  InitWeight(feed_forward_weight_2_, gen);
  is_init_ = true;
  return SUCCESS;
}

TransformerBlock::RC
TransformerBlock::Forward(const Matrix &input, Matrix &output,
                          const Matrix *mask) {
  if (!is_init_) {
    err_msg_ = "[TransformerBlock::Forward] TransformerBlock not init";
    return NOT_INIT;
  }
  auto rc = ValidateInput(input);
  if (rc != SUCCESS) {
    return rc;
  }

  last_input_ = input;

  auto attention_rc = self_attention_.Forward(input, last_attention_output_, mask);
  if (attention_rc != SelfAttention::SUCCESS) {
    err_msg_ = self_attention_.err_msg();
    return INVALID_DATA;
  }

  last_residual_1_ = input;
  for (int i = 0; i < static_cast<int>(last_residual_1_.size()); i++) {
    for (int j = 0; j < model_dim_; j++) {
      last_residual_1_[i][j] += last_attention_output_[i][j];
    }
  }

  if (attention_norm_.Forward(last_residual_1_, last_norm_1_) != LayerNorm::SUCCESS) {
    err_msg_ = attention_norm_.err_msg();
    return INVALID_DATA;
  }

  last_feed_forward_hidden_linear_.assign(last_norm_1_.size(),
                                          std::vector<double>(feed_forward_dim_, 0));
  last_feed_forward_hidden_.assign(last_norm_1_.size(),
                                   std::vector<double>(feed_forward_dim_, 0));
  last_feed_forward_output_.assign(last_norm_1_.size(),
                                   std::vector<double>(model_dim_, 0));
  for (int i = 0; i < static_cast<int>(last_norm_1_.size()); i++) {
    last_feed_forward_hidden_linear_[i] =
        ApplyLinear(last_norm_1_[i], feed_forward_weight_1_, feed_forward_bias_1_);
    last_feed_forward_hidden_[i] = last_feed_forward_hidden_linear_[i];
    for (double &value : last_feed_forward_hidden_[i]) {
      value = Relu(value);
    }
    last_feed_forward_output_[i] = ApplyLinear(last_feed_forward_hidden_[i],
                                               feed_forward_weight_2_,
                                               feed_forward_bias_2_);
  }

  last_residual_2_ = last_norm_1_;
  for (int i = 0; i < static_cast<int>(last_residual_2_.size()); i++) {
    for (int j = 0; j < model_dim_; j++) {
      last_residual_2_[i][j] += last_feed_forward_output_[i][j];
    }
  }

  if (feed_forward_norm_.Forward(last_residual_2_, output) != LayerNorm::SUCCESS) {
    err_msg_ = feed_forward_norm_.err_msg();
    return INVALID_DATA;
  }
  return SUCCESS;
}

TransformerBlock::RC TransformerBlock::Backward(const Matrix &grad_output,
                                                Matrix &grad_input,
                                                double learning_rate) {
  ClearGradients();
  auto rc = BackwardAccumulate(grad_output, grad_input);
  if (rc != SUCCESS) {
    return rc;
  }
  ApplyGradient(learning_rate);
  return SUCCESS;
}

TransformerBlock::RC
TransformerBlock::BackwardAccumulate(const Matrix &grad_output,
                                     Matrix &grad_input) {
  if (!is_init_) {
    err_msg_ =
        "[TransformerBlock::BackwardAccumulate] TransformerBlock not init";
    return NOT_INIT;
  }
  if (grad_output.size() != last_input_.size() || grad_output.empty()) {
    err_msg_ =
        "[TransformerBlock::BackwardAccumulate] Invalid data input";
    return INVALID_DATA;
  }

  Matrix grad_residual_2;
  if (feed_forward_norm_.BackwardAccumulate(grad_output, grad_residual_2) !=
      LayerNorm::SUCCESS) {
    err_msg_ = feed_forward_norm_.err_msg();
    return INVALID_DATA;
  }

  Matrix grad_norm_1 = grad_residual_2;
  Matrix grad_feed_forward_output = grad_residual_2;
  Matrix grad_hidden(last_feed_forward_hidden_.size(),
                     std::vector<double>(feed_forward_dim_, 0));
  Matrix grad_hidden_linear(last_feed_forward_hidden_.size(),
                            std::vector<double>(feed_forward_dim_, 0));

  for (int token_idx = 0;
       token_idx < static_cast<int>(grad_feed_forward_output.size());
       token_idx++) {
    for (int out = 0; out < model_dim_; out++) {
      grad_feed_forward_bias_2_[out] +=
          grad_feed_forward_output[token_idx][out];
      for (int in = 0; in < feed_forward_dim_; in++) {
        grad_feed_forward_weight_2_[out][in] +=
            grad_feed_forward_output[token_idx][out] * last_feed_forward_hidden_[token_idx][in];
        grad_hidden[token_idx][in] +=
            feed_forward_weight_2_[out][in] * grad_feed_forward_output[token_idx][out];
      }
    }

    for (int i = 0; i < feed_forward_dim_; i++) {
      grad_hidden_linear[token_idx][i] =
          last_feed_forward_hidden_linear_[token_idx][i] > 0 ? grad_hidden[token_idx][i] : 0;
      grad_feed_forward_bias_1_[i] += grad_hidden_linear[token_idx][i];
      for (int j = 0; j < model_dim_; j++) {
        grad_feed_forward_weight_1_[i][j] +=
            grad_hidden_linear[token_idx][i] * last_norm_1_[token_idx][j];
        grad_norm_1[token_idx][j] +=
            feed_forward_weight_1_[i][j] * grad_hidden_linear[token_idx][i];
      }
    }
  }

  Matrix grad_residual_1;
  if (attention_norm_.BackwardAccumulate(grad_norm_1, grad_residual_1) !=
      LayerNorm::SUCCESS) {
    err_msg_ = attention_norm_.err_msg();
    return INVALID_DATA;
  }

  Matrix grad_attention_output = grad_residual_1;
  Matrix grad_attention_input;
  if (self_attention_.BackwardAccumulate(grad_attention_output,
                                         grad_attention_input) !=
      SelfAttention::SUCCESS) {
    err_msg_ = self_attention_.err_msg();
    return INVALID_DATA;
  }

  grad_input = grad_residual_1;
  for (int i = 0; i < static_cast<int>(grad_input.size()); i++) {
    for (int j = 0; j < model_dim_; j++) {
      grad_input[i][j] += grad_attention_input[i][j];
    }
  }
  return SUCCESS;
}

void TransformerBlock::ApplyGradient(double learning_rate,
                                     double gradient_scale) {
  feed_forward_weight_2_optimizer_.Apply(feed_forward_weight_2_,
                                         grad_feed_forward_weight_2_,
                                         learning_rate, gradient_scale);
  feed_forward_bias_2_optimizer_.Apply(feed_forward_bias_2_,
                                       grad_feed_forward_bias_2_,
                                       learning_rate, gradient_scale);
  feed_forward_weight_1_optimizer_.Apply(feed_forward_weight_1_,
                                         grad_feed_forward_weight_1_,
                                         learning_rate, gradient_scale);
  feed_forward_bias_1_optimizer_.Apply(feed_forward_bias_1_,
                                       grad_feed_forward_bias_1_,
                                       learning_rate, gradient_scale);
  attention_norm_.ApplyGradient(learning_rate, gradient_scale);
  feed_forward_norm_.ApplyGradient(learning_rate, gradient_scale);
  self_attention_.ApplyGradient(learning_rate, gradient_scale);
  ClearGradients();
}

void TransformerBlock::ClearGradients() {
  grad_feed_forward_weight_1_.assign(
      feed_forward_dim_, std::vector<double>(model_dim_, 0));
  grad_feed_forward_weight_2_.assign(
      model_dim_, std::vector<double>(feed_forward_dim_, 0));
  grad_feed_forward_bias_1_.assign(feed_forward_dim_, 0);
  grad_feed_forward_bias_2_.assign(model_dim_, 0);
  attention_norm_.ClearGradients();
  feed_forward_norm_.ClearGradients();
  self_attention_.ClearGradients();
}

void TransformerBlock::AddGradientsFrom(const TransformerBlock &source) {
  AddMatrix(grad_feed_forward_weight_1_,
            source.grad_feed_forward_weight_1_);
  AddMatrix(grad_feed_forward_weight_2_,
            source.grad_feed_forward_weight_2_);
  AddVector(grad_feed_forward_bias_1_, source.grad_feed_forward_bias_1_);
  AddVector(grad_feed_forward_bias_2_, source.grad_feed_forward_bias_2_);
  attention_norm_.AddGradientsFrom(source.attention_norm_);
  feed_forward_norm_.AddGradientsFrom(source.feed_forward_norm_);
  self_attention_.AddGradientsFrom(source.self_attention_);
}

void TransformerBlock::set_random_seed(int seed) { rand_seed_ = seed; }

TransformerBlock::RC
TransformerBlock::set_feed_forward_weight_1(const Matrix &weight) {
  auto rc = ValidateWeightShape(weight, feed_forward_dim_, model_dim_,
                                "set_feed_forward_weight_1");
  if (rc != SUCCESS) {
    return rc;
  }
  feed_forward_weight_1_ = weight;
  return SUCCESS;
}

TransformerBlock::RC
TransformerBlock::set_feed_forward_bias_1(const std::vector<double> &bias) {
  if (!is_init_) {
    err_msg_ =
        "[TransformerBlock::set_feed_forward_bias_1] TransformerBlock not init";
    return NOT_INIT;
  }
  if (bias.size() != static_cast<size_t>(feed_forward_dim_)) {
    err_msg_ =
        "[TransformerBlock::set_feed_forward_bias_1] Invalid bias size";
    return INVALID_DATA;
  }
  feed_forward_bias_1_ = bias;
  return SUCCESS;
}

TransformerBlock::RC
TransformerBlock::set_feed_forward_weight_2(const Matrix &weight) {
  auto rc = ValidateWeightShape(weight, model_dim_, feed_forward_dim_,
                                "set_feed_forward_weight_2");
  if (rc != SUCCESS) {
    return rc;
  }
  feed_forward_weight_2_ = weight;
  return SUCCESS;
}

TransformerBlock::RC
TransformerBlock::set_feed_forward_bias_2(const std::vector<double> &bias) {
  if (!is_init_) {
    err_msg_ =
        "[TransformerBlock::set_feed_forward_bias_2] TransformerBlock not init";
    return NOT_INIT;
  }
  if (bias.size() != static_cast<size_t>(model_dim_)) {
    err_msg_ =
        "[TransformerBlock::set_feed_forward_bias_2] Invalid bias size";
    return INVALID_DATA;
  }
  feed_forward_bias_2_ = bias;
  return SUCCESS;
}

SelfAttention &TransformerBlock::self_attention() { return self_attention_; }

const SelfAttention &TransformerBlock::self_attention() const {
  return self_attention_;
}

LayerNorm &TransformerBlock::attention_norm() { return attention_norm_; }

const LayerNorm &TransformerBlock::attention_norm() const {
  return attention_norm_;
}

LayerNorm &TransformerBlock::feed_forward_norm() { return feed_forward_norm_; }

const LayerNorm &TransformerBlock::feed_forward_norm() const {
  return feed_forward_norm_;
}

const TransformerBlock::Matrix &TransformerBlock::feed_forward_weight_1() const {
  return feed_forward_weight_1_;
}

const std::vector<double> &TransformerBlock::feed_forward_bias_1() const {
  return feed_forward_bias_1_;
}

const TransformerBlock::Matrix &TransformerBlock::feed_forward_weight_2() const {
  return feed_forward_weight_2_;
}

const std::vector<double> &TransformerBlock::feed_forward_bias_2() const {
  return feed_forward_bias_2_;
}

std::string TransformerBlock::err_msg() { return err_msg_; }

TransformerBlock::RC TransformerBlock::ValidateInput(const Matrix &input) {
  if (input.empty()) {
    err_msg_ = "[TransformerBlock::ValidateInput] Invalid data input";
    return INVALID_DATA;
  }
  for (const auto &token : input) {
    if (token.size() != static_cast<size_t>(model_dim_)) {
      err_msg_ = "[TransformerBlock::ValidateInput] Invalid data input";
      return INVALID_DATA;
    }
  }
  return SUCCESS;
}

TransformerBlock::RC
TransformerBlock::ValidateWeightShape(const Matrix &weight, int row, int col,
                                      const char *func_name) {
  if (!is_init_) {
    err_msg_ = std::string("[TransformerBlock::") + func_name +
               "] TransformerBlock not init";
    return NOT_INIT;
  }
  if (weight.size() != static_cast<size_t>(row)) {
    err_msg_ = std::string("[TransformerBlock::") + func_name +
               "] Invalid weight size";
    return INVALID_DATA;
  }
  for (const auto &weight_row : weight) {
    if (weight_row.size() != static_cast<size_t>(col)) {
      err_msg_ = std::string("[TransformerBlock::") + func_name +
                 "] Invalid weight size";
      return INVALID_DATA;
    }
  }
  return SUCCESS;
}

void TransformerBlock::InitWeight(Matrix &weight, std::mt19937 &gen) {
  const int row = static_cast<int>(weight.size());
  const int col = row == 0 ? 0 : static_cast<int>(weight[0].size());
  const double limit = std::sqrt(6.0 / (row + col));
  std::uniform_real_distribution<double> dist(-limit, limit);
  for (auto &weight_row : weight) {
    for (double &value : weight_row) {
      value = dist(gen);
    }
  }
}

} // namespace deeplearning
