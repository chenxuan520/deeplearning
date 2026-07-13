#include "self_attention.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

namespace deeplearning {
namespace {

using Matrix = SelfAttention::Matrix;

bool IsValidSequence(const Matrix &input, int model_dim) {
  if (input.empty()) {
    return false;
  }
  for (const auto &token : input) {
    if (token.size() != static_cast<size_t>(model_dim)) {
      return false;
    }
  }
  return true;
}

std::vector<double> ProjectToken(const std::vector<double> &token,
                                 const Matrix &weight) {
  std::vector<double> result(weight.size(), 0);
  for (int out = 0; out < static_cast<int>(weight.size()); out++) {
    for (int in = 0; in < static_cast<int>(token.size()); in++) {
      result[out] += weight[out][in] * token[in];
    }
  }
  return result;
}

Matrix ProjectSequence(const Matrix &input, const Matrix &weight) {
  Matrix result;
  result.reserve(input.size());
  for (const auto &token : input) {
    result.push_back(ProjectToken(token, weight));
  }
  return result;
}

std::vector<double> Softmax(const std::vector<double> &score) {
  std::vector<double> weight(score.size(), 0);
  double max_score = *std::max_element(score.begin(), score.end());
  double sum = 0;
  for (int i = 0; i < static_cast<int>(score.size()); i++) {
    if (score[i] <= std::numeric_limits<double>::lowest() / 2) {
      continue;
    }
    weight[i] = std::exp(score[i] - max_score);
    sum += weight[i];
  }
  if (sum == 0) {
    return {};
  }
  for (double &value : weight) {
    value /= sum;
  }
  return weight;
}

} // namespace

SelfAttention::RC SelfAttention::Init(int model_dim, int head_num) {
  if (is_init_) {
    err_msg_ = "[SelfAttention::Init] SelfAttention has init";
    return ALREADY_INIT;
  }
  if (model_dim <= 0 || head_num <= 0 || model_dim % head_num != 0) {
    err_msg_ = "[SelfAttention::Init] Invalid model dim or head num";
    return INVALID_DATA;
  }

  model_dim_ = model_dim;
  head_num_ = head_num;
  head_dim_ = model_dim / head_num;

  std::mt19937 gen(rand_seed_);
  query_weight_.assign(model_dim_, std::vector<double>(model_dim_, 0));
  key_weight_.assign(model_dim_, std::vector<double>(model_dim_, 0));
  value_weight_.assign(model_dim_, std::vector<double>(model_dim_, 0));
  output_weight_.assign(model_dim_, std::vector<double>(model_dim_, 0));
  grad_query_weight_.assign(model_dim_, std::vector<double>(model_dim_, 0));
  grad_key_weight_.assign(model_dim_, std::vector<double>(model_dim_, 0));
  grad_value_weight_.assign(model_dim_, std::vector<double>(model_dim_, 0));
  grad_output_weight_.assign(model_dim_, std::vector<double>(model_dim_, 0));
  InitProjectionWeight(query_weight_, gen);
  InitProjectionWeight(key_weight_, gen);
  InitProjectionWeight(value_weight_, gen);
  InitProjectionWeight(output_weight_, gen);
  is_init_ = true;
  return SUCCESS;
}

SelfAttention::RC SelfAttention::Forward(const Matrix &input, Matrix &output,
                                         const Matrix *mask) {
  if (!is_init_) {
    err_msg_ = "[SelfAttention::Forward] SelfAttention not init";
    return NOT_INIT;
  }
  if (!IsValidSequence(input, model_dim_)) {
    err_msg_ = "[SelfAttention::Forward] Invalid data input";
    return INVALID_DATA;
  }
  if (mask != nullptr) {
    if (mask->size() != input.size()) {
      err_msg_ = "[SelfAttention::Forward] Invalid attention mask";
      return INVALID_DATA;
    }
    for (const auto &row : *mask) {
      if (row.size() != input.size()) {
        err_msg_ = "[SelfAttention::Forward] Invalid attention mask";
        return INVALID_DATA;
      }
    }
  }

  last_input_ = input;
  last_query_ = ProjectSequence(input, query_weight_);
  last_key_ = ProjectSequence(input, key_weight_);
  last_value_ = ProjectSequence(input, value_weight_);
  has_last_mask_ = mask != nullptr;
  last_mask_ = has_last_mask_ ? *mask : Matrix();

  Matrix merged_output(input.size(), std::vector<double>(model_dim_, 0));
  last_attention_weight_.clear();
  last_attention_weight_.reserve(head_num_);
  for (int head = 0; head < head_num_; head++) {
    Matrix head_weight(input.size(), std::vector<double>(input.size(), 0));
    const int head_start = head * head_dim_;
    for (int row = 0; row < static_cast<int>(input.size()); row++) {
      std::vector<double> score(input.size(), 0);
      for (int col = 0; col < static_cast<int>(input.size()); col++) {
        if (mask != nullptr && (*mask)[row][col] <= 0) {
          score[col] = std::numeric_limits<double>::lowest();
          continue;
        }

        double dot = 0;
        for (int i = 0; i < head_dim_; i++) {
          dot += last_query_[row][head_start + i] *
                 last_key_[col][head_start + i];
        }
        score[col] = dot / std::sqrt(static_cast<double>(head_dim_));
      }

      auto weight = Softmax(score);
      if (weight.empty()) {
        err_msg_ = "[SelfAttention::Forward] Invalid attention mask";
        return INVALID_DATA;
      }
        head_weight[row] = weight;
        for (int col = 0; col < static_cast<int>(input.size()); col++) {
          for (int i = 0; i < head_dim_; i++) {
            merged_output[row][head_start + i] +=
                weight[col] * last_value_[col][head_start + i];
          }
        }
      }
    last_attention_weight_.push_back(head_weight);
  }

  last_merged_output_ = merged_output;
  output = ProjectSequence(last_merged_output_, output_weight_);
  return SUCCESS;
}

SelfAttention::RC SelfAttention::Backward(const Matrix &grad_output,
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

SelfAttention::RC SelfAttention::BackwardAccumulate(const Matrix &grad_output,
                                                    Matrix &grad_input) {
  if (!is_init_) {
    err_msg_ = "[SelfAttention::BackwardAccumulate] SelfAttention not init";
    return NOT_INIT;
  }
  if (grad_output.size() != last_input_.size() || grad_output.empty()) {
    err_msg_ = "[SelfAttention::BackwardAccumulate] Invalid data input";
    return INVALID_DATA;
  }

  Matrix grad_merged(last_merged_output_.size(), std::vector<double>(model_dim_, 0));
  for (int token_idx = 0; token_idx < static_cast<int>(grad_output.size());
       token_idx++) {
    if (grad_output[token_idx].size() != static_cast<size_t>(model_dim_)) {
      err_msg_ = "[SelfAttention::BackwardAccumulate] Invalid data input";
      return INVALID_DATA;
    }
    for (int out = 0; out < model_dim_; out++) {
      for (int in = 0; in < model_dim_; in++) {
        grad_output_weight_[out][in] +=
            grad_output[token_idx][out] * last_merged_output_[token_idx][in];
        grad_merged[token_idx][in] += output_weight_[out][in] * grad_output[token_idx][out];
      }
    }
  }

  Matrix grad_query(last_query_.size(), std::vector<double>(model_dim_, 0));
  Matrix grad_key(last_key_.size(), std::vector<double>(model_dim_, 0));
  Matrix grad_value(last_value_.size(), std::vector<double>(model_dim_, 0));
  const double scale = std::sqrt(static_cast<double>(head_dim_));
  for (int head = 0; head < head_num_; head++) {
    int head_start = head * head_dim_;
    for (int row = 0; row < static_cast<int>(last_input_.size()); row++) {
      std::vector<double> grad_weight(last_input_.size(), 0);
      for (int col = 0; col < static_cast<int>(last_input_.size()); col++) {
        for (int i = 0; i < head_dim_; i++) {
          grad_weight[col] += grad_merged[row][head_start + i] *
                              last_value_[col][head_start + i];
          grad_value[col][head_start + i] +=
              last_attention_weight_[head][row][col] * grad_merged[row][head_start + i];
        }
      }

      double weighted_sum = 0;
      for (int col = 0; col < static_cast<int>(last_input_.size()); col++) {
        weighted_sum += grad_weight[col] * last_attention_weight_[head][row][col];
      }

      for (int col = 0; col < static_cast<int>(last_input_.size()); col++) {
        if (has_last_mask_ && last_mask_[row][col] <= 0) {
          continue;
        }
        double grad_score = last_attention_weight_[head][row][col] *
                            (grad_weight[col] - weighted_sum);
        for (int i = 0; i < head_dim_; i++) {
          grad_query[row][head_start + i] +=
              grad_score * last_key_[col][head_start + i] / scale;
          grad_key[col][head_start + i] +=
              grad_score * last_query_[row][head_start + i] / scale;
        }
      }
    }
  }

  grad_input.assign(last_input_.size(), std::vector<double>(model_dim_, 0));
  for (int token_idx = 0; token_idx < static_cast<int>(last_input_.size());
       token_idx++) {
    for (int out = 0; out < model_dim_; out++) {
      for (int in = 0; in < model_dim_; in++) {
        grad_query_weight_[out][in] += grad_query[token_idx][out] * last_input_[token_idx][in];
        grad_key_weight_[out][in] += grad_key[token_idx][out] * last_input_[token_idx][in];
        grad_value_weight_[out][in] += grad_value[token_idx][out] * last_input_[token_idx][in];
        grad_input[token_idx][in] += query_weight_[out][in] * grad_query[token_idx][out] +
                                     key_weight_[out][in] * grad_key[token_idx][out] +
                                     value_weight_[out][in] * grad_value[token_idx][out];
      }
    }
  }

  return SUCCESS;
}

void SelfAttention::ApplyGradient(double learning_rate, double gradient_scale) {
  query_optimizer_.Apply(query_weight_, grad_query_weight_, learning_rate,
                         gradient_scale);
  key_optimizer_.Apply(key_weight_, grad_key_weight_, learning_rate,
                       gradient_scale);
  value_optimizer_.Apply(value_weight_, grad_value_weight_, learning_rate,
                         gradient_scale);
  output_optimizer_.Apply(output_weight_, grad_output_weight_, learning_rate,
                          gradient_scale);
  ClearGradients();
}

void SelfAttention::ClearGradients() {
  grad_query_weight_.assign(model_dim_, std::vector<double>(model_dim_, 0));
  grad_key_weight_.assign(model_dim_, std::vector<double>(model_dim_, 0));
  grad_value_weight_.assign(model_dim_, std::vector<double>(model_dim_, 0));
  grad_output_weight_.assign(model_dim_, std::vector<double>(model_dim_, 0));
}

void SelfAttention::set_random_seed(int seed) { rand_seed_ = seed; }

SelfAttention::RC SelfAttention::set_query_weight(const Matrix &weight) {
  auto rc = ValidateWeight(weight, "set_query_weight");
  if (rc != SUCCESS) {
    return rc;
  }
  query_weight_ = weight;
  return SUCCESS;
}

SelfAttention::RC SelfAttention::set_key_weight(const Matrix &weight) {
  auto rc = ValidateWeight(weight, "set_key_weight");
  if (rc != SUCCESS) {
    return rc;
  }
  key_weight_ = weight;
  return SUCCESS;
}

SelfAttention::RC SelfAttention::set_value_weight(const Matrix &weight) {
  auto rc = ValidateWeight(weight, "set_value_weight");
  if (rc != SUCCESS) {
    return rc;
  }
  value_weight_ = weight;
  return SUCCESS;
}

SelfAttention::RC SelfAttention::set_output_weight(const Matrix &weight) {
  auto rc = ValidateWeight(weight, "set_output_weight");
  if (rc != SUCCESS) {
    return rc;
  }
  output_weight_ = weight;
  return SUCCESS;
}

std::string SelfAttention::err_msg() { return err_msg_; }

const SelfAttention::Matrix &SelfAttention::query_weight() const {
  return query_weight_;
}

const SelfAttention::Matrix &SelfAttention::key_weight() const {
  return key_weight_;
}

const SelfAttention::Matrix &SelfAttention::value_weight() const {
  return value_weight_;
}

const SelfAttention::Matrix &SelfAttention::output_weight() const {
  return output_weight_;
}

const SelfAttention::Tensor3D &SelfAttention::last_attention_weight() const {
  return last_attention_weight_;
}

SelfAttention::RC SelfAttention::ValidateWeight(const Matrix &weight,
                                                const char *func_name) {
  if (!is_init_) {
    err_msg_ = std::string("[SelfAttention::") + func_name +
               "] SelfAttention not init";
    return NOT_INIT;
  }
  if (weight.size() != static_cast<size_t>(model_dim_)) {
    err_msg_ = std::string("[SelfAttention::") + func_name +
               "] Invalid weight size";
    return INVALID_DATA;
  }
  for (const auto &row : weight) {
    if (row.size() != static_cast<size_t>(model_dim_)) {
      err_msg_ = std::string("[SelfAttention::") + func_name +
                 "] Invalid weight size";
      return INVALID_DATA;
    }
  }
  return SUCCESS;
}

void SelfAttention::InitProjectionWeight(Matrix &weight, std::mt19937 &gen) {
  const double limit = std::sqrt(6.0 / (model_dim_ + model_dim_));
  std::uniform_real_distribution<double> dist(-limit, limit);
  for (auto &row : weight) {
    for (double &value : row) {
      value = dist(gen);
    }
  }
}

} // namespace deeplearning
