#include "rnn/mini_rnn_lm.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace deeplearning {
namespace {

using Matrix = MiniRNNLM::Matrix;

std::vector<double> Softmax(const std::vector<double> &logits) {
  std::vector<double> probs(logits.size(), 0.0);
  if (logits.empty()) {
    return probs;
  }
  double max_logit = logits[0];
  for (double value : logits) {
    if (value > max_logit) {
      max_logit = value;
    }
  }
  double sum = 0.0;
  for (int i = 0; i < static_cast<int>(logits.size()); i++) {
    probs[i] = std::exp(logits[i] - max_logit);
    sum += probs[i];
  }
  if (sum == 0.0) {
    return probs;
  }
  for (double &value : probs) {
    value /= sum;
  }
  return probs;
}

double MatrixSquaredNorm(const Matrix &matrix) {
  double sq_sum = 0.0;
  for (const auto &row : matrix) {
    for (double value : row) {
      sq_sum += value * value;
    }
  }
  return sq_sum;
}

double VectorSquaredNorm(const std::vector<double> &values) {
  double sq_sum = 0.0;
  for (double value : values) {
    sq_sum += value * value;
  }
  return sq_sum;
}

void ScaleMatrix(Matrix &matrix, double scale) {
  for (auto &row : matrix) {
    for (double &value : row) {
      value *= scale;
    }
  }
}

void ScaleVector(std::vector<double> &values, double scale) {
  for (double &value : values) {
    value *= scale;
  }
}

} // namespace

MiniRNNLM::RC MiniRNNLM::Init(const Config &config) {
  if (is_init_) {
    err_msg_ = "[MiniRNNLM::Init] MiniRNNLM has init";
    return ALREADY_INIT;
  }
  if (config.vocab_size_ <= 0 || config.hidden_dim_ <= 0) {
    err_msg_ = "[MiniRNNLM::Init] Invalid config";
    return INVALID_DATA;
  }

  vocab_size_ = config.vocab_size_;
  hidden_dim_ = config.hidden_dim_;
  rand_seed_ = config.rand_seed_;
  gradient_clip_norm_ = config.gradient_clip_norm_;

  rnn_.set_random_seed(rand_seed_);
  auto rnn_rc = rnn_.Init(vocab_size_, hidden_dim_);
  if (rnn_rc != SimpleRNN::SUCCESS) {
    err_msg_ = rnn_.err_msg();
    return INVALID_DATA;
  }

  output_weight_.assign(vocab_size_, std::vector<double>(hidden_dim_, 0.0));
  output_bias_.assign(vocab_size_, 0.0);
  const double limit =
      std::sqrt(6.0 / static_cast<double>(vocab_size_ + hidden_dim_));
  std::mt19937 gen(static_cast<std::mt19937::result_type>(rand_seed_ + 19));
  std::uniform_real_distribution<double> dist(-limit, limit);
  for (auto &row : output_weight_) {
    for (double &value : row) {
      value = dist(gen);
    }
  }

  is_init_ = true;
  return SUCCESS;
}

MiniRNNLM::RC MiniRNNLM::Init(int vocab_size, int hidden_dim) {
  Config config;
  config.vocab_size_ = vocab_size;
  config.hidden_dim_ = hidden_dim;
  config.rand_seed_ = rand_seed_;
  config.gradient_clip_norm_ = gradient_clip_norm_;
  return Init(config);
}

MiniRNNLM::RC MiniRNNLM::Forward(const std::vector<int> &token_ids,
                                 Matrix &logits) {
  if (!is_init_) {
    err_msg_ = "[MiniRNNLM::Forward] MiniRNNLM not init";
    return NOT_INIT;
  }
  Matrix input_sequence;
  auto encode_rc = EncodeTokenIds(token_ids, input_sequence);
  if (encode_rc != SUCCESS) {
    return encode_rc;
  }

  Matrix hidden_sequence;
  auto rnn_rc = rnn_.Forward(input_sequence, hidden_sequence);
  if (rnn_rc != SimpleRNN::SUCCESS) {
    err_msg_ = rnn_.err_msg();
    return INVALID_DATA;
  }

  logits.assign(hidden_sequence.size(), std::vector<double>(vocab_size_, 0.0));
  for (int pos = 0; pos < static_cast<int>(hidden_sequence.size()); pos++) {
    for (int token_id = 0; token_id < vocab_size_; token_id++) {
      logits[pos][token_id] = output_bias_[token_id];
      for (int dim = 0; dim < hidden_dim_; dim++) {
        logits[pos][token_id] +=
            output_weight_[token_id][dim] * hidden_sequence[pos][dim];
      }
    }
  }
  return SUCCESS;
}

MiniRNNLM::RC MiniRNNLM::PredictNextToken(const std::vector<int> &token_ids,
                                          int &token_id) {
  Matrix logits;
  auto rc = Forward(token_ids, logits);
  if (rc != SUCCESS) {
    return rc;
  }
  token_id = 0;
  const auto &last_logits = logits.back();
  for (int i = 1; i < static_cast<int>(last_logits.size()); i++) {
    if (last_logits[i] > last_logits[token_id]) {
      token_id = i;
    }
  }
  return SUCCESS;
}

MiniRNNLM::RC MiniRNNLM::Generate(
    const std::vector<int> &prompt_token_ids, int generate_num,
    std::vector<int> &generated_token_ids) {
  if (!is_init_) {
    err_msg_ = "[MiniRNNLM::Generate] MiniRNNLM not init";
    return NOT_INIT;
  }
  if (prompt_token_ids.empty() || generate_num < 0) {
    err_msg_ = "[MiniRNNLM::Generate] Invalid token input";
    return INVALID_DATA;
  }

  generated_token_ids = prompt_token_ids;
  for (int i = 0; i < generate_num; i++) {
    int token_id = 0;
    auto rc = PredictNextToken(generated_token_ids, token_id);
    if (rc != SUCCESS) {
      return rc;
    }
    generated_token_ids.push_back(token_id);
  }
  return SUCCESS;
}

MiniRNNLM::RC MiniRNNLM::CalcNextTokenLoss(
    const std::vector<std::vector<int>> &input_samples,
    const std::vector<int> &target_tokens, double &average_loss) {
  if (!is_init_) {
    err_msg_ = "[MiniRNNLM::CalcNextTokenLoss] MiniRNNLM not init";
    return NOT_INIT;
  }
  if (input_samples.empty() || input_samples.size() != target_tokens.size()) {
    err_msg_ = "[MiniRNNLM::CalcNextTokenLoss] Invalid input";
    return INVALID_DATA;
  }

  double loss_sum = 0.0;
  long long loss_count = 0;
  for (int sample_idx = 0; sample_idx < static_cast<int>(input_samples.size());
       sample_idx++) {
    if (target_tokens[sample_idx] < 0 || target_tokens[sample_idx] >= vocab_size_) {
      err_msg_ = "[MiniRNNLM::CalcNextTokenLoss] Invalid target token";
      return INVALID_DATA;
    }
    Matrix logits;
    auto rc = Forward(input_samples[sample_idx], logits);
    if (rc != SUCCESS) {
      return rc;
    }
    std::vector<int> position_targets(input_samples[sample_idx].size());
    for (int i = 0; i + 1 < static_cast<int>(input_samples[sample_idx].size()); i++) {
      position_targets[i] = input_samples[sample_idx][i + 1];
    }
    position_targets.back() = target_tokens[sample_idx];
    for (int pos = 0; pos < static_cast<int>(logits.size()); pos++) {
      auto probs = Softmax(logits[pos]);
      loss_sum += -std::log(std::max(probs[position_targets[pos]], 1e-12));
      loss_count++;
    }
  }

  average_loss = loss_count == 0 ? 0.0 : loss_sum / loss_count;
  return SUCCESS;
}

MiniRNNLM::RC MiniRNNLM::CalcPerplexity(
    const std::vector<std::vector<int>> &input_samples,
    const std::vector<int> &target_tokens, double &perplexity) {
  double average_loss = 0.0;
  auto rc = CalcNextTokenLoss(input_samples, target_tokens, average_loss);
  if (rc != SUCCESS) {
    return rc;
  }
  perplexity = std::exp(average_loss);
  return SUCCESS;
}

MiniRNNLM::RC MiniRNNLM::TrainNextToken(
    const std::vector<std::vector<int>> &input_samples,
    const std::vector<int> &target_tokens,
    std::function<void(int epoch_num, double average_loss, bool &early_stop)>
        each_epoch_call,
    int epoch_num, double learning_rate, LRScheduler *lr_scheduler) {
  if (!is_init_) {
    err_msg_ = "[MiniRNNLM::TrainNextToken] MiniRNNLM not init";
    return NOT_INIT;
  }
  if (input_samples.empty() || input_samples.size() != target_tokens.size() ||
      epoch_num <= 0 || learning_rate <= 0.0) {
    err_msg_ = "[MiniRNNLM::TrainNextToken] Invalid training input";
    return INVALID_DATA;
  }

  std::vector<int> order(input_samples.size(), 0);
  for (int i = 0; i < static_cast<int>(input_samples.size()); i++) {
    order[i] = i;
  }
  std::mt19937 gen(static_cast<std::mt19937::result_type>(rand_seed_));

  for (int epoch = 0; epoch < epoch_num; epoch++) {
    std::shuffle(order.begin(), order.end(), gen);
    const double epoch_learning_rate =
        lr_scheduler != nullptr ? lr_scheduler->GetLR(epoch) : learning_rate;
    double loss_sum = 0.0;
    long long loss_count = 0;
    for (int pos = 0; pos < static_cast<int>(order.size()); pos++) {
      const int sample_idx = order[pos];
      const auto &sample = input_samples[sample_idx];
      if (sample.empty()) {
        err_msg_ = "[MiniRNNLM::TrainNextToken] Empty sample";
        return INVALID_DATA;
      }
      if (target_tokens[sample_idx] < 0 ||
          target_tokens[sample_idx] >= vocab_size_) {
        err_msg_ = "[MiniRNNLM::TrainNextToken] Invalid target token";
        return INVALID_DATA;
      }

      std::vector<int> position_targets(sample.size());
      for (int i = 0; i + 1 < static_cast<int>(sample.size()); i++) {
        position_targets[i] = sample[i + 1];
      }
      position_targets.back() = target_tokens[sample_idx];

      Matrix input_sequence;
      auto encode_rc = EncodeTokenIds(sample, input_sequence);
      if (encode_rc != SUCCESS) {
        return encode_rc;
      }
      Matrix hidden_sequence;
      auto rnn_forward_rc = rnn_.Forward(input_sequence, hidden_sequence);
      if (rnn_forward_rc != SimpleRNN::SUCCESS) {
        err_msg_ = rnn_.err_msg();
        return INVALID_DATA;
      }

      const int seq_len = static_cast<int>(sample.size());
      const double inv_seq_len = 1.0 / seq_len;
      Matrix grad_hidden(seq_len, std::vector<double>(hidden_dim_, 0.0));
      Matrix grad_output_weight(vocab_size_, std::vector<double>(hidden_dim_, 0.0));
      std::vector<double> grad_output_bias(vocab_size_, 0.0);

      for (int t = 0; t < seq_len; t++) {
        std::vector<double> logits(vocab_size_, 0.0);
        for (int token_id = 0; token_id < vocab_size_; token_id++) {
          logits[token_id] = output_bias_[token_id];
          for (int dim = 0; dim < hidden_dim_; dim++) {
            logits[token_id] +=
                output_weight_[token_id][dim] * hidden_sequence[t][dim];
          }
        }
        auto probs = Softmax(logits);
        loss_sum += -std::log(std::max(probs[position_targets[t]], 1e-12));
        loss_count++;

        std::vector<double> grad_logits = probs;
        grad_logits[position_targets[t]] -= 1.0;
        for (double &value : grad_logits) {
          value *= inv_seq_len;
        }

        for (int token_id = 0; token_id < vocab_size_; token_id++) {
          const double grad = grad_logits[token_id];
          grad_output_bias[token_id] += grad;
          for (int dim = 0; dim < hidden_dim_; dim++) {
            grad_output_weight[token_id][dim] += grad * hidden_sequence[t][dim];
            grad_hidden[t][dim] += output_weight_[token_id][dim] * grad;
          }
        }
      }

      Matrix grad_inputs;
      auto rnn_backward_rc = rnn_.Backward(grad_hidden, grad_inputs);
      if (rnn_backward_rc != SimpleRNN::SUCCESS) {
        err_msg_ = rnn_.err_msg();
        return INVALID_DATA;
      }

      if (gradient_clip_norm_ > 0.0) {
        const double sq_sum = MatrixSquaredNorm(grad_output_weight) +
                              VectorSquaredNorm(grad_output_bias) +
                              rnn_.GradSquaredNorm();
        const double norm = std::sqrt(sq_sum);
        if (norm > gradient_clip_norm_ && norm > 0.0) {
          const double scale = gradient_clip_norm_ / norm;
          ScaleMatrix(grad_output_weight, scale);
          ScaleVector(grad_output_bias, scale);
          rnn_.ScaleGradients(scale);
        }
      }

      output_weight_optimizer_.Apply(output_weight_, grad_output_weight,
                                     epoch_learning_rate);
      output_bias_optimizer_.Apply(output_bias_, grad_output_bias,
                                   epoch_learning_rate);
      rnn_.ApplyGradient(epoch_learning_rate);
    }

    bool early_stop = false;
    if (each_epoch_call != nullptr) {
      each_epoch_call(epoch, loss_count == 0 ? 0.0 : loss_sum / loss_count,
                      early_stop);
    }
    if (early_stop) {
      break;
    }
  }
  return SUCCESS;
}

void MiniRNNLM::set_random_seed(int seed) { rand_seed_ = seed; }

void MiniRNNLM::set_gradient_clip_norm(double max_norm) {
  gradient_clip_norm_ = max_norm;
}

MiniRNNLM::RC MiniRNNLM::set_output_weight(const Matrix &weight) {
  return ValidateOutputWeight(weight, "set_output_weight");
}

MiniRNNLM::RC MiniRNNLM::set_output_bias(const std::vector<double> &bias) {
  if (!is_init_) {
    err_msg_ = "[MiniRNNLM::set_output_bias] MiniRNNLM not init";
    return NOT_INIT;
  }
  if (static_cast<int>(bias.size()) != vocab_size_) {
    err_msg_ = "[MiniRNNLM::set_output_bias] Invalid bias size";
    return INVALID_DATA;
  }
  output_bias_ = bias;
  return SUCCESS;
}

std::string MiniRNNLM::err_msg() { return err_msg_; }

int MiniRNNLM::vocab_size() const { return vocab_size_; }

int MiniRNNLM::hidden_dim() const { return hidden_dim_; }

double MiniRNNLM::gradient_clip_norm() const { return gradient_clip_norm_; }

const MiniRNNLM::Matrix &MiniRNNLM::output_weight() const { return output_weight_; }

const std::vector<double> &MiniRNNLM::output_bias() const { return output_bias_; }

SimpleRNN &MiniRNNLM::rnn() { return rnn_; }

const SimpleRNN &MiniRNNLM::rnn() const { return rnn_; }

MiniRNNLM::RC MiniRNNLM::EncodeTokenIds(const std::vector<int> &token_ids,
                                        Matrix &input_sequence) {
  if (token_ids.empty()) {
    err_msg_ = "[MiniRNNLM::EncodeTokenIds] Invalid token input";
    return INVALID_DATA;
  }
  input_sequence.assign(token_ids.size(), std::vector<double>(vocab_size_, 0.0));
  for (int i = 0; i < static_cast<int>(token_ids.size()); i++) {
    const int token_id = token_ids[i];
    if (token_id < 0 || token_id >= vocab_size_) {
      err_msg_ = "[MiniRNNLM::EncodeTokenIds] Invalid token input";
      return INVALID_DATA;
    }
    input_sequence[i][token_id] = 1.0;
  }
  return SUCCESS;
}

MiniRNNLM::RC MiniRNNLM::ValidateOutputWeight(const Matrix &weight,
                                              const char *func_name) {
  if (!is_init_) {
    err_msg_ = std::string("[MiniRNNLM::") + func_name +
               "] MiniRNNLM not init";
    return NOT_INIT;
  }
  if (static_cast<int>(weight.size()) != vocab_size_) {
    err_msg_ = std::string("[MiniRNNLM::") + func_name +
               "] Invalid weight size";
    return INVALID_DATA;
  }
  for (const auto &row : weight) {
    if (static_cast<int>(row.size()) != hidden_dim_) {
      err_msg_ = std::string("[MiniRNNLM::") + func_name +
                 "] Invalid weight size";
      return INVALID_DATA;
    }
  }
  output_weight_ = weight;
  return SUCCESS;
}

} // namespace deeplearning
