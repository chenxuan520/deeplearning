#include "mini_transformer_lm.h"

#include "positional_encoding.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace deeplearning {
namespace {

using Matrix = MiniTransformerLM::Matrix;

Matrix CreateCausalMask(int sequence_length) {
  Matrix mask(sequence_length, std::vector<double>(sequence_length, 0));
  for (int i = 0; i < sequence_length; i++) {
    for (int j = 0; j <= i; j++) {
      mask[i][j] = 1.0;
    }
  }
  return mask;
}

std::vector<double> Softmax(const std::vector<double> &logits) {
  std::vector<double> probs(logits.size(), 0);
  if (logits.empty()) {
    return probs;
  }

  double max_logit = logits[0];
  for (int i = 1; i < logits.size(); i++) {
    if (logits[i] > max_logit) {
      max_logit = logits[i];
    }
  }

  double sum = 0;
  for (int i = 0; i < logits.size(); i++) {
    probs[i] = std::exp(logits[i] - max_logit);
    sum += probs[i];
  }
  if (sum == 0) {
    return probs;
  }
  for (double &value : probs) {
    value /= sum;
  }
  return probs;
}

std::vector<double> SoftmaxWithTemperature(const std::vector<double> &logits,
                                           double temperature) {
  std::vector<double> scaled_logits = logits;
  for (double &value : scaled_logits) {
    value /= temperature;
  }
  return Softmax(scaled_logits);
}

} // namespace

MiniTransformerLM::RC MiniTransformerLM::Init(const Config &config) {
  if (is_init_) {
    err_msg_ = "[MiniTransformerLM::Init] MiniTransformerLM has init";
    return ALREADY_INIT;
  }
  if (config.vocab_size_ <= 0 || config.model_dim_ <= 0 ||
      config.head_num_ <= 0 || config.feed_forward_dim_ <= 0 ||
      config.block_num_ < 0) {
    err_msg_ = "[MiniTransformerLM::Init] Invalid config";
    return INVALID_DATA;
  }

  vocab_size_ = config.vocab_size_;
  model_dim_ = config.model_dim_;
  head_num_ = config.head_num_;
  feed_forward_dim_ = config.feed_forward_dim_;
  block_num_ = config.block_num_;
  rand_seed_ = config.rand_seed_;
  backbone_type_ = config.backbone_type_;
  use_positional_encoding_ = config.use_positional_encoding_;
  scale_embedding_ = config.scale_embedding_;
  block_learning_rate_scale_ = config.block_learning_rate_scale_;

  token_embedding_.set_random_seed(rand_seed_);
  if (token_embedding_.Init(vocab_size_, model_dim_) != TokenEmbedding::SUCCESS) {
    err_msg_ = token_embedding_.err_msg();
    return INVALID_DATA;
  }
  encoder_.set_random_seed(rand_seed_ + 7);
  if (encoder_.Init(block_num_, model_dim_, head_num_, feed_forward_dim_) !=
      TransformerEncoder::SUCCESS) {
    err_msg_ = encoder_.err_msg();
    return INVALID_DATA;
  }
  decoder_.set_random_seed(rand_seed_ + 11);
  if (decoder_.Init(block_num_, model_dim_, head_num_, feed_forward_dim_) !=
      TransformerDecoder::SUCCESS) {
    err_msg_ = decoder_.err_msg();
    return INVALID_DATA;
  }

  output_weight_.assign(vocab_size_, std::vector<double>(model_dim_, 0));
  output_bias_.assign(vocab_size_, 0);
  const double limit = std::sqrt(6.0 / (vocab_size_ + model_dim_));
  std::mt19937 gen(rand_seed_ + 13);
  std::uniform_real_distribution<double> dist(-limit, limit);
  for (auto &row : output_weight_) {
    for (double &value : row) {
      value = dist(gen);
    }
  }

  is_init_ = true;
  return SUCCESS;
}

MiniTransformerLM::RC MiniTransformerLM::Init(int vocab_size, int model_dim,
                                              int head_num,
                                              int feed_forward_dim,
                                              int block_num) {
  Config config;
  config.vocab_size_ = vocab_size;
  config.model_dim_ = model_dim;
  config.head_num_ = head_num;
  config.feed_forward_dim_ = feed_forward_dim;
  config.block_num_ = block_num;
  config.rand_seed_ = rand_seed_;
  config.backbone_type_ = backbone_type_;
  config.use_positional_encoding_ = use_positional_encoding_;
  config.scale_embedding_ = scale_embedding_;
  config.block_learning_rate_scale_ = block_learning_rate_scale_;
  return Init(config);
}

MiniTransformerLM::RC
MiniTransformerLM::Forward(const std::vector<int> &token_ids, Matrix &logits,
                           bool use_causal_mask) {
  Matrix encoded;
  auto rc = EncodeSequence(token_ids, encoded, use_causal_mask);
  if (rc != SUCCESS) {
    return rc;
  }

  logits.assign(encoded.size(), std::vector<double>(vocab_size_, 0));
  for (int pos = 0; pos < encoded.size(); pos++) {
    for (int token_id = 0; token_id < vocab_size_; token_id++) {
      logits[pos][token_id] = output_bias_[token_id];
      for (int dim = 0; dim < model_dim_; dim++) {
        logits[pos][token_id] += output_weight_[token_id][dim] * encoded[pos][dim];
      }
    }
  }
  return SUCCESS;
}

MiniTransformerLM::RC MiniTransformerLM::EncodeSequence(
    const std::vector<int> &token_ids, Matrix &encoded, bool use_causal_mask) {
  if (!is_init_) {
    err_msg_ = "[MiniTransformerLM::EncodeSequence] MiniTransformerLM not init";
    return NOT_INIT;
  }
  if (token_ids.empty()) {
    err_msg_ = "[MiniTransformerLM::EncodeSequence] Invalid token input";
    return INVALID_DATA;
  }

  Matrix hidden;
  if (token_embedding_.Encode(token_ids, hidden) != TokenEmbedding::SUCCESS) {
    err_msg_ = token_embedding_.err_msg();
    return INVALID_DATA;
  }
  if (scale_embedding_) {
    const double scale = std::sqrt(static_cast<double>(model_dim_));
    for (auto &token : hidden) {
      for (double &value : token) {
        value *= scale;
      }
    }
  }
  if (use_positional_encoding_ && !PositionalEncoding::Apply(hidden)) {
    err_msg_ =
        "[MiniTransformerLM::EncodeSequence] Apply positional encoding failed";
    return INVALID_DATA;
  }

  Matrix mask;
  const Matrix *mask_ptr = nullptr;
  if (use_causal_mask) {
    mask = CreateCausalMask(token_ids.size());
    mask_ptr = &mask;
  }
  if (backbone_type_ == BACKBONE_DECODER) {
    (void)mask_ptr;
    if (decoder_.Forward(hidden, encoded) != TransformerDecoder::SUCCESS) {
      err_msg_ = decoder_.err_msg();
      return INVALID_DATA;
    }
  } else {
    if (encoder_.Forward(hidden, encoded, mask_ptr) != TransformerEncoder::SUCCESS) {
      err_msg_ = encoder_.err_msg();
      return INVALID_DATA;
    }
  }
  return SUCCESS;
}

MiniTransformerLM::RC MiniTransformerLM::PredictNextToken(
    const std::vector<int> &token_ids, int &token_id, bool use_causal_mask) {
  std::vector<double> logits;
  auto rc = CalcNextTokenLogits(token_ids, logits, use_causal_mask);
  if (rc != SUCCESS) {
    return rc;
  }

  token_id = 0;
  double best_logit = logits[0];
  for (int i = 1; i < logits.size(); i++) {
    if (logits[i] > best_logit) {
      best_logit = logits[i];
      token_id = i;
    }
  }
  return SUCCESS;
}

MiniTransformerLM::RC MiniTransformerLM::SampleNextToken(
    const std::vector<int> &token_ids, int &token_id,
    const SamplingOption &option, bool use_causal_mask) {
  if (option.temperature_ <= 0 || option.top_p_ <= 0 || option.top_p_ > 1.0 ||
      option.top_k_ < 0) {
    err_msg_ = "[MiniTransformerLM::SampleNextToken] Invalid sampling option";
    return INVALID_DATA;
  }

  std::vector<double> logits;
  auto rc = CalcNextTokenLogits(token_ids, logits, use_causal_mask);
  if (rc != SUCCESS) {
    return rc;
  }

  auto probs = SoftmaxWithTemperature(logits, option.temperature_);
  std::vector<int> order(probs.size(), 0);
  for (int i = 0; i < order.size(); i++) {
    order[i] = i;
  }
  std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
    return probs[lhs] > probs[rhs];
  });

  int keep_count = order.size();
  if (option.top_k_ > 0) {
    keep_count = std::min(keep_count, option.top_k_);
  }
  double cumulative_prob = 0;
  int top_p_keep_count = 0;
  for (int i = 0; i < keep_count; i++) {
    cumulative_prob += probs[order[i]];
    top_p_keep_count++;
    if (cumulative_prob >= option.top_p_) {
      break;
    }
  }
  keep_count = std::max(1, top_p_keep_count);

  std::vector<double> filtered_probs(probs.size(), 0);
  double filtered_sum = 0;
  for (int i = 0; i < keep_count; i++) {
    filtered_probs[order[i]] = probs[order[i]];
    filtered_sum += probs[order[i]];
  }
  for (double &value : filtered_probs) {
    value /= filtered_sum;
  }

  std::mt19937 gen(rand_seed_ + token_ids.size());
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  double rand_value = dist(gen);
  double prob_sum = 0;
  token_id = order[0];
  for (int i = 0; i < filtered_probs.size(); i++) {
    prob_sum += filtered_probs[i];
    if (rand_value <= prob_sum) {
      token_id = i;
      break;
    }
  }
  return SUCCESS;
}

MiniTransformerLM::RC MiniTransformerLM::Generate(
    const std::vector<int> &prompt_token_ids, int generate_num,
    std::vector<int> &generated_token_ids, bool use_causal_mask) {
  if (!is_init_) {
    err_msg_ = "[MiniTransformerLM::Generate] MiniTransformerLM not init";
    return NOT_INIT;
  }
  if (prompt_token_ids.empty() || generate_num < 0) {
    err_msg_ = "[MiniTransformerLM::Generate] Invalid token input";
    return INVALID_DATA;
  }

  generated_token_ids = prompt_token_ids;
  for (int i = 0; i < generate_num; i++) {
    int token_id = 0;
    auto rc = PredictNextToken(generated_token_ids, token_id, use_causal_mask);
    if (rc != SUCCESS) {
      return rc;
    }
    generated_token_ids.push_back(token_id);
  }
  return SUCCESS;
}

MiniTransformerLM::RC MiniTransformerLM::GenerateSample(
    const std::vector<int> &prompt_token_ids, int generate_num,
    std::vector<int> &generated_token_ids, const SamplingOption &option,
    bool use_causal_mask) {
  if (!is_init_) {
    err_msg_ = "[MiniTransformerLM::GenerateSample] MiniTransformerLM not init";
    return NOT_INIT;
  }
  if (prompt_token_ids.empty() || generate_num < 0) {
    err_msg_ = "[MiniTransformerLM::GenerateSample] Invalid token input";
    return INVALID_DATA;
  }

  generated_token_ids = prompt_token_ids;
  for (int i = 0; i < generate_num; i++) {
    int token_id = 0;
    auto rc = SampleNextToken(generated_token_ids, token_id, option,
                              use_causal_mask);
    if (rc != SUCCESS) {
      return rc;
    }
    generated_token_ids.push_back(token_id);
  }
  return SUCCESS;
}

MiniTransformerLM::RC MiniTransformerLM::CalcNextTokenLoss(
    const std::vector<std::vector<int>> &input_samples,
    const std::vector<int> &target_tokens, double &average_loss) {
  if (!is_init_) {
    err_msg_ = "[MiniTransformerLM::CalcNextTokenLoss] MiniTransformerLM not init";
    return NOT_INIT;
  }
  if (input_samples.empty() || input_samples.size() != target_tokens.size()) {
    err_msg_ = "[MiniTransformerLM::CalcNextTokenLoss] Invalid input";
    return INVALID_DATA;
  }

  double loss_sum = 0;
  for (int sample_idx = 0; sample_idx < input_samples.size(); sample_idx++) {
    if (target_tokens[sample_idx] < 0 || target_tokens[sample_idx] >= vocab_size_) {
      err_msg_ = "[MiniTransformerLM::CalcNextTokenLoss] Invalid target token";
      return INVALID_DATA;
    }

    std::vector<double> logits;
    auto rc = CalcNextTokenLogits(input_samples[sample_idx], logits, true);
    if (rc != SUCCESS) {
      return rc;
    }
    auto probs = Softmax(logits);
    loss_sum += -std::log(std::max(probs[target_tokens[sample_idx]], 1e-12));
  }

  average_loss = loss_sum / input_samples.size();
  return SUCCESS;
}

MiniTransformerLM::RC MiniTransformerLM::CalcPerplexity(
    const std::vector<std::vector<int>> &input_samples,
    const std::vector<int> &target_tokens, double &perplexity) {
  double average_loss = 0;
  auto rc = CalcNextTokenLoss(input_samples, target_tokens, average_loss);
  if (rc != SUCCESS) {
    return rc;
  }
  perplexity = std::exp(average_loss);
  return SUCCESS;
}

MiniTransformerLM::RC MiniTransformerLM::InitTrainingHead(int context_size) {
  if (!is_init_) {
    err_msg_ = "[MiniTransformerLM::InitTrainingHead] MiniTransformerLM not init";
    return NOT_INIT;
  }
  if (context_size <= 0) {
    err_msg_ = "[MiniTransformerLM::InitTrainingHead] Invalid context size";
    return INVALID_DATA;
  }

  context_size_ = context_size;
  context_weight_.assign(vocab_size_,
                         std::vector<double>(context_size_ * model_dim_, 0));
  context_bias_.assign(vocab_size_, 0);

  const double limit = std::sqrt(6.0 / (vocab_size_ + context_size_ * model_dim_));
  std::mt19937 gen(rand_seed_ + 29);
  std::uniform_real_distribution<double> dist(-limit, limit);
  for (auto &row : context_weight_) {
    for (double &value : row) {
      value = dist(gen);
    }
  }
  return SUCCESS;
}

std::vector<double>
MiniTransformerLM::BuildContextFeature(const Matrix &encoded) const {
  std::vector<double> feature(context_size_ * model_dim_, 0);
  if (context_size_ == 0 || encoded.empty()) {
    return feature;
  }

  int use_token_size = std::min((int)encoded.size(), context_size_);
  int encoded_start = encoded.size() - use_token_size;
  int feature_start = context_size_ - use_token_size;
  for (int i = 0; i < use_token_size; i++) {
    for (int j = 0; j < model_dim_; j++) {
      feature[(feature_start + i) * model_dim_ + j] =
          encoded[encoded_start + i][j];
    }
  }
  return feature;
}

MiniTransformerLM::RC MiniTransformerLM::ForwardContextHead(
    const std::vector<int> &token_ids, std::vector<double> &logits) {
  if (context_size_ <= 0) {
    err_msg_ = "[MiniTransformerLM::ForwardContextHead] Context head not init";
    return NOT_INIT;
  }

  Matrix encoded;
  auto rc = EncodeSequence(token_ids, encoded, true);
  if (rc != SUCCESS) {
    return rc;
  }
  auto feature = BuildContextFeature(encoded);
  logits.assign(vocab_size_, 0);
  for (int token_id = 0; token_id < vocab_size_; token_id++) {
    logits[token_id] = context_bias_[token_id];
    for (int i = 0; i < feature.size(); i++) {
      logits[token_id] += context_weight_[token_id][i] * feature[i];
    }
  }
  return SUCCESS;
}

MiniTransformerLM::RC MiniTransformerLM::CalcNextTokenLogits(
    const std::vector<int> &token_ids, std::vector<double> &logits,
    bool use_causal_mask) {
  if (has_context_head()) {
    return ForwardContextHead(token_ids, logits);
  }

  Matrix all_logits;
  auto rc = Forward(token_ids, all_logits, use_causal_mask);
  if (rc != SUCCESS) {
    return rc;
  }
  logits = all_logits.back();
  return SUCCESS;
}

MiniTransformerLM::RC MiniTransformerLM::TrainNextToken(
    const std::vector<std::vector<int>> &input_samples,
    const std::vector<int> &target_tokens,
    std::function<void(int epoch_num, double average_loss, bool &early_stop)>
        each_epoch_call,
    int epoch_num, double learning_rate) {
  if (!is_init_) {
    err_msg_ = "[MiniTransformerLM::TrainNextToken] MiniTransformerLM not init";
    return NOT_INIT;
  }
  if (context_size_ <= 0) {
    err_msg_ = "[MiniTransformerLM::TrainNextToken] Context head not init";
    return NOT_INIT;
  }
  if (input_samples.empty() || input_samples.size() != target_tokens.size() ||
      epoch_num <= 0 || learning_rate <= 0) {
    err_msg_ = "[MiniTransformerLM::TrainNextToken] Invalid training input";
    return INVALID_DATA;
  }

  const double embedding_scale =
      scale_embedding_ ? std::sqrt(static_cast<double>(model_dim_)) : 1.0;
  for (int epoch = 0; epoch < epoch_num; epoch++) {
    double loss_sum = 0;
    for (int sample_idx = 0; sample_idx < input_samples.size(); sample_idx++) {
      const auto &sample = input_samples[sample_idx];
      int target_token = target_tokens[sample_idx];
      if (target_token < 0 || target_token >= vocab_size_) {
        err_msg_ = "[MiniTransformerLM::TrainNextToken] Invalid target token";
        return INVALID_DATA;
      }

      Matrix encoded;
      auto rc = EncodeSequence(sample, encoded, true);
      if (rc != SUCCESS) {
        return rc;
      }
      auto feature = BuildContextFeature(encoded);

      std::vector<double> logits(vocab_size_, 0);
      for (int token_id = 0; token_id < vocab_size_; token_id++) {
        logits[token_id] = context_bias_[token_id];
        for (int i = 0; i < feature.size(); i++) {
          logits[token_id] += context_weight_[token_id][i] * feature[i];
        }
      }

      auto probs = Softmax(logits);
      loss_sum += -std::log(std::max(probs[target_token], 1e-12));
      std::vector<double> grad_logits = probs;
      grad_logits[target_token] -= 1.0;

      std::vector<double> grad_feature(feature.size(), 0);
      for (int token_id = 0; token_id < vocab_size_; token_id++) {
        for (int i = 0; i < feature.size(); i++) {
          grad_feature[i] += grad_logits[token_id] * context_weight_[token_id][i];
        }
      }

      for (int token_id = 0; token_id < vocab_size_; token_id++) {
        for (int i = 0; i < feature.size(); i++) {
          context_weight_[token_id][i] -=
              learning_rate * grad_logits[token_id] * feature[i];
        }
        context_bias_[token_id] -= learning_rate * grad_logits[token_id];
      }

      Matrix grad_encoded(encoded.size(), std::vector<double>(model_dim_, 0));
      int use_token_size = std::min((int)sample.size(), context_size_);
      int feature_start = context_size_ - use_token_size;
      int encoded_start = encoded.size() - use_token_size;
      for (int i = 0; i < use_token_size; i++) {
        for (int dim = 0; dim < model_dim_; dim++) {
          grad_encoded[encoded_start + i][dim] =
              grad_feature[(feature_start + i) * model_dim_ + dim];
        }
      }

      Matrix grad_hidden;
      double block_learning_rate =
          block_num_ == 0 ? learning_rate : learning_rate * block_learning_rate_scale_;
      if (backbone_type_ == BACKBONE_DECODER) {
        if (decoder_.Backward(grad_encoded, grad_hidden, block_learning_rate) !=
            TransformerDecoder::SUCCESS) {
          err_msg_ = decoder_.err_msg();
          return INVALID_DATA;
        }
      } else {
        if (encoder_.Backward(grad_encoded, grad_hidden, block_learning_rate) !=
            TransformerEncoder::SUCCESS) {
          err_msg_ = encoder_.err_msg();
          return INVALID_DATA;
        }
      }

      auto &embedding_table = token_embedding_.mutable_embedding_table();
      if (grad_hidden.size() != sample.size()) {
        err_msg_ = "[MiniTransformerLM::TrainNextToken] Invalid backward result";
        return INVALID_DATA;
      }
      for (int i = 0; i < sample.size(); i++) {
        int token_id = sample[i];
        for (int dim = 0; dim < model_dim_; dim++) {
          embedding_table[token_id][dim] -=
              learning_rate * grad_hidden[i][dim] * embedding_scale;
        }
      }
    }

    bool early_stop = false;
    if (each_epoch_call != nullptr) {
      each_epoch_call(epoch, loss_sum / input_samples.size(), early_stop);
    }
    if (early_stop) {
      break;
    }
  }
  return SUCCESS;
}

void MiniTransformerLM::set_random_seed(int seed) { rand_seed_ = seed; }

void MiniTransformerLM::set_backbone_type(BackboneType backbone_type) {
  backbone_type_ = backbone_type;
}

void MiniTransformerLM::set_use_positional_encoding(bool use_positional_encoding) {
  use_positional_encoding_ = use_positional_encoding;
}

void MiniTransformerLM::set_scale_embedding(bool scale_embedding) {
  scale_embedding_ = scale_embedding;
}

void MiniTransformerLM::set_block_learning_rate_scale(double scale) {
  block_learning_rate_scale_ = scale;
}

MiniTransformerLM::RC MiniTransformerLM::set_output_weight(const Matrix &weight) {
  auto rc = ValidateOutputWeight(weight, "set_output_weight");
  if (rc != SUCCESS) {
    return rc;
  }
  output_weight_ = weight;
  return SUCCESS;
}

MiniTransformerLM::RC
MiniTransformerLM::set_output_bias(const std::vector<double> &bias) {
  if (!is_init_) {
    err_msg_ = "[MiniTransformerLM::set_output_bias] MiniTransformerLM not init";
    return NOT_INIT;
  }
  if (bias.size() != vocab_size_) {
    err_msg_ = "[MiniTransformerLM::set_output_bias] Invalid bias size";
    return INVALID_DATA;
  }
  output_bias_ = bias;
  return SUCCESS;
}

MiniTransformerLM::RC MiniTransformerLM::set_context_weight(const Matrix &weight) {
  if (context_size_ == 0) {
    if (weight.empty()) {
      context_weight_.clear();
      return SUCCESS;
    }
    err_msg_ = "[MiniTransformerLM::set_context_weight] Context head not init";
    return NOT_INIT;
  }
  auto rc = ValidateContextWeight(weight, "set_context_weight");
  if (rc != SUCCESS) {
    return rc;
  }
  context_weight_ = weight;
  return SUCCESS;
}

MiniTransformerLM::RC
MiniTransformerLM::set_context_bias(const std::vector<double> &bias) {
  if (context_size_ == 0) {
    if (bias.empty()) {
      context_bias_.clear();
      return SUCCESS;
    }
    err_msg_ = "[MiniTransformerLM::set_context_bias] Context head not init";
    return NOT_INIT;
  }
  if (bias.size() != vocab_size_) {
    err_msg_ = "[MiniTransformerLM::set_context_bias] Invalid bias size";
    return INVALID_DATA;
  }
  context_bias_ = bias;
  return SUCCESS;
}

TokenEmbedding &MiniTransformerLM::token_embedding() { return token_embedding_; }

TransformerEncoder &MiniTransformerLM::encoder() { return encoder_; }

TransformerDecoder &MiniTransformerLM::decoder() { return decoder_; }

const TokenEmbedding &MiniTransformerLM::token_embedding() const {
  return token_embedding_;
}

const TransformerEncoder &MiniTransformerLM::encoder() const { return encoder_; }

const TransformerDecoder &MiniTransformerLM::decoder() const { return decoder_; }

int MiniTransformerLM::vocab_size() const { return vocab_size_; }

int MiniTransformerLM::model_dim() const { return model_dim_; }

int MiniTransformerLM::head_num() const { return head_num_; }

int MiniTransformerLM::feed_forward_dim() const { return feed_forward_dim_; }

int MiniTransformerLM::block_num() const { return block_num_; }

int MiniTransformerLM::rand_seed() const { return rand_seed_; }

int MiniTransformerLM::context_size() const { return context_size_; }

MiniTransformerLM::BackboneType MiniTransformerLM::backbone_type() const {
  return backbone_type_;
}

bool MiniTransformerLM::use_positional_encoding() const {
  return use_positional_encoding_;
}

bool MiniTransformerLM::scale_embedding() const { return scale_embedding_; }

bool MiniTransformerLM::has_context_head() const { return context_size_ > 0; }

double MiniTransformerLM::block_learning_rate_scale() const {
  return block_learning_rate_scale_;
}

MiniTransformerLM::Config MiniTransformerLM::config() const {
  Config config;
  config.vocab_size_ = vocab_size_;
  config.model_dim_ = model_dim_;
  config.head_num_ = head_num_;
  config.feed_forward_dim_ = feed_forward_dim_;
  config.block_num_ = block_num_;
  config.rand_seed_ = rand_seed_;
  config.backbone_type_ = backbone_type_;
  config.use_positional_encoding_ = use_positional_encoding_;
  config.scale_embedding_ = scale_embedding_;
  config.block_learning_rate_scale_ = block_learning_rate_scale_;
  return config;
}

const MiniTransformerLM::Matrix &MiniTransformerLM::output_weight() const {
  return output_weight_;
}

const std::vector<double> &MiniTransformerLM::output_bias() const {
  return output_bias_;
}

const MiniTransformerLM::Matrix &MiniTransformerLM::context_weight() const {
  return context_weight_;
}

const std::vector<double> &MiniTransformerLM::context_bias() const {
  return context_bias_;
}

std::string MiniTransformerLM::err_msg() { return err_msg_; }

MiniTransformerLM::RC
MiniTransformerLM::ValidateOutputWeight(const Matrix &weight,
                                        const char *func_name) {
  if (!is_init_) {
    err_msg_ = std::string("[MiniTransformerLM::") + func_name +
               "] MiniTransformerLM not init";
    return NOT_INIT;
  }
  if (weight.size() != vocab_size_) {
    err_msg_ = std::string("[MiniTransformerLM::") + func_name +
               "] Invalid weight size";
    return INVALID_DATA;
  }
  for (const auto &row : weight) {
    if (row.size() != model_dim_) {
      err_msg_ = std::string("[MiniTransformerLM::") + func_name +
                 "] Invalid weight size";
      return INVALID_DATA;
    }
  }
  return SUCCESS;
}

MiniTransformerLM::RC
MiniTransformerLM::ValidateContextWeight(const Matrix &weight,
                                         const char *func_name) {
  if (context_size_ == 0) {
    err_msg_ = std::string("[MiniTransformerLM::") + func_name +
               "] Context head not init";
    return NOT_INIT;
  }
  if (weight.size() != vocab_size_) {
    err_msg_ = std::string("[MiniTransformerLM::") + func_name +
               "] Invalid weight size";
    return INVALID_DATA;
  }
  for (const auto &row : weight) {
    if (row.size() != context_size_ * model_dim_) {
      err_msg_ = std::string("[MiniTransformerLM::") + func_name +
                 "] Invalid weight size";
      return INVALID_DATA;
    }
  }
  return SUCCESS;
}

} // namespace deeplearning
