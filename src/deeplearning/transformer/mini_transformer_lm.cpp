#include "mini_transformer_lm.h"

#include "positional_encoding.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <thread>

namespace deeplearning {
namespace {

using Matrix = MiniTransformerLM::Matrix;

// Distinct seed offsets so each sub-module draws an independent initialization
// stream from the same base seed. This keeps a run fully reproducible while
// decorrelating the initial parameters of the embedding, encoder, decoder and
// output head.
constexpr int kEncoderSeedOffset = 7;
constexpr int kDecoderSeedOffset = 11;
constexpr int kOutputHeadSeedOffset = 13;

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
  for (int i = 1; i < static_cast<int>(logits.size()); i++) {
    if (logits[i] > max_logit) {
      max_logit = logits[i];
    }
  }

  double sum = 0;
  for (int i = 0; i < static_cast<int>(logits.size()); i++) {
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

std::vector<int> BuildPositionTargets(const std::vector<int> &sample,
                                      int target_token) {
  std::vector<int> position_targets(sample.size());
  for (int i = 0; i + 1 < static_cast<int>(sample.size()); i++) {
    position_targets[i] = sample[i + 1];
  }
  position_targets.back() = target_token;
  return position_targets;
}

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

bool CopyBlockParameters(const TransformerBlock &source,
                         TransformerBlock &target) {
  return target.self_attention().set_query_weight(
             source.self_attention().query_weight()) == SelfAttention::SUCCESS &&
         target.self_attention().set_key_weight(
             source.self_attention().key_weight()) == SelfAttention::SUCCESS &&
         target.self_attention().set_value_weight(
             source.self_attention().value_weight()) == SelfAttention::SUCCESS &&
         target.self_attention().set_output_weight(
             source.self_attention().output_weight()) == SelfAttention::SUCCESS &&
         target.attention_norm().set_scale(source.attention_norm().scale()) ==
             LayerNorm::SUCCESS &&
         target.attention_norm().set_bias(source.attention_norm().bias()) ==
             LayerNorm::SUCCESS &&
         target.set_feed_forward_weight_1(source.feed_forward_weight_1()) ==
             TransformerBlock::SUCCESS &&
         target.set_feed_forward_bias_1(source.feed_forward_bias_1()) ==
             TransformerBlock::SUCCESS &&
         target.set_feed_forward_weight_2(source.feed_forward_weight_2()) ==
             TransformerBlock::SUCCESS &&
         target.set_feed_forward_bias_2(source.feed_forward_bias_2()) ==
             TransformerBlock::SUCCESS &&
         target.feed_forward_norm().set_scale(
             source.feed_forward_norm().scale()) == LayerNorm::SUCCESS &&
         target.feed_forward_norm().set_bias(
             source.feed_forward_norm().bias()) == LayerNorm::SUCCESS &&
         target.set_depth_residual(source.depth_residual_scale(),
                                   source.depth_residual_trainable()) ==
             TransformerBlock::SUCCESS;
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
  max_context_size_ = config.max_context_size_;
  backbone_type_ = config.backbone_type_;
  use_positional_encoding_ = config.use_positional_encoding_;
  scale_embedding_ = config.scale_embedding_;
  block_learning_rate_scale_ = config.block_learning_rate_scale_;
  train_rng_.seed(static_cast<std::mt19937::result_type>(rand_seed_));
  sample_rng_.seed(static_cast<std::mt19937::result_type>(rand_seed_));

  token_embedding_.set_random_seed(rand_seed_);
  if (token_embedding_.Init(vocab_size_, model_dim_) != TokenEmbedding::SUCCESS) {
    err_msg_ = token_embedding_.err_msg();
    return INVALID_DATA;
  }
  encoder_.set_random_seed(rand_seed_ + kEncoderSeedOffset);
  if (encoder_.Init(block_num_, model_dim_, head_num_, feed_forward_dim_) !=
      TransformerEncoder::SUCCESS) {
    err_msg_ = encoder_.err_msg();
    return INVALID_DATA;
  }
  decoder_.set_random_seed(rand_seed_ + kDecoderSeedOffset);
  if (decoder_.Init(block_num_, model_dim_, head_num_, feed_forward_dim_) !=
      TransformerDecoder::SUCCESS) {
    err_msg_ = decoder_.err_msg();
    return INVALID_DATA;
  }

  output_weight_.assign(vocab_size_, std::vector<double>(model_dim_, 0));
  output_bias_.assign(vocab_size_, 0);
  grad_output_weight_.assign(vocab_size_, std::vector<double>(model_dim_, 0));
  grad_output_bias_.assign(vocab_size_, 0);
  grad_embedding_.assign(vocab_size_, std::vector<double>(model_dim_, 0));
  const double limit = std::sqrt(6.0 / (vocab_size_ + model_dim_));
  std::mt19937 gen(rand_seed_ + kOutputHeadSeedOffset);
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
  config.max_context_size_ = max_context_size_;
  config.backbone_type_ = backbone_type_;
  config.use_positional_encoding_ = use_positional_encoding_;
  config.scale_embedding_ = scale_embedding_;
  config.block_learning_rate_scale_ = block_learning_rate_scale_;
  return Init(config);
}

MiniTransformerLM::RC MiniTransformerLM::ExpandVocabulary(
    int new_vocab_size, const std::vector<int> &old_to_new_token_id) {
  if (!is_init_) {
    err_msg_ = "[MiniTransformerLM::ExpandVocabulary] MiniTransformerLM not init";
    return NOT_INIT;
  }
  if (new_vocab_size <= vocab_size_ ||
      old_to_new_token_id.size() != static_cast<size_t>(vocab_size_)) {
    err_msg_ = "[MiniTransformerLM::ExpandVocabulary] Invalid expansion";
    return INVALID_DATA;
  }
  std::vector<bool> used(new_vocab_size, false);
  for (int new_id : old_to_new_token_id) {
    if (new_id < 0 || new_id >= new_vocab_size || used[new_id]) {
      err_msg_ = "[MiniTransformerLM::ExpandVocabulary] Invalid token mapping";
      return INVALID_DATA;
    }
    used[new_id] = true;
  }

  Config new_config = config();
  new_config.vocab_size_ = new_vocab_size;
  MiniTransformerLM expanded;
  if (expanded.Init(new_config) != SUCCESS) {
    err_msg_ = expanded.err_msg();
    return INVALID_DATA;
  }
  expanded.encoder_ = encoder_;
  expanded.decoder_ = decoder_;
  expanded.train_rng_ = train_rng_;
  expanded.sample_rng_ = sample_rng_;
  auto &expanded_embedding =
      expanded.token_embedding_.mutable_embedding_table();
  const auto &old_embedding = token_embedding_.embedding_table();
  for (int old_id = 0; old_id < vocab_size_; old_id++) {
    const int new_id = old_to_new_token_id[old_id];
    expanded_embedding[new_id] = old_embedding[old_id];
    expanded.output_weight_[new_id] = output_weight_[old_id];
    expanded.output_bias_[new_id] = output_bias_[old_id];
  }
  *this = std::move(expanded);
  return SUCCESS;
}

MiniTransformerLM::RC MiniTransformerLM::AppendBlock(AppendBlockMode mode,
                                                       int copy_index) {
  if (!is_init_) {
    err_msg_ = "[MiniTransformerLM::AppendBlock] MiniTransformerLM not init";
    return NOT_INIT;
  }
  if (mode != APPEND_COPY_LAST && mode != APPEND_COPY_INDEX &&
      mode != APPEND_ZERO_RESIDUAL) {
    err_msg_ = "[MiniTransformerLM::AppendBlock] Invalid mode";
    return INVALID_DATA;
  }
  int source_index = copy_index;
  if (mode == APPEND_COPY_LAST) {
    source_index = block_num_ - 1;
  }
  if ((mode == APPEND_COPY_LAST || mode == APPEND_COPY_INDEX) &&
      (source_index < 0 || source_index >= block_num_)) {
    err_msg_ = "[MiniTransformerLM::AppendBlock] Invalid source block";
    return INVALID_DATA;
  }

  TransformerBlock encoder_block;
  encoder_block.set_random_seed(rand_seed_ + kEncoderSeedOffset + block_num_);
  if (encoder_block.Init(model_dim_, head_num_, feed_forward_dim_) !=
      TransformerBlock::SUCCESS) {
    err_msg_ = encoder_block.err_msg();
    return INVALID_DATA;
  }
  TransformerBlock decoder_block;
  decoder_block.set_random_seed(rand_seed_ + kDecoderSeedOffset + block_num_);
  if (decoder_block.Init(model_dim_, head_num_, feed_forward_dim_) !=
      TransformerBlock::SUCCESS) {
    err_msg_ = decoder_block.err_msg();
    return INVALID_DATA;
  }

  if (mode == APPEND_ZERO_RESIDUAL) {
    if (encoder_block.set_depth_residual(0.0, true) !=
            TransformerBlock::SUCCESS ||
        decoder_block.set_depth_residual(0.0, true) !=
            TransformerBlock::SUCCESS) {
      err_msg_ = "[MiniTransformerLM::AppendBlock] Set residual gate failed";
      return INVALID_DATA;
    }
  } else if (!CopyBlockParameters(encoder_.blocks()[source_index],
                                  encoder_block) ||
             !CopyBlockParameters(decoder_.blocks()[source_index],
                                  decoder_block)) {
    err_msg_ = "[MiniTransformerLM::AppendBlock] Copy block failed";
    return INVALID_DATA;
  }

  encoder_.blocks_.push_back(std::move(encoder_block));
  decoder_.blocks_.push_back(std::move(decoder_block));
  block_num_++;
  encoder_.block_num_ = block_num_;
  decoder_.block_num_ = block_num_;
  ClearAccumulatedGradients();
  return SUCCESS;
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
  for (int pos = 0; pos < static_cast<int>(encoded.size()); pos++) {
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
    // The decoder is always causal (it builds its own causal mask internally),
    // so use_causal_mask only affects the encoder backbone and is ignored here.
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
  for (int i = 1; i < static_cast<int>(logits.size()); i++) {
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
  for (int i = 0; i < static_cast<int>(order.size()); i++) {
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

  std::uniform_real_distribution<double> dist(0.0, 1.0);
  double rand_value = dist(sample_rng_);
  double prob_sum = 0;
  token_id = order[0];
  for (int i = 0; i < static_cast<int>(filtered_probs.size()); i++) {
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

  double loss_sum = 0.0;
  long long loss_count = 0;
  for (int sample_idx = 0; sample_idx < static_cast<int>(input_samples.size());
       sample_idx++) {
    const auto &sample = input_samples[sample_idx];
    if (sample.empty()) {
      err_msg_ = "[MiniTransformerLM::CalcNextTokenLoss] Empty sample";
      return INVALID_DATA;
    }
    if (target_tokens[sample_idx] < 0 || target_tokens[sample_idx] >= vocab_size_) {
      err_msg_ = "[MiniTransformerLM::CalcNextTokenLoss] Invalid target token";
      return INVALID_DATA;
    }

    Matrix logits;
    auto rc = Forward(sample, logits, true);
    if (rc != SUCCESS) {
      return rc;
    }
    if (logits.size() != sample.size()) {
      err_msg_ = "[MiniTransformerLM::CalcNextTokenLoss] Invalid logits size";
      return INVALID_DATA;
    }

    const auto position_targets =
        BuildPositionTargets(sample, target_tokens[sample_idx]);
    for (int pos = 0; pos < static_cast<int>(logits.size()); pos++) {
      auto probs = Softmax(logits[pos]);
      loss_sum += -std::log(std::max(probs[position_targets[pos]], 1e-12));
      loss_count++;
    }
  }

  average_loss = loss_count == 0 ? 0.0 : loss_sum / loss_count;
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

MiniTransformerLM::RC MiniTransformerLM::CalcNextTokenLogits(
    const std::vector<int> &token_ids, std::vector<double> &logits,
    bool use_causal_mask) {
  // Only keep the most recent max_context_size_ tokens so the fed positions stay
  // within the range the model was trained on (keeps positional encoding valid).
  const std::vector<int> *window = &token_ids;
  std::vector<int> cropped;
  if (max_context_size_ > 0 &&
      static_cast<int>(token_ids.size()) > max_context_size_) {
    cropped.assign(token_ids.end() - max_context_size_, token_ids.end());
    window = &cropped;
  }

  Matrix all_logits;
  auto rc = Forward(*window, all_logits, use_causal_mask);
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
    int epoch_num, double learning_rate, LRScheduler *lr_scheduler,
    std::function<void(int epoch_num, int finished_sample_num, int sample_num,
                       double average_loss, bool &early_stop)>
        each_sample_call) {
  if (!is_init_) {
    err_msg_ = "[MiniTransformerLM::TrainNextToken] MiniTransformerLM not init";
    return NOT_INIT;
  }
  if (input_samples.empty() || input_samples.size() != target_tokens.size() ||
      epoch_num <= 0 || learning_rate <= 0) {
    err_msg_ = "[MiniTransformerLM::TrainNextToken] Invalid training input";
    return INVALID_DATA;
  }

  ClearAccumulatedGradients();
  std::vector<int> order(input_samples.size(), 0);
  for (int i = 0; i < static_cast<int>(input_samples.size()); i++) {
    order[i] = i;
  }

  for (int epoch = 0; epoch < epoch_num; epoch++) {
    std::shuffle(order.begin(), order.end(), train_rng_);
    const double epoch_learning_rate =
        lr_scheduler != nullptr ? lr_scheduler->GetLR(epoch) : learning_rate;
    const double block_learning_rate =
        block_num_ == 0 ? epoch_learning_rate
                        : epoch_learning_rate * block_learning_rate_scale_;
    double loss_sum = 0;
    long long loss_count = 0;
    bool stop_epoch = false;
    for (int order_pos = 0; order_pos < static_cast<int>(order.size());
         order_pos++) {
      const int sample_idx = order[order_pos];
      const auto &sample = input_samples[sample_idx];
      auto rc = BackwardSample(sample, target_tokens[sample_idx],
                               epoch_learning_rate, block_learning_rate, false,
                               loss_sum, loss_count);
      if (rc != SUCCESS) {
        return rc;
      }
      if (each_sample_call != nullptr) {
        each_sample_call(epoch, order_pos + 1,
                         static_cast<int>(input_samples.size()),
                         loss_count == 0 ? 0.0 : loss_sum / loss_count,
                         stop_epoch);
      }
      if (stop_epoch) {
        break;
      }
    }

    bool early_stop = stop_epoch;
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

MiniTransformerLM::RC MiniTransformerLM::TrainNextTokenBatch(
    const std::vector<std::vector<int>> &input_samples,
    const std::vector<int> &target_tokens, int batch_size,
    std::function<void(int epoch_num, double average_loss, bool &early_stop)>
        each_epoch_call,
    int epoch_num, double learning_rate, LRScheduler *lr_scheduler,
    std::function<void(int epoch_num, int finished_sample_num, int sample_num,
                       double average_loss, bool &early_stop)>
        each_sample_call) {
  if (!is_init_) {
    err_msg_ =
        "[MiniTransformerLM::TrainNextTokenBatch] MiniTransformerLM not init";
    return NOT_INIT;
  }
  if (input_samples.empty() || input_samples.size() != target_tokens.size() ||
      epoch_num <= 0 || learning_rate <= 0 || batch_size <= 0) {
    err_msg_ = "[MiniTransformerLM::TrainNextTokenBatch] Invalid training input";
    return INVALID_DATA;
  }
  if (batch_size == 1) {
    return TrainNextToken(input_samples, target_tokens, each_epoch_call,
                          epoch_num, learning_rate, lr_scheduler,
                          each_sample_call);
  }

  std::vector<int> order(input_samples.size(), 0);
  for (int i = 0; i < static_cast<int>(input_samples.size()); i++) {
    order[i] = i;
  }

  for (int epoch = 0; epoch < epoch_num; epoch++) {
    std::shuffle(order.begin(), order.end(), train_rng_);
    const double epoch_learning_rate =
        lr_scheduler != nullptr ? lr_scheduler->GetLR(epoch) : learning_rate;
    const double block_learning_rate =
        block_num_ == 0 ? epoch_learning_rate
                        : epoch_learning_rate * block_learning_rate_scale_;
    double loss_sum = 0.0;
    long long loss_count = 0;
    bool stop_epoch = false;
    int pending_sample_num = 0;
    ClearAccumulatedGradients();

    for (int order_pos = 0; order_pos < static_cast<int>(order.size());
         order_pos++) {
      const int sample_idx = order[order_pos];
      auto rc = BackwardSample(input_samples[sample_idx],
                               target_tokens[sample_idx], epoch_learning_rate,
                               block_learning_rate, true, loss_sum, loss_count);
      if (rc != SUCCESS) {
        ClearAccumulatedGradients();
        return rc;
      }
      pending_sample_num++;

      const bool should_apply = pending_sample_num == batch_size ||
                                order_pos + 1 == static_cast<int>(order.size());
      if (should_apply) {
        ApplyAccumulatedGradients(epoch_learning_rate, block_learning_rate,
                                  1.0 / pending_sample_num);
        pending_sample_num = 0;
      }

      if (each_sample_call != nullptr) {
        each_sample_call(epoch, order_pos + 1,
                         static_cast<int>(input_samples.size()),
                         loss_count == 0 ? 0.0 : loss_sum / loss_count,
                         stop_epoch);
      }
      if (stop_epoch) {
        break;
      }
    }

    if (pending_sample_num > 0) {
      ApplyAccumulatedGradients(epoch_learning_rate, block_learning_rate,
                                1.0 / pending_sample_num);
    }

    bool early_stop = stop_epoch;
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

MiniTransformerLM::RC MiniTransformerLM::TrainNextTokenBatchParallel(
    const std::vector<std::vector<int>> &input_samples,
    const std::vector<int> &target_tokens, int batch_size, int thread_num,
    std::function<void(int epoch_num, double average_loss, bool &early_stop)>
        each_epoch_call,
    int epoch_num, double learning_rate, LRScheduler *lr_scheduler,
    std::function<void(int epoch_num, int finished_sample_num, int sample_num,
                       double average_loss, bool &early_stop)>
        each_sample_call) {
  if (!is_init_) {
    err_msg_ =
        "[MiniTransformerLM::TrainNextTokenBatchParallel] MiniTransformerLM "
        "not init";
    return NOT_INIT;
  }
  if (input_samples.empty() || input_samples.size() != target_tokens.size() ||
      epoch_num <= 0 || learning_rate <= 0 || batch_size <= 0 ||
      thread_num <= 0) {
    err_msg_ =
        "[MiniTransformerLM::TrainNextTokenBatchParallel] Invalid training "
        "input";
    return INVALID_DATA;
  }
  if (thread_num == 1 || batch_size == 1) {
    return TrainNextTokenBatch(input_samples, target_tokens, batch_size,
                               each_epoch_call, epoch_num, learning_rate,
                               lr_scheduler, each_sample_call);
  }

  std::vector<int> order(input_samples.size(), 0);
  for (int i = 0; i < static_cast<int>(input_samples.size()); i++) {
    order[i] = i;
  }

  for (int epoch = 0; epoch < epoch_num; epoch++) {
    std::shuffle(order.begin(), order.end(), train_rng_);
    const double epoch_learning_rate =
        lr_scheduler != nullptr ? lr_scheduler->GetLR(epoch) : learning_rate;
    const double block_learning_rate =
        block_num_ == 0 ? epoch_learning_rate
                        : epoch_learning_rate * block_learning_rate_scale_;
    double loss_sum = 0.0;
    long long loss_count = 0;
    bool stop_epoch = false;

    for (int batch_begin = 0; batch_begin < static_cast<int>(order.size());
         batch_begin += batch_size) {
      const int batch_end =
          std::min(batch_begin + batch_size, static_cast<int>(order.size()));
      double batch_loss_sum = 0.0;
      long long batch_loss_count = 0;
      auto rc = BackwardBatchParallel(input_samples, target_tokens, order,
                                      batch_begin, batch_end, thread_num,
                                      epoch_learning_rate, block_learning_rate,
                                      batch_loss_sum, batch_loss_count);
      if (rc != SUCCESS) {
        ClearAccumulatedGradients();
        return rc;
      }
      const int batch_sample_num = batch_end - batch_begin;
      ApplyAccumulatedGradients(epoch_learning_rate, block_learning_rate,
                                1.0 / batch_sample_num);
      loss_sum += batch_loss_sum;
      loss_count += batch_loss_count;

      if (each_sample_call != nullptr) {
        each_sample_call(epoch, batch_end, static_cast<int>(input_samples.size()),
                         loss_count == 0 ? 0.0 : loss_sum / loss_count,
                         stop_epoch);
      }
      if (stop_epoch) {
        break;
      }
    }

    bool early_stop = stop_epoch;
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

MiniTransformerLM::RC
MiniTransformerLM::BackwardSample(const std::vector<int> &sample,
                                  int target_token, double learning_rate,
                                  double block_learning_rate,
                                  bool accumulate_gradient, double &loss_sum,
                                  long long &loss_count) {
  if (sample.empty()) {
    err_msg_ = "[MiniTransformerLM::BackwardSample] Empty sample";
    return INVALID_DATA;
  }
  if (target_token < 0 || target_token >= vocab_size_) {
    err_msg_ = "[MiniTransformerLM::BackwardSample] Invalid target token";
    return INVALID_DATA;
  }

  const auto position_targets = BuildPositionTargets(sample, target_token);
  Matrix encoded;
  auto rc = EncodeSequence(sample, encoded, true);
  if (rc != SUCCESS) {
    return rc;
  }

  const int seq_len = static_cast<int>(encoded.size());
  const double inv_seq_len = 1.0 / seq_len;
  const double embedding_scale =
      scale_embedding_ ? std::sqrt(static_cast<double>(model_dim_)) : 1.0;
  Matrix grad_encoded(seq_len, std::vector<double>(model_dim_, 0));
  Matrix grad_output_weight(vocab_size_, std::vector<double>(model_dim_, 0));
  std::vector<double> grad_output_bias(vocab_size_, 0);

  for (int pos = 0; pos < seq_len; pos++) {
    std::vector<double> logits(vocab_size_, 0);
    for (int token_id = 0; token_id < vocab_size_; token_id++) {
      logits[token_id] = output_bias_[token_id];
      for (int dim = 0; dim < model_dim_; dim++) {
        logits[token_id] += output_weight_[token_id][dim] * encoded[pos][dim];
      }
    }

    auto probs = Softmax(logits);
    const int position_target = position_targets[pos];
    loss_sum += -std::log(std::max(probs[position_target], 1e-12));
    loss_count++;

    // Average the per-position gradient so the update matches a mean
    // cross-entropy loss over the sequence regardless of its length.
    std::vector<double> grad_logits = probs;
    grad_logits[position_target] -= 1.0;
    for (double &value : grad_logits) {
      value *= inv_seq_len;
    }

    for (int token_id = 0; token_id < vocab_size_; token_id++) {
      const double grad = grad_logits[token_id];
      grad_output_bias[token_id] += grad;
      for (int dim = 0; dim < model_dim_; dim++) {
        grad_output_weight[token_id][dim] += grad * encoded[pos][dim];
        grad_encoded[pos][dim] += output_weight_[token_id][dim] * grad;
      }
    }
  }

  Matrix grad_hidden;
  if (accumulate_gradient) {
    if (backbone_type_ == BACKBONE_DECODER) {
      if (decoder_.BackwardAccumulate(grad_encoded, grad_hidden) !=
          TransformerDecoder::SUCCESS) {
        err_msg_ = decoder_.err_msg();
        return INVALID_DATA;
      }
    } else {
      if (encoder_.BackwardAccumulate(grad_encoded, grad_hidden) !=
          TransformerEncoder::SUCCESS) {
        err_msg_ = encoder_.err_msg();
        return INVALID_DATA;
      }
    }
  } else if (backbone_type_ == BACKBONE_DECODER) {
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

  if (grad_hidden.size() != sample.size()) {
    err_msg_ = "[MiniTransformerLM::BackwardSample] Invalid backward result";
    return INVALID_DATA;
  }

  Matrix grad_embedding(vocab_size_, std::vector<double>(model_dim_, 0));
  for (int i = 0; i < static_cast<int>(sample.size()); i++) {
    int token_id = sample[i];
    for (int dim = 0; dim < model_dim_; dim++) {
      grad_embedding[token_id][dim] += grad_hidden[i][dim] * embedding_scale;
    }
  }

  if (accumulate_gradient) {
    for (int token_id = 0; token_id < vocab_size_; token_id++) {
      grad_output_bias_[token_id] += grad_output_bias[token_id];
      for (int dim = 0; dim < model_dim_; dim++) {
        grad_output_weight_[token_id][dim] +=
            grad_output_weight[token_id][dim];
        grad_embedding_[token_id][dim] += grad_embedding[token_id][dim];
      }
    }
  } else {
    output_weight_optimizer_.Apply(output_weight_, grad_output_weight,
                                   learning_rate);
    output_bias_optimizer_.Apply(output_bias_, grad_output_bias,
                                 learning_rate);
    auto &embedding_table = token_embedding_.mutable_embedding_table();
    embedding_optimizer_.Apply(embedding_table, grad_embedding,
                               learning_rate);
  }
  return SUCCESS;
}

void MiniTransformerLM::ApplyAccumulatedGradients(double learning_rate,
                                                  double block_learning_rate,
                                                  double gradient_scale) {
  output_weight_optimizer_.Apply(output_weight_, grad_output_weight_,
                                 learning_rate, gradient_scale);
  output_bias_optimizer_.Apply(output_bias_, grad_output_bias_, learning_rate,
                               gradient_scale);
  auto &embedding_table = token_embedding_.mutable_embedding_table();
  embedding_optimizer_.Apply(embedding_table, grad_embedding_, learning_rate,
                             gradient_scale);
  if (backbone_type_ == BACKBONE_DECODER) {
    decoder_.ApplyGradient(block_learning_rate, gradient_scale);
  } else {
    encoder_.ApplyGradient(block_learning_rate, gradient_scale);
  }
  ClearAccumulatedGradients();
}

void MiniTransformerLM::AddAccumulatedGradientsFrom(
    const MiniTransformerLM &source) {
  AddMatrix(grad_output_weight_, source.grad_output_weight_);
  AddVector(grad_output_bias_, source.grad_output_bias_);
  AddMatrix(grad_embedding_, source.grad_embedding_);
  if (backbone_type_ == BACKBONE_DECODER) {
    decoder_.AddGradientsFrom(source.decoder_);
  } else {
    encoder_.AddGradientsFrom(source.encoder_);
  }
}

void MiniTransformerLM::ClearAccumulatedGradients() {
  grad_output_weight_.assign(vocab_size_, std::vector<double>(model_dim_, 0));
  grad_output_bias_.assign(vocab_size_, 0);
  grad_embedding_.assign(vocab_size_, std::vector<double>(model_dim_, 0));
  encoder_.ClearGradients();
  decoder_.ClearGradients();
}

MiniTransformerLM::RC MiniTransformerLM::BackwardBatchParallel(
    const std::vector<std::vector<int>> &input_samples,
    const std::vector<int> &target_tokens, const std::vector<int> &order,
    int begin, int end, int thread_num, double learning_rate,
    double block_learning_rate, double &loss_sum, long long &loss_count) {
  ClearAccumulatedGradients();

  const int batch_sample_num = end - begin;
  const int worker_num = std::min(thread_num, batch_sample_num);
  std::vector<MiniTransformerLM> workers;
  workers.reserve(worker_num);
  for (int i = 0; i < worker_num; i++) {
    workers.push_back(*this);
    workers.back().ClearAccumulatedGradients();
  }

  std::vector<RC> worker_rc(worker_num, SUCCESS);
  std::vector<double> worker_loss_sum(worker_num, 0.0);
  std::vector<long long> worker_loss_count(worker_num, 0);
  std::vector<std::thread> threads;
  threads.reserve(worker_num);

  for (int worker_idx = 0; worker_idx < worker_num; worker_idx++) {
    threads.emplace_back([&, worker_idx]() {
      for (int order_pos = begin + worker_idx; order_pos < end;
           order_pos += worker_num) {
        const int sample_idx = order[order_pos];
        auto rc = workers[worker_idx].BackwardSample(
            input_samples[sample_idx], target_tokens[sample_idx],
            learning_rate, block_learning_rate, true,
            worker_loss_sum[worker_idx], worker_loss_count[worker_idx]);
        if (rc != SUCCESS) {
          worker_rc[worker_idx] = rc;
          return;
        }
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  loss_sum = 0.0;
  loss_count = 0;
  for (int worker_idx = 0; worker_idx < worker_num; worker_idx++) {
    if (worker_rc[worker_idx] != SUCCESS) {
      err_msg_ = workers[worker_idx].err_msg();
      return worker_rc[worker_idx];
    }
    loss_sum += worker_loss_sum[worker_idx];
    loss_count += worker_loss_count[worker_idx];
    AddAccumulatedGradientsFrom(workers[worker_idx]);
  }
  return SUCCESS;
}

void MiniTransformerLM::set_random_seed(int seed) {
  rand_seed_ = seed;
  train_rng_.seed(static_cast<std::mt19937::result_type>(seed));
  sample_rng_.seed(static_cast<std::mt19937::result_type>(seed));
}

void MiniTransformerLM::set_backbone_type(BackboneType backbone_type) {
  backbone_type_ = backbone_type;
}

void MiniTransformerLM::set_use_positional_encoding(bool use_positional_encoding) {
  use_positional_encoding_ = use_positional_encoding;
}

void MiniTransformerLM::set_scale_embedding(bool scale_embedding) {
  scale_embedding_ = scale_embedding;
}

void MiniTransformerLM::set_max_context_size(int max_context_size) {
  max_context_size_ = max_context_size < 0 ? 0 : max_context_size;
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
  if (bias.size() != static_cast<size_t>(vocab_size_)) {
    err_msg_ = "[MiniTransformerLM::set_output_bias] Invalid bias size";
    return INVALID_DATA;
  }
  output_bias_ = bias;
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

MiniTransformerLM::BackboneType MiniTransformerLM::backbone_type() const {
  return backbone_type_;
}

bool MiniTransformerLM::use_positional_encoding() const {
  return use_positional_encoding_;
}

bool MiniTransformerLM::scale_embedding() const { return scale_embedding_; }

int MiniTransformerLM::max_context_size() const { return max_context_size_; }

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
  config.max_context_size_ = max_context_size_;
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

std::string MiniTransformerLM::err_msg() { return err_msg_; }

MiniTransformerLM::RC
MiniTransformerLM::ValidateOutputWeight(const Matrix &weight,
                                        const char *func_name) {
  if (!is_init_) {
    err_msg_ = std::string("[MiniTransformerLM::") + func_name +
               "] MiniTransformerLM not init";
    return NOT_INIT;
  }
  if (weight.size() != static_cast<size_t>(vocab_size_)) {
    err_msg_ = std::string("[MiniTransformerLM::") + func_name +
               "] Invalid weight size";
    return INVALID_DATA;
  }
  for (const auto &row : weight) {
    if (row.size() != static_cast<size_t>(model_dim_)) {
      err_msg_ = std::string("[MiniTransformerLM::") + func_name +
                 "] Invalid weight size";
      return INVALID_DATA;
    }
  }
  return SUCCESS;
}

} // namespace deeplearning
