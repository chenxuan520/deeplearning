#pragma once

#include "lr_scheduler/lr_scheduler_base.h"
#include "tensor_optimizer.h"
#include "token_embedding.h"
#include "transformer_decoder.h"
#include "transformer_encoder.h"

#include <functional>
#include <random>
#include <string>
#include <vector>

namespace deeplearning {

class MiniTransformerLM {
public:
  using Matrix = std::vector<std::vector<double>>;

  enum BackboneType {
    BACKBONE_ENCODER,
    BACKBONE_DECODER,
  };

  struct SamplingOption {
    double temperature_ = 1.0;
    int top_k_ = 0;
    double top_p_ = 1.0;
  };

  struct Config {
    int vocab_size_ = 0;
    int model_dim_ = 0;
    int head_num_ = 1;
    int feed_forward_dim_ = 0;
    int block_num_ = 0;
    int rand_seed_ = 0;
    // Maximum number of most-recent tokens fed to the model when predicting the
    // next token during generation. 0 means "no limit". Keeping this aligned
    // with the training window keeps positional encodings in-distribution.
    int max_context_size_ = 0;
    BackboneType backbone_type_ = BACKBONE_ENCODER;
    bool use_positional_encoding_ = true;
    bool scale_embedding_ = true;
    double block_learning_rate_scale_ = 0.1;
  };

  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
  };

public:
  RC Init(const Config &config);
  RC Init(int vocab_size, int model_dim, int head_num, int feed_forward_dim,
          int block_num);
  // use_causal_mask only applies to the encoder backbone; the decoder backbone
  // is always causal and ignores this flag.
  RC Forward(const std::vector<int> &token_ids, Matrix &logits,
             bool use_causal_mask = true);
  RC PredictNextToken(const std::vector<int> &token_ids, int &token_id,
                      bool use_causal_mask = true);
  RC SampleNextToken(const std::vector<int> &token_ids, int &token_id,
                     const SamplingOption &option,
                     bool use_causal_mask = true);
  RC Generate(const std::vector<int> &prompt_token_ids, int generate_num,
              std::vector<int> &generated_token_ids,
              bool use_causal_mask = true);
  RC GenerateSample(const std::vector<int> &prompt_token_ids, int generate_num,
                    std::vector<int> &generated_token_ids,
                    const SamplingOption &option,
                    bool use_causal_mask = true);
  RC CalcNextTokenLoss(const std::vector<std::vector<int>> &input_samples,
                       const std::vector<int> &target_tokens,
                       double &average_loss);
  RC CalcPerplexity(const std::vector<std::vector<int>> &input_samples,
                    const std::vector<int> &target_tokens,
                    double &perplexity);
  // Trains the model with per-position next-token cross-entropy using Adam
  // (shared with the MLP path). When lr_scheduler != nullptr the base learning
  // rate is taken from lr_scheduler->GetLR(epoch) at the start of each epoch.
  RC TrainNextToken(
      const std::vector<std::vector<int>> &input_samples,
      const std::vector<int> &target_tokens,
      std::function<void(int epoch_num, double average_loss, bool &early_stop)>
          each_epoch_call = nullptr,
      int epoch_num = 1, double learning_rate = 0.1,
      LRScheduler *lr_scheduler = nullptr);

  void set_random_seed(int seed);
  void set_backbone_type(BackboneType backbone_type);
  void set_use_positional_encoding(bool use_positional_encoding);
  void set_scale_embedding(bool scale_embedding);
  void set_max_context_size(int max_context_size);
  void set_block_learning_rate_scale(double scale);
  RC set_output_weight(const Matrix &weight);
  RC set_output_bias(const std::vector<double> &bias);

  TokenEmbedding &token_embedding();
  TransformerEncoder &encoder();
  TransformerDecoder &decoder();
  const TokenEmbedding &token_embedding() const;
  const TransformerEncoder &encoder() const;
  const TransformerDecoder &decoder() const;
  int vocab_size() const;
  int model_dim() const;
  int head_num() const;
  int feed_forward_dim() const;
  int block_num() const;
  int rand_seed() const;
  BackboneType backbone_type() const;
  bool use_positional_encoding() const;
  bool scale_embedding() const;
  int max_context_size() const;
  double block_learning_rate_scale() const;
  Config config() const;
  const Matrix &output_weight() const;
  const std::vector<double> &output_bias() const;
  std::string err_msg();

private:
  RC EncodeSequence(const std::vector<int> &token_ids, Matrix &encoded,
                    bool use_causal_mask = true);
  RC CalcNextTokenLogits(const std::vector<int> &token_ids,
                         std::vector<double> &logits,
                         bool use_causal_mask = true);
  RC ValidateOutputWeight(const Matrix &weight, const char *func_name);

private:
  int vocab_size_ = 0;
  int model_dim_ = 0;
  int head_num_ = 0;
  int feed_forward_dim_ = 0;
  int block_num_ = 0;
  int rand_seed_ = 0;
  int max_context_size_ = 0;
  BackboneType backbone_type_ = BACKBONE_ENCODER;
  bool use_positional_encoding_ = true;
  bool scale_embedding_ = true;
  // Extra learning-rate multiplier for the Transformer blocks. Adam already
  // normalizes per-parameter step sizes, so this defaults to 1.0 (off); it is
  // kept only as an optional knob and is still persisted for compatibility.
  double block_learning_rate_scale_ = 1.0;
  Matrix output_weight_;
  std::vector<double> output_bias_;
  TokenEmbedding token_embedding_;
  TransformerEncoder encoder_;
  TransformerDecoder decoder_;
  TensorOptimizer output_weight_optimizer_;
  TensorOptimizer output_bias_optimizer_;
  TensorOptimizer embedding_optimizer_;
  std::mt19937 sample_rng_;
  std::string err_msg_;
  bool is_init_ = false;
};

} // namespace deeplearning
