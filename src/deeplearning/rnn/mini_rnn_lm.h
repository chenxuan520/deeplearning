#pragma once

#include "lr_scheduler/lr_scheduler_base.h"
#include "rnn/simple_rnn.h"
#include "transformer/tensor_optimizer.h"

#include <functional>
#include <string>
#include <vector>

namespace deeplearning {

class MiniRNNLM {
public:
  using Matrix = std::vector<std::vector<double>>;

  struct Config {
    int vocab_size_ = 0;
    int hidden_dim_ = 0;
    int rand_seed_ = 0;
    double gradient_clip_norm_ = 0.0;
  };

  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
  };

public:
  RC Init(const Config &config);
  RC Init(int vocab_size, int hidden_dim);
  RC Forward(const std::vector<int> &token_ids, Matrix &logits);
  RC PredictNextToken(const std::vector<int> &token_ids, int &token_id);
  RC Generate(const std::vector<int> &prompt_token_ids, int generate_num,
              std::vector<int> &generated_token_ids);
  RC CalcNextTokenLoss(const std::vector<std::vector<int>> &input_samples,
                       const std::vector<int> &target_tokens,
                       double &average_loss);
  RC CalcPerplexity(const std::vector<std::vector<int>> &input_samples,
                    const std::vector<int> &target_tokens, double &perplexity);
  RC TrainNextToken(
      const std::vector<std::vector<int>> &input_samples,
      const std::vector<int> &target_tokens,
      std::function<void(int epoch_num, double average_loss, bool &early_stop)>
          each_epoch_call = nullptr,
      int epoch_num = 1, double learning_rate = 0.1,
      LRScheduler *lr_scheduler = nullptr);

  void set_random_seed(int seed);
  void set_gradient_clip_norm(double max_norm);
  RC set_output_weight(const Matrix &weight);
  RC set_output_bias(const std::vector<double> &bias);

  std::string err_msg();
  int vocab_size() const;
  int hidden_dim() const;
  double gradient_clip_norm() const;
  const Matrix &output_weight() const;
  const std::vector<double> &output_bias() const;
  SimpleRNN &rnn();
  const SimpleRNN &rnn() const;

private:
  RC EncodeTokenIds(const std::vector<int> &token_ids, Matrix &input_sequence);
  RC ValidateOutputWeight(const Matrix &weight, const char *func_name);

private:
  int vocab_size_ = 0;
  int hidden_dim_ = 0;
  int rand_seed_ = 0;
  double gradient_clip_norm_ = 0.0;
  Matrix output_weight_;
  std::vector<double> output_bias_;
  SimpleRNN rnn_;
  TensorOptimizer output_weight_optimizer_;
  TensorOptimizer output_bias_optimizer_;
  std::string err_msg_;
  bool is_init_ = false;
};

} // namespace deeplearning
