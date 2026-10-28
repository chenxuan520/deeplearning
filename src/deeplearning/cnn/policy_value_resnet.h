#pragma once

#include "cnn/batch_norm2d.h"
#include "cnn/batched_conv2d.h"
#include "cnn/float_linear.h"
#include "cnn/paired_residual_adapter2d.h"
#include "cnn/residual_block2d.h"
#include "cnn/single_residual_adapter2d.h"
#include "cnn/wdl_value.h"

#include <string>
#include <vector>

namespace deeplearning {

class PolicyValueResNet {
public:
  struct TrainableParameter {
    std::string name_;
    std::vector<float> *value_ = nullptr;
    const std::vector<float> *gradient_ = nullptr;
    bool apply_weight_decay_ = true;
  };

  struct Config {
    int input_channels_ = 14;
    int board_height_ = 10;
    int board_width_ = 9;
    int trunk_channels_ = 96;
    int residual_block_num_ = 6;
    int policy_channels_ = 4;
    int policy_size_ = 2086;
    int value_channels_ = 2;
    int value_hidden_dim_ = 256;
    // Zero preserves the legacy architecture and checkpoint format.
    int adapter_bottleneck_channels_ = 0;
    int adapter_rand_seed_ = 0;
    int single_adapter_bottleneck_channels_ = 0;
    int single_adapter_rand_seed_ = 0;
    // One output preserves the legacy scalar tanh head. Three classes use
    // two raw outputs: an EV logit and a draw-propensity logit.
    int value_class_num_ = 1;
    int thread_num_ = 1;
    int rand_seed_ = 0;
    float batch_norm_epsilon_ = 1e-5f;
    float batch_norm_momentum_ = 0.1f;
  };

  struct Output {
    std::vector<float> policy_logits_;
    std::vector<float> values_;
    std::vector<float> value_logits_;
    std::vector<float> value_probabilities_;
    int batch_ = 0;
  };

  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
    MISSING_CACHE,
  };

  RC Init(const Config &config);
  RC InitWdlFromScalar(const PolicyValueResNet &source,
                       float initial_draw_logit = -16.0f);
  RC Forward(const FloatTensor4D &input, Output &output, bool training,
             bool use_running_statistics = false);
  RC Backward(const std::vector<float> &grad_policy_logits,
              const std::vector<float> &grad_values,
              FloatTensor4D &grad_input);
  // WDL cross-entropy supplies gradients for win/draw/loss probabilities.
  RC BackwardValueProbabilities(
      const std::vector<float> &grad_policy_logits,
      const std::vector<float> &grad_value_probabilities,
      FloatTensor4D &grad_input);
  RC BackwardWdlOutputOnly(
      const std::vector<float> &grad_value_probabilities);
  void ZeroGrad();

  const Config &config() const { return config_; }
  BatchedConv2D &stem_conv() { return stem_conv_; }
  BatchNorm2D &stem_norm() { return stem_norm_; }
  std::vector<ResidualBlock2D> &blocks() { return blocks_; }
  BatchedConv2D &policy_conv() { return policy_conv_; }
  BatchNorm2D &policy_norm() { return policy_norm_; }
  FloatLinear &policy_linear() { return policy_linear_; }
  BatchedConv2D &value_conv() { return value_conv_; }
  BatchNorm2D &value_norm() { return value_norm_; }
  FloatLinear &value_hidden() { return value_hidden_; }
  FloatLinear &value_output() { return value_output_; }
  PairedResidualAdapter2D &adapter() { return adapter_; }
  const PairedResidualAdapter2D &adapter() const { return adapter_; }
  SingleResidualAdapter2D &single_adapter() { return single_adapter_; }
  const SingleResidualAdapter2D &single_adapter() const {
    return single_adapter_;
  }
  bool has_adapter() const {
    return config_.adapter_bottleneck_channels_ > 0;
  }
  bool has_single_adapter() const {
    return config_.single_adapter_bottleneck_channels_ > 0;
  }
  std::vector<TrainableParameter> TrainableParameters();
  RC Save(const std::string &file) const;
  RC Load(const std::string &file);
  std::size_t parameter_count() const;
  std::string err_msg() const { return err_msg_; }

private:
  bool ValidateInput(const FloatTensor4D &input, const char *function);
  static void ApplyRelu(FloatTensor4D &tensor);
  static std::vector<float> FlattenNCHW(const FloatTensor4D &tensor);
  static FloatTensor4D UnflattenNCHW(const std::vector<float> &values,
                                     int batch, int channels, int height,
                                     int width);
  RC BackwardImpl(const std::vector<float> &grad_policy_logits,
                  const std::vector<float> &grad_raw_value,
                  FloatTensor4D &grad_input);
  RC WdlProbabilityGradientToRaw(
      const std::vector<float> &grad_value_probabilities,
      std::vector<float> &grad_raw_value);

  Config config_;
  BatchedConv2D stem_conv_;
  BatchNorm2D stem_norm_;
  std::vector<ResidualBlock2D> blocks_;
  PairedResidualAdapter2D adapter_;
  SingleResidualAdapter2D single_adapter_;

  BatchedConv2D policy_conv_;
  BatchNorm2D policy_norm_;
  FloatLinear policy_linear_;

  BatchedConv2D value_conv_;
  BatchNorm2D value_norm_;
  FloatLinear value_hidden_;
  FloatLinear value_output_;

  FloatTensor4D stem_relu_cache_;
  FloatTensor4D policy_relu_cache_;
  FloatTensor4D value_relu_cache_;
  std::vector<float> value_hidden_relu_cache_;
  std::vector<float> value_cache_;
  std::vector<float> value_raw_cache_;
  std::vector<float> value_probability_cache_;
  int cached_batch_ = 0;
  bool is_init_ = false;
  bool has_cache_ = false;
  std::string err_msg_;
};

} // namespace deeplearning
