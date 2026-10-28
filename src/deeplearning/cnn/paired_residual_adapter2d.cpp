#include "cnn/paired_residual_adapter2d.h"

#include "util/thread_pool.h"

#include <algorithm>

#ifdef DEEPLEARNING_EMBED_PAIRED_RESIDUAL_ADAPTER2D

namespace deeplearning {
namespace {

bool CopyWeights(const BatchedConv2D &source, BatchedConv2D &target) {
  return target.set_weight(source.weight()) == BatchedConv2D::SUCCESS;
}

} // namespace

PairedResidualAdapter2D::RC PairedResidualAdapter2D::Init(
    const Config &config) {
  if (is_init_) {
    err_msg_ = "[PairedResidualAdapter2D::Init] already initialized";
    return ALREADY_INIT;
  }
  if (config.channels_ <= 0 || config.bottleneck_channels_ <= 0 ||
      config.height_ <= 0 || config.width_ <= 0 ||
      config.thread_num_ <= 0) {
    err_msg_ = "[PairedResidualAdapter2D::Init] invalid config";
    return INVALID_DATA;
  }
  config_ = config;
  BatchedConv2D::Config down;
  down.input_channels_ = config.channels_;
  down.output_channels_ = config.bottleneck_channels_;
  down.kernel_height_ = 1;
  down.kernel_width_ = 1;
  down.padding_ = 0;
  down.use_bias_ = false;
  down.thread_num_ = config.thread_num_;
  down.rand_seed_ = config.rand_seed_;
  BatchedConv2D::Config up = down;
  up.input_channels_ = config.bottleneck_channels_;
  up.output_channels_ = config.channels_;
  up.rand_seed_ = config.rand_seed_ + 1;
  if (branch_a_down_.Init(down) != BatchedConv2D::SUCCESS ||
      branch_a_up_.Init(up) != BatchedConv2D::SUCCESS ||
      branch_b_down_.Init(down) != BatchedConv2D::SUCCESS ||
      branch_b_up_.Init(up) != BatchedConv2D::SUCCESS ||
      !CopyWeights(branch_a_down_, branch_b_down_) ||
      !CopyWeights(branch_a_up_, branch_b_up_)) {
    err_msg_ = "[PairedResidualAdapter2D::Init] layer initialization failed";
    return INVALID_DATA;
  }
  is_init_ = true;
  return SUCCESS;
}

bool PairedResidualAdapter2D::ValidateInput(
    const FloatTensor4D &input, const char *function) {
  if (!is_init_) {
    err_msg_ = std::string("[PairedResidualAdapter2D::") + function +
               "] not initialized";
    return false;
  }
  if (input.batch() <= 0 || input.channels() != config_.channels_ ||
      input.height() != config_.height_ ||
      input.width() != config_.width_) {
    err_msg_ = std::string("[PairedResidualAdapter2D::") + function +
               "] invalid input shape";
    return false;
  }
  return true;
}

PairedResidualAdapter2D::RC PairedResidualAdapter2D::Forward(
    const FloatTensor4D &input, FloatTensor4D &output, bool training) {
  if (!ValidateInput(input, "Forward")) {
    return is_init_ ? INVALID_DATA : NOT_INIT;
  }
  if (!training) return ForwardInference(input, output);
  FloatTensor4D branch_a_hidden;
  FloatTensor4D branch_b_hidden;
  if (branch_a_down_.Forward(input, branch_a_hidden, training) !=
          BatchedConv2D::SUCCESS ||
      branch_b_down_.Forward(input, branch_b_hidden, training) !=
          BatchedConv2D::SUCCESS) {
    err_msg_ = "[PairedResidualAdapter2D::Forward] down projection failed";
    return INVALID_DATA;
  }
  for (float &value : branch_a_hidden.values()) {
    value = std::max(0.0f, value);
  }
  for (float &value : branch_b_hidden.values()) {
    value = std::max(0.0f, value);
  }
  FloatTensor4D branch_a_output;
  FloatTensor4D branch_b_output;
  if (branch_a_up_.Forward(branch_a_hidden, branch_a_output, training) !=
          BatchedConv2D::SUCCESS ||
      branch_b_up_.Forward(branch_b_hidden, branch_b_output, training) !=
          BatchedConv2D::SUCCESS) {
    err_msg_ = "[PairedResidualAdapter2D::Forward] up projection failed";
    return INVALID_DATA;
  }
  output = input;
  for (std::size_t index = 0; index < output.size(); index++) {
    output.values()[index] +=
        branch_a_output.values()[index] - branch_b_output.values()[index];
  }
  if (training) {
    branch_a_relu_cache_ = std::move(branch_a_hidden);
    branch_b_relu_cache_ = std::move(branch_b_hidden);
    has_cache_ = true;
  } else {
    has_cache_ = false;
  }
  return SUCCESS;
}

PairedResidualAdapter2D::RC PairedResidualAdapter2D::ForwardInference(
    const FloatTensor4D &input, FloatTensor4D &output) {
  const int batch = input.batch();
  const int channels = config_.channels_;
  const int bottleneck = config_.bottleneck_channels_;
  const int spatial = config_.height_ * config_.width_;
  output = input;
  const auto &down_a = branch_a_down_.weight();
  const auto &up_a = branch_a_up_.weight();
  const auto &down_b = branch_b_down_.weight();
  const auto &up_b = branch_b_up_.weight();
  const int positions = batch * spatial;
  ThreadPool::Global().Run(
      positions, config_.thread_num_,
      [&](int begin, int end) {
        std::vector<float> hidden_a(bottleneck);
        std::vector<float> hidden_b(bottleneck);
        for (int position = begin; position < end; position++) {
          const int sample = position / spatial;
          const int cell = position % spatial;
          for (int hidden = 0; hidden < bottleneck; hidden++) {
            float sum_a = 0.0f;
            float sum_b = 0.0f;
            const float *row_a =
                down_a.data() + static_cast<std::size_t>(hidden) * channels;
            const float *row_b =
                down_b.data() + static_cast<std::size_t>(hidden) * channels;
            for (int channel = 0; channel < channels; channel++) {
              const float value = input.values()[
                  (static_cast<std::size_t>(sample) * channels + channel) *
                      spatial +
                  cell];
              sum_a += row_a[channel] * value;
              sum_b += row_b[channel] * value;
            }
            hidden_a[hidden] = std::max(0.0f, sum_a);
            hidden_b[hidden] = std::max(0.0f, sum_b);
          }
          for (int channel = 0; channel < channels; channel++) {
            float sum_a = 0.0f;
            float sum_b = 0.0f;
            const float *row_a =
                up_a.data() + static_cast<std::size_t>(channel) * bottleneck;
            const float *row_b =
                up_b.data() + static_cast<std::size_t>(channel) * bottleneck;
            for (int hidden = 0; hidden < bottleneck; hidden++) {
              sum_a += row_a[hidden] * hidden_a[hidden];
              sum_b += row_b[hidden] * hidden_b[hidden];
            }
            const std::size_t offset =
                (static_cast<std::size_t>(sample) * channels + channel) *
                    spatial +
                cell;
            output.values()[offset] += sum_a - sum_b;
          }
        }
      });
  has_cache_ = false;
  return SUCCESS;
}

PairedResidualAdapter2D::RC PairedResidualAdapter2D::Backward(
    const FloatTensor4D &grad_output, FloatTensor4D &grad_input) {
  if (!is_init_) {
    err_msg_ = "[PairedResidualAdapter2D::Backward] not initialized";
    return NOT_INIT;
  }
  if (!has_cache_) {
    err_msg_ = "[PairedResidualAdapter2D::Backward] missing training cache";
    return MISSING_CACHE;
  }
  if (grad_output.batch() <= 0 ||
      grad_output.channels() != config_.channels_ ||
      grad_output.height() != config_.height_ ||
      grad_output.width() != config_.width_) {
    err_msg_ = "[PairedResidualAdapter2D::Backward] invalid gradient shape";
    return INVALID_DATA;
  }
  FloatTensor4D grad_branch_a_hidden;
  if (branch_a_up_.Backward(grad_output, grad_branch_a_hidden) !=
      BatchedConv2D::SUCCESS) {
    err_msg_ = branch_a_up_.err_msg();
    return INVALID_DATA;
  }
  FloatTensor4D negated = grad_output;
  for (float &value : negated.values()) value = -value;
  FloatTensor4D grad_branch_b_hidden;
  if (branch_b_up_.Backward(negated, grad_branch_b_hidden) !=
      BatchedConv2D::SUCCESS) {
    err_msg_ = branch_b_up_.err_msg();
    return INVALID_DATA;
  }
  for (std::size_t index = 0; index < grad_branch_a_hidden.size(); index++) {
    if (branch_a_relu_cache_.values()[index] <= 0.0f) {
      grad_branch_a_hidden.values()[index] = 0.0f;
    }
    if (branch_b_relu_cache_.values()[index] <= 0.0f) {
      grad_branch_b_hidden.values()[index] = 0.0f;
    }
  }
  FloatTensor4D grad_branch_a_input;
  FloatTensor4D grad_branch_b_input;
  if (branch_a_down_.Backward(
          grad_branch_a_hidden, grad_branch_a_input) !=
          BatchedConv2D::SUCCESS ||
      branch_b_down_.Backward(
          grad_branch_b_hidden, grad_branch_b_input) !=
          BatchedConv2D::SUCCESS) {
    err_msg_ = "[PairedResidualAdapter2D::Backward] down projection failed";
    return INVALID_DATA;
  }
  grad_input = grad_output;
  for (std::size_t index = 0; index < grad_input.size(); index++) {
    grad_input.values()[index] +=
        grad_branch_a_input.values()[index] +
        grad_branch_b_input.values()[index];
  }
  return SUCCESS;
}

void PairedResidualAdapter2D::ZeroGrad() {
  branch_a_down_.ZeroGrad();
  branch_a_up_.ZeroGrad();
  branch_b_down_.ZeroGrad();
  branch_b_up_.ZeroGrad();
}

} // namespace deeplearning

#endif
