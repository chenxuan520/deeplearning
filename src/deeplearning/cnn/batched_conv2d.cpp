#include "cnn/batched_conv2d.h"
#include "util/thread_pool.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>

namespace deeplearning {
namespace {

template <class Function>
void ParallelFor(int count, int thread_num, Function function) {
  if (count <= 0) {
    return;
  }
  ThreadPool::Global().Run(count, thread_num, function);
}

} // namespace

BatchedConv2D::RC BatchedConv2D::Init(const Config &config) {
  if (is_init_) {
    err_msg_ = "[BatchedConv2D::Init] already initialized";
    return ALREADY_INIT;
  }
  if (config.input_channels_ <= 0 || config.output_channels_ <= 0 ||
      config.kernel_height_ <= 0 || config.kernel_width_ <= 0 ||
      config.stride_ <= 0 || config.padding_ < 0 ||
      config.thread_num_ <= 0) {
    err_msg_ = "[BatchedConv2D::Init] invalid config";
    return INVALID_DATA;
  }
  config_ = config;
  const int kernel_size = config.input_channels_ * config.kernel_height_ *
                          config.kernel_width_;
  weight_.resize(static_cast<std::size_t>(config.output_channels_) *
                 kernel_size);
  bias_.assign(config.output_channels_, 0.0f);
  grad_weight_.assign(weight_.size(), 0.0f);
  grad_bias_.assign(bias_.size(), 0.0f);

  const float stddev = std::sqrt(2.0f / static_cast<float>(kernel_size));
  std::mt19937 random(static_cast<std::mt19937::result_type>(config.rand_seed_));
  std::normal_distribution<float> distribution(0.0f, stddev);
  for (float &value : weight_) {
    value = distribution(random);
  }
  is_init_ = true;
  return SUCCESS;
}

int BatchedConv2D::output_height(int input_height) const {
  if (!is_init_) {
    return 0;
  }
  return (input_height + 2 * config_.padding_ - config_.kernel_height_) /
             config_.stride_ +
         1;
}

int BatchedConv2D::output_width(int input_width) const {
  if (!is_init_) {
    return 0;
  }
  return (input_width + 2 * config_.padding_ - config_.kernel_width_) /
             config_.stride_ +
         1;
}

bool BatchedConv2D::ValidateInput(const FloatTensor4D &input,
                                  const char *function) {
  if (!is_init_) {
    err_msg_ = std::string("[BatchedConv2D::") + function +
               "] not initialized";
    return false;
  }
  if (input.batch() <= 0 || input.channels() != config_.input_channels_ ||
      input.height() <= 0 || input.width() <= 0 ||
      output_height(input.height()) <= 0 || output_width(input.width()) <= 0) {
    err_msg_ = std::string("[BatchedConv2D::") + function +
               "] invalid input shape";
    return false;
  }
  return true;
}

bool BatchedConv2D::ValidateGradOutput(
    const FloatTensor4D &grad_output) {
  if (!has_cache_) {
    err_msg_ = "[BatchedConv2D::Backward] missing forward cache";
    return false;
  }
  if (grad_output.batch() != last_input_.batch() ||
      grad_output.channels() != config_.output_channels_ ||
      grad_output.height() != last_output_height_ ||
      grad_output.width() != last_output_width_) {
    err_msg_ = "[BatchedConv2D::Backward] invalid gradient shape";
    return false;
  }
  return true;
}

void BatchedConv2D::EnsurePackedWeight() {
  if (packed_weight_valid_) {
    return;
  }
  const int kernel_size = config_.input_channels_ * config_.kernel_height_ *
                          config_.kernel_width_;
  packed_weight_.resize(weight_.size());
  for (int output_channel = 0;
       output_channel < config_.output_channels_; output_channel++) {
    const float *source =
        weight_.data() +
        static_cast<std::size_t>(output_channel) * kernel_size;
    for (int kernel_position = 0; kernel_position < kernel_size;
         kernel_position++) {
      packed_weight_[static_cast<std::size_t>(kernel_position) *
                         config_.output_channels_ +
                     output_channel] = source[kernel_position];
    }
  }
  packed_weight_valid_ = true;
}

void BatchedConv2D::BuildIm2Col(const FloatTensor4D &input,
                                int output_height, int output_width,
                                std::vector<float> &columns) const {
  const int kernel_size = config_.input_channels_ * config_.kernel_height_ *
                          config_.kernel_width_;
  const int row_count = input.batch() * output_height * output_width;
  columns.assign(static_cast<std::size_t>(row_count) * kernel_size, 0.0f);
  ParallelFor(row_count, config_.thread_num_, [&](int begin, int end) {
    for (int row = begin; row < end; row++) {
      int position = row;
      const int output_column = position % output_width;
      position /= output_width;
      const int output_row = position % output_height;
      const int batch = position / output_height;
      float *destination =
          columns.data() + static_cast<std::size_t>(row) * kernel_size;
      int kernel_position = 0;
      for (int input_channel = 0;
           input_channel < config_.input_channels_; input_channel++) {
        for (int kernel_row = 0; kernel_row < config_.kernel_height_;
             kernel_row++) {
          const int input_row =
              output_row * config_.stride_ + kernel_row - config_.padding_;
          for (int kernel_column = 0;
               kernel_column < config_.kernel_width_; kernel_column++) {
            const int input_column =
                output_column * config_.stride_ + kernel_column -
                config_.padding_;
            if (input_row >= 0 && input_row < input.height() &&
                input_column >= 0 && input_column < input.width()) {
              destination[kernel_position] =
                  input(batch, input_channel, input_row, input_column);
            }
            kernel_position++;
          }
        }
      }
    }
  });
}

BatchedConv2D::RC BatchedConv2D::Forward(const FloatTensor4D &input,
                                         FloatTensor4D &output,
                                         bool cache_for_backward) {
  return ForwardInternal(input, output, cache_for_backward, nullptr, nullptr);
}

BatchedConv2D::RC BatchedConv2D::ForwardAffine(
    const FloatTensor4D &input, FloatTensor4D &output,
    const std::vector<float> &channel_scale,
    const std::vector<float> &channel_bias) {
  if (channel_scale.size() !=
          static_cast<std::size_t>(config_.output_channels_) ||
      channel_bias.size() !=
          static_cast<std::size_t>(config_.output_channels_)) {
    err_msg_ = "[BatchedConv2D::ForwardAffine] invalid affine size";
    return INVALID_DATA;
  }
  return ForwardInternal(input, output, false, &channel_scale, &channel_bias);
}

BatchedConv2D::RC BatchedConv2D::ForwardInternal(
    const FloatTensor4D &input, FloatTensor4D &output,
    bool cache_for_backward, const std::vector<float> *channel_scale,
    const std::vector<float> *channel_bias) {
  if (!ValidateInput(input, "Forward")) {
    return is_init_ ? INVALID_DATA : NOT_INIT;
  }
  const int output_height = this->output_height(input.height());
  const int output_width = this->output_width(input.width());
  const int kernel_size = config_.input_channels_ * config_.kernel_height_ *
                          config_.kernel_width_;
  const int row_count = input.batch() * output_height * output_width;
  std::vector<float> columns;
  BuildIm2Col(input, output_height, output_width, columns);
  EnsurePackedWeight();

  output.Resize(input.batch(), config_.output_channels_, output_height,
                output_width);
  std::vector<float> output_matrix(
      static_cast<std::size_t>(row_count) * config_.output_channels_);
  ParallelFor(row_count, config_.thread_num_, [&](int begin, int end) {
    for (int row = begin; row < end; row++) {
      const float *column =
          columns.data() + static_cast<std::size_t>(row) * kernel_size;
      float *destination =
          output_matrix.data() +
          static_cast<std::size_t>(row) * config_.output_channels_;
      if (config_.use_bias_) {
        std::copy(bias_.begin(), bias_.end(), destination);
      } else {
        std::fill(destination,
                  destination + config_.output_channels_, 0.0f);
      }
      for (int kernel_position = 0; kernel_position < kernel_size;
           kernel_position++) {
        const float input_value = column[kernel_position];
        const float *packed_kernel =
            packed_weight_.data() +
            static_cast<std::size_t>(kernel_position) *
                config_.output_channels_;
        for (int output_channel = 0;
             output_channel < config_.output_channels_; output_channel++) {
          destination[output_channel] +=
              input_value * packed_kernel[output_channel];
        }
      }
    }
  });
  ParallelFor(row_count, config_.thread_num_, [&](int begin, int end) {
    for (int row = begin; row < end; row++) {
      int position = row;
      const int output_column = position % output_width;
      position /= output_width;
      const int output_row = position % output_height;
      const int batch = position / output_height;
      const float *source =
          output_matrix.data() +
          static_cast<std::size_t>(row) * config_.output_channels_;
      for (int output_channel = 0;
           output_channel < config_.output_channels_; output_channel++) {
        float value = source[output_channel];
        if (channel_scale != nullptr) {
          value = (*channel_scale)[output_channel] * value +
                  (*channel_bias)[output_channel];
        }
        output(batch, output_channel, output_row, output_column) = value;
      }
    }
  });

  if (cache_for_backward) {
    last_input_.Resize(input.batch(), input.channels(), input.height(),
                       input.width());
    last_columns_ = std::move(columns);
    last_output_height_ = output_height;
    last_output_width_ = output_width;
    has_cache_ = true;
  } else {
    has_cache_ = false;
    last_columns_.clear();
  }
  return SUCCESS;
}

void BatchedConv2D::ColumnsToInput(
    const std::vector<float> &grad_columns,
    FloatTensor4D &grad_input) const {
  const int kernel_size = config_.input_channels_ * config_.kernel_height_ *
                          config_.kernel_width_;
  grad_input.Resize(last_input_.batch(), config_.input_channels_,
                    last_input_.height(), last_input_.width());
  ParallelFor(last_input_.batch(), config_.thread_num_,
              [&](int batch_begin, int batch_end) {
    for (int batch = batch_begin; batch < batch_end; batch++) {
      for (int output_row = 0; output_row < last_output_height_;
           output_row++) {
        for (int output_column = 0;
             output_column < last_output_width_; output_column++) {
          const int row =
              (batch * last_output_height_ + output_row) * last_output_width_ +
              output_column;
          const float *source =
              grad_columns.data() +
              static_cast<std::size_t>(row) * kernel_size;
          int kernel_position = 0;
          for (int input_channel = 0;
               input_channel < config_.input_channels_; input_channel++) {
            for (int kernel_row = 0; kernel_row < config_.kernel_height_;
                 kernel_row++) {
              const int input_row =
                  output_row * config_.stride_ + kernel_row -
                  config_.padding_;
              for (int kernel_column = 0;
                   kernel_column < config_.kernel_width_; kernel_column++) {
                const int input_column =
                    output_column * config_.stride_ + kernel_column -
                    config_.padding_;
                if (input_row >= 0 && input_row < last_input_.height() &&
                    input_column >= 0 &&
                    input_column < last_input_.width()) {
                  grad_input(batch, input_channel, input_row, input_column) +=
                      source[kernel_position];
                }
                kernel_position++;
              }
            }
          }
        }
      }
    }
  });
}

BatchedConv2D::RC BatchedConv2D::Backward(
    const FloatTensor4D &grad_output, FloatTensor4D &grad_input) {
  if (!is_init_) {
    err_msg_ = "[BatchedConv2D::Backward] not initialized";
    return NOT_INIT;
  }
  if (!has_cache_) {
    err_msg_ = "[BatchedConv2D::Backward] missing forward cache";
    return MISSING_CACHE;
  }
  if (!ValidateGradOutput(grad_output)) {
    return INVALID_DATA;
  }
  const int kernel_size = config_.input_channels_ * config_.kernel_height_ *
                          config_.kernel_width_;
  const int row_count =
      grad_output.batch() * last_output_height_ * last_output_width_;

  ParallelFor(config_.output_channels_, config_.thread_num_,
              [&](int begin, int end) {
    for (int output_channel = begin; output_channel < end; output_channel++) {
      float *weight_gradient =
          grad_weight_.data() +
          static_cast<std::size_t>(output_channel) * kernel_size;
      float bias_gradient = 0.0f;
      for (int row = 0; row < row_count; row++) {
        int position = row;
        const int output_column = position % last_output_width_;
        position /= last_output_width_;
        const int output_row = position % last_output_height_;
        const int batch = position / last_output_height_;
        const float gradient =
            grad_output(batch, output_channel, output_row, output_column);
        bias_gradient += gradient;
        const float *column =
            last_columns_.data() +
            static_cast<std::size_t>(row) * kernel_size;
        for (int index = 0; index < kernel_size; index++) {
          weight_gradient[index] += gradient * column[index];
        }
      }
      if (config_.use_bias_) {
        grad_bias_[output_channel] += bias_gradient;
      }
    }
  });

  std::vector<float> grad_columns(
      static_cast<std::size_t>(row_count) * kernel_size, 0.0f);
  ParallelFor(row_count, config_.thread_num_, [&](int begin, int end) {
    for (int row = begin; row < end; row++) {
      int position = row;
      const int output_column = position % last_output_width_;
      position /= last_output_width_;
      const int output_row = position % last_output_height_;
      const int batch = position / last_output_height_;
      float *column_gradient =
          grad_columns.data() + static_cast<std::size_t>(row) * kernel_size;
      for (int output_channel = 0;
           output_channel < config_.output_channels_; output_channel++) {
        const float gradient =
            grad_output(batch, output_channel, output_row, output_column);
        const float *kernel =
            weight_.data() +
            static_cast<std::size_t>(output_channel) * kernel_size;
        for (int index = 0; index < kernel_size; index++) {
          column_gradient[index] += gradient * kernel[index];
        }
      }
    }
  });
  ColumnsToInput(grad_columns, grad_input);
  // Optimizers update weight_ after Backward through the model's contiguous
  // parameter views. Force the next forward pass to rebuild the transposed
  // inference layout from those updated weights.
  packed_weight_valid_ = false;
  return SUCCESS;
}

void BatchedConv2D::ZeroGrad() {
  std::fill(grad_weight_.begin(), grad_weight_.end(), 0.0f);
  std::fill(grad_bias_.begin(), grad_bias_.end(), 0.0f);
}

BatchedConv2D::RC BatchedConv2D::set_weight(
    const std::vector<float> &weight) {
  if (!is_init_) {
    err_msg_ = "[BatchedConv2D::set_weight] not initialized";
    return NOT_INIT;
  }
  if (weight.size() != weight_.size()) {
    err_msg_ = "[BatchedConv2D::set_weight] invalid weight size";
    return INVALID_DATA;
  }
  weight_ = weight;
  packed_weight_valid_ = false;
  return SUCCESS;
}

BatchedConv2D::RC BatchedConv2D::set_bias(
    const std::vector<float> &bias) {
  if (!is_init_) {
    err_msg_ = "[BatchedConv2D::set_bias] not initialized";
    return NOT_INIT;
  }
  if (bias.size() != bias_.size()) {
    err_msg_ = "[BatchedConv2D::set_bias] invalid bias size";
    return INVALID_DATA;
  }
  bias_ = bias;
  return SUCCESS;
}

void BatchedConv2D::set_thread_num(int thread_num) {
  config_.thread_num_ = std::max(1, thread_num);
}

} // namespace deeplearning
