#pragma once

#include "cnn/float_tensor.h"

#include <string>
#include <vector>

namespace deeplearning {

class BatchedConv2D {
public:
  struct Config {
    int input_channels_ = 0;
    int output_channels_ = 0;
    int kernel_height_ = 3;
    int kernel_width_ = 3;
    int stride_ = 1;
    int padding_ = 0;
    bool use_bias_ = true;
    int thread_num_ = 1;
    int rand_seed_ = 0;
  };

  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
    MISSING_CACHE,
  };

  RC Init(const Config &config);

  // cache_for_backward=false is the intended inference path.
  RC Forward(const FloatTensor4D &input, FloatTensor4D &output,
             bool cache_for_backward = false);
  // Inference-only fused per-output-channel affine transform:
  // output = channel_scale * convolution(input) + channel_bias.
  RC ForwardAffine(const FloatTensor4D &input, FloatTensor4D &output,
                   const std::vector<float> &channel_scale,
                   const std::vector<float> &channel_bias);

  // Accumulates parameter gradients. Call ZeroGrad before a new optimizer
  // step unless accumulation across micro-batches is intended.
  RC Backward(const FloatTensor4D &grad_output, FloatTensor4D &grad_input);
  void ZeroGrad();

  RC set_weight(const std::vector<float> &weight);
  RC set_bias(const std::vector<float> &bias);
  void set_thread_num(int thread_num);

  const Config &config() const { return config_; }
  const std::vector<float> &weight() const { return weight_; }
  const std::vector<float> &bias() const { return bias_; }
  std::vector<float> &mutable_weight() {
    packed_weight_valid_ = false;
    return weight_;
  }
  std::vector<float> &mutable_bias() { return bias_; }
  const std::vector<float> &grad_weight() const { return grad_weight_; }
  const std::vector<float> &grad_bias() const { return grad_bias_; }
  int output_height(int input_height) const;
  int output_width(int input_width) const;
  std::string err_msg() const { return err_msg_; }

private:
  bool ValidateInput(const FloatTensor4D &input, const char *function);
  RC ForwardInternal(const FloatTensor4D &input, FloatTensor4D &output,
                     bool cache_for_backward,
                     const std::vector<float> *channel_scale,
                     const std::vector<float> *channel_bias);
  bool ValidateGradOutput(const FloatTensor4D &grad_output);
  void EnsurePackedWeight();
  void BuildIm2Col(const FloatTensor4D &input, int output_height,
                   int output_width, std::vector<float> &columns) const;
  void ColumnsToInput(const std::vector<float> &grad_columns,
                      FloatTensor4D &grad_input) const;

  Config config_;
  std::vector<float> weight_;
  // Transposed [kernel_element, output_channel] layout. It lets the forward
  // inner loop update consecutive output channels with SIMD FMA and avoids a
  // horizontal reduction for every output scalar.
  std::vector<float> packed_weight_;
  bool packed_weight_valid_ = false;
  std::vector<float> bias_;
  std::vector<float> grad_weight_;
  std::vector<float> grad_bias_;

  FloatTensor4D last_input_;
  std::vector<float> last_columns_;
  int last_output_height_ = 0;
  int last_output_width_ = 0;
  bool is_init_ = false;
  bool has_cache_ = false;
  std::string err_msg_;
};

} // namespace deeplearning
