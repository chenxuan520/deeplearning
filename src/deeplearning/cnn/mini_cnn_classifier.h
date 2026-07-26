#pragma once

#include "cnn/conv2d.h"
#include "cnn/max_pool2d.h"

#include <functional>
#include <string>
#include <vector>

namespace deeplearning {

class MiniCNNClassifier {
public:
  using Matrix = std::vector<std::vector<double>>;
  using Tensor3D = Conv2D::Tensor3D;

  struct Config {
    int input_channels_ = 1;
    int input_height_ = 0;
    int input_width_ = 0;
    int conv_channels_ = 1;
    int kernel_height_ = 3;
    int kernel_width_ = 3;
    int conv_stride_ = 1;
    int conv_padding_ = 0;
    int pool_height_ = 2;
    int pool_width_ = 2;
    int pool_stride_ = 2;
    int class_num_ = 0;
    int rand_seed_ = 0;
  };

  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
  };

public:
  RC Init(const Config &config);
  RC Forward(const Tensor3D &input, std::vector<double> &logits);
  RC PredictProbs(const Tensor3D &input, std::vector<double> &probs);
  RC Predict(const Tensor3D &input, int &label);
  RC Train(const std::vector<Tensor3D> &images, const std::vector<int> &labels,
           std::function<void(int epoch_num, double average_loss,
                               bool &early_stop)>
                each_epoch_call = nullptr,
           int epoch_num = 1, double learning_rate = 0.1);
  RC TrainBatch(
      const std::vector<Tensor3D> &images, const std::vector<int> &labels,
      int batch_size,
      std::function<void(int epoch_num, double average_loss, bool &early_stop)>
          each_epoch_call = nullptr,
      int epoch_num = 1, double learning_rate = 0.1);

  void set_random_seed(int seed);
  void set_train_thread_num(int thread_num);
  int train_thread_num() const { return train_thread_num_; }
  RC set_fc_weight(const Matrix &weight);
  RC set_fc_bias(const std::vector<double> &bias);

  std::string err_msg();
  Config config() const;
  int flattened_dim() const;
  Conv2D &conv();
  MaxPool2D &pool();
  const Matrix &fc_weight() const;
  const std::vector<double> &fc_bias() const;

private:
  struct GradientBuffer {
    Matrix fc_weight;
    std::vector<double> fc_bias;
    Conv2D::Tensor4D conv_weight;
    std::vector<double> conv_bias;
  };

  RC ValidateInput(const Tensor3D &input, const char *func_name);
  bool InputHasValidShape(const Tensor3D &input) const;
  RC ForwardFeature(const Tensor3D &input, Tensor3D &conv_output,
                    Tensor3D &relu_output, Tensor3D &pooled_output,
                    std::vector<double> &flattened);
  RC ForwardFeatureWithLayers(const Tensor3D &input, Conv2D &conv,
                              MaxPool2D &pool, Tensor3D &conv_output,
                              Tensor3D &relu_output,
                              Tensor3D &pooled_output,
                              std::vector<double> &flattened,
                              std::string &err_msg) const;
  void InitGradientBuffer(GradientBuffer &gradient) const;
  void AddGradientBuffer(GradientBuffer &dst,
                         const GradientBuffer &src) const;
  RC ApplyGradientBuffer(const GradientBuffer &gradient, double learning_rate,
                         double gradient_scale);
  RC AccumulateGradientsRange(
      const std::vector<Tensor3D> &images, const std::vector<int> &labels,
      const std::vector<int> &order, int begin, int end,
      GradientBuffer &gradient, double &loss_sum, std::string &err_msg) const;
  RC AccumulateGradientsBatchParallel(
      const std::vector<Tensor3D> &images, const std::vector<int> &labels,
      const std::vector<int> &order, int begin, int end,
      GradientBuffer &gradient, double &loss_sum);

private:
  Config config_;
  int flattened_dim_ = 0;
  int train_thread_num_ = 1;
  Matrix fc_weight_;
  std::vector<double> fc_bias_;
  Conv2D conv_;
  MaxPool2D pool_;
  std::string err_msg_;
  bool is_init_ = false;
};

} // namespace deeplearning
