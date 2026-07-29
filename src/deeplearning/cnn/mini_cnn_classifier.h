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
    int conv2_channels_ = 0;
    int kernel_height_ = 3;
    int kernel_width_ = 3;
    int conv_stride_ = 1;
    int conv_padding_ = 0;
    int pool_height_ = 2;
    int pool_width_ = 2;
    int pool_stride_ = 2;
    int class_num_ = 0;
    int rand_seed_ = 0;
    int hidden_dim_ = 0;
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
  RC ForwardScore(const Tensor3D &input, int class_idx, double &score);
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
  RC TrainParallel(
      const std::vector<Tensor3D> &images, const std::vector<int> &labels,
      std::function<void(int epoch_num, double average_loss, bool &early_stop)>
          each_epoch_call = nullptr,
      int epoch_num = 1, double learning_rate = 0.1, int batch_size = 32,
      int thread_num = 4);
  RC TrainPairwise(
      const std::vector<Tensor3D> &positive_images,
      const std::vector<Tensor3D> &negative_images,
      std::function<void(int epoch_num, double average_loss, bool &early_stop)>
          each_epoch_call = nullptr,
      int epoch_num = 1, double learning_rate = 0.1);
  RC TrainPairwiseParallel(
      const std::vector<Tensor3D> &positive_images,
      const std::vector<Tensor3D> &negative_images,
      std::function<void(int epoch_num, double average_loss, bool &early_stop)>
          each_epoch_call = nullptr,
      int epoch_num = 1, double learning_rate = 0.1, int batch_size = 32,
      int thread_num = 4);

  void set_random_seed(int seed);
  void set_train_thread_num(int thread_num);
  int train_thread_num() const { return train_thread_num_; }
  RC set_fc_weight(const Matrix &weight);
  RC set_fc_bias(const std::vector<double> &bias);
  RC set_hidden_weight(const Matrix &weight);
  RC set_hidden_bias(const std::vector<double> &bias);

  std::string err_msg();
  Config config() const;
  int flattened_dim() const;
  Conv2D &conv();
  const Conv2D &conv() const;
  Conv2D &conv2();
  const Conv2D &conv2() const;
  MaxPool2D &pool();
  const MaxPool2D &pool() const;
  const Matrix &fc_weight() const;
  const std::vector<double> &fc_bias() const;
  const Matrix &hidden_weight() const;
  const std::vector<double> &hidden_bias() const;

private:
  RC ValidateInput(const Tensor3D &input, const char *func_name);
  RC ForwardFeature(const Tensor3D &input, Tensor3D &conv_output,
                    Tensor3D &relu_output, Tensor3D &pooled_output,
                    std::vector<double> &flattened);
  RC ApplyLogitGradient(const Tensor3D &input, int class_idx,
                        double grad_logit, double learning_rate);
  RC ApplyPairwiseGradient(const Tensor3D &positive_image,
                           const Tensor3D &negative_image,
                           double learning_rate, double *loss);

private:
  Config config_;
  int flattened_dim_ = 0;
  int train_thread_num_ = 1;
  Matrix hidden_weight_;
  std::vector<double> hidden_bias_;
  Matrix fc_weight_;
  std::vector<double> fc_bias_;
  Conv2D conv_;
  Conv2D conv2_;
  MaxPool2D pool_;
  std::string err_msg_;
  bool is_init_ = false;
};

} // namespace deeplearning
