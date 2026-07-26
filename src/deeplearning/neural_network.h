#pragma once
#include "activate/activate_factory.h"
#include "loss/loss_factory.h"
#include "lr_scheduler/lr_scheduler_base.h"
#include "optimizer/optimizer_factory.h"
#include "param_init/param_init_factory.h"
#include "softmax/softmax_factory.h"
#include "util/random.h"
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
namespace deeplearning {

class NeuralNetwork {
public:
  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
  };
  enum NetworkStatus {
    NETWORK_STATUS_UNINIT,
    NETWORK_STATUS_INIT,
  };
  struct NetworkParam {
    std::vector<int> layer_;
    std::vector<std::vector<double>> neuron_bias_;
    std::vector<std::vector<std::vector<double>>> neuron_weight_;
  };
  struct NetworkOption {
    double learning_rate_;
    int rand_seed_;
    LossType loss_type_;
    ActivateType activate_type_;
    SoftmaxType softmax_type_;
    OptimizerType optimizer_type_;
  };

public:
  NeuralNetwork();
  ~NeuralNetwork();
  NeuralNetwork(const NeuralNetwork &) = delete;
  NeuralNetwork &operator=(const NeuralNetwork &) = delete;

  NeuralNetwork(const std::vector<int> &layer);

  RC Init(const std::vector<int> &layer);

  RC Train(const std::vector<std::vector<double>> &data,
           const std::vector<std::vector<double>> &target,
           std::function<void(NeuralNetwork &network, int epoch_num,
                               bool &early_stop)>
               each_epoch_call = nullptr,
           int epoch_num = 0, int batch_num = 1, double learning_rate = 0);

  RC Predict(const std::vector<double> &data, std::vector<double> &result);

  // 一次前向多个样本, 比循环调用 Predict 快很多 (单矩阵乘法 + 一次激活).
  // result.size() == data.size(), result[i] 是第 i 个样本的输出.
  RC PredictBatch(const std::vector<std::vector<double>> &data,
                  std::vector<std::vector<double>> &result);

  RC CalcLoss(const std::vector<std::vector<double>> &data,
              const std::vector<std::vector<double>> &target, double &loss);

  RC ExportNetworkParam(NetworkParam &param, NetworkOption &option);

  RC ImportNetworkParam(const NetworkParam &param,
                        const NetworkOption &option);

  RC Clone(const NeuralNetwork &old);

public:
  std::string err_msg();
  double learning_rate();
  int rand_seed();
  NetworkStatus network_status();
  const std::vector<std::vector<std::vector<double>>> &neuron_weight();
  const std::vector<std::vector<double>> &neuron_bias();

  void set_learning_rate(double rate);
  // Optional CPU parallelism for minibatch gradient calculation. Default is 1,
  // which preserves the original single-threaded training path.
  void set_train_thread_num(int thread_num);
  int train_thread_num() const { return train_thread_num_; }
  void set_random_seed(int seed);
  RC set_loss_function(LossType type);
  RC set_activate_function(ActivateType type);
  RC set_softmax_function(SoftmaxType type);
  RC set_param_init_function(ParamInitType type);
  RC set_optimizer_function(OptimizerType type);
  // 直接注入自定义优化器实例 (用于设置 Adam beta1/beta2、weight decay 等
  // 超参). 比起 OptimizerType 枚举更灵活, 适合需要细调的场景.
  RC set_optimizer_function(std::shared_ptr<OptimizerFunction> optimizer);

  // 访问当前优化器, 用于设置 weight decay / beta / momentum 等超参.
  // 返回 nullptr 表示尚未 Init / Import.
  std::shared_ptr<OptimizerFunction> optimizer_function();

  // 梯度裁剪:
  //  - clip_norm > 0 时, 整体梯度 L2 范数超过阈值就按比例缩放 (全局裁剪)
  //  - clip_value > 0 时, 每个梯度分量分别裁剪到 [-value, value]
  // 二者可以同时开启 (先 by-value 再 by-norm).
  void set_gradient_clip_norm(double max_norm);
  void set_gradient_clip_value(double max_value);
  double gradient_clip_norm() const { return grad_clip_norm_; }
  double gradient_clip_value() const { return grad_clip_value_; }

  // 学习率调度器: 设置之后, Train 会在每个 step 调用 scheduler->GetLR(step)
  // 并通过 set_learning_rate 更新. 传 nullptr 表示禁用 (Train 仍按
  // learning_rate_ / 入参覆盖的方式运作).
  void set_lr_scheduler(std::shared_ptr<LRScheduler> scheduler);
  std::shared_ptr<LRScheduler> lr_scheduler() { return lr_scheduler_; }

private:
  struct GradientBuffer {
    std::vector<std::vector<double>> bias;
    std::vector<std::vector<std::vector<double>>> weight;
  };

  void InitParamWithLayer(const std::vector<int> &layer);

  void ResizeBatchBuffers(int batch_size);

  void ResetGradients();

  void InitGradientBuffer(GradientBuffer &buffer) const;

  RC AccumulateGradientsBatchParallel(
      const std::vector<std::vector<double>> &batch_data,
      const std::vector<std::vector<double>> &batch_target);

  RC AccumulateGradientsRange(
      const std::vector<std::vector<double>> &batch_data,
      const std::vector<std::vector<double>> &batch_target, int begin, int end,
      const std::shared_ptr<LossFunction> &loss_function,
      const std::shared_ptr<ActivateFunction> &activate_function,
      const std::shared_ptr<SoftmaxFunction> &softmax_function,
      GradientBuffer &gradient, std::string &err_msg) const;

  RC ForwardPropagationBatch(
      const std::vector<std::vector<double>> &batch_data);

  RC UpdateNeuronOutputBatchSoftMax();

  RC BackPropagationBatch(
      const std::vector<std::vector<double>> &batch_target);

  RC ApplyGradient(int batch_size);

  // 对累加好的 grad_bias_ / grad_weight_ 做就地裁剪.
  // 调用前 grad 已经按 sum 累加 (未除 batch_size); 这里裁剪的是
  // 平均后的梯度, 故先除 batch_size 再判定.
  void ClipGradients(int batch_size);

private:
  std::shared_ptr<LossFunction> loss_function_ = nullptr;
  std::shared_ptr<ActivateFunction> activate_function_ = nullptr;
  std::shared_ptr<SoftmaxFunction> softmax_function_ = nullptr;
  std::shared_ptr<ParamInitFunction> param_init_function_ = nullptr;
  std::shared_ptr<OptimizerFunction> optimizer_function_ = nullptr;

  NetworkStatus network_status_ = NETWORK_STATUS_UNINIT;
  int rand_seed_ = 0;
  double learning_rate_ = 0.1;
  int train_thread_num_ = 1;
  std::vector<int> layer_;

  // 共享参数 (训练期间被 ApplyGradient 更新)
  std::vector<std::vector<double>> neuron_bias_;            // [layer][neuron]
  std::vector<std::vector<std::vector<double>>> neuron_weight_; // [layer][out][in]

  // 批量前向/反向激活 (按 batch 重置)
  std::vector<std::vector<std::vector<double>>> neuron_output_; // [layer][batch][neuron]
  std::vector<std::vector<std::vector<double>>> neuron_preact_; // [layer][batch][neuron], 激活前 z
  std::vector<std::vector<std::vector<double>>> neuron_delta_;  // [layer][batch][neuron]

  // 批量梯度累加 (按 batch 重置, ApplyGradient 时取平均)
  std::vector<std::vector<double>> grad_bias_;              // [layer][neuron]
  std::vector<std::vector<std::vector<double>>> grad_weight_;   // [layer][out][in]

  int batch_buffer_size_ = 0; // 当前 batch buffer 容量

  // 梯度裁剪阈值 (<=0 表示关闭)
  double grad_clip_norm_ = 0.0;
  double grad_clip_value_ = 0.0;

  // 可选 LR scheduler
  std::shared_ptr<LRScheduler> lr_scheduler_;

  std::string err_msg_;
};

} // namespace deeplearning
