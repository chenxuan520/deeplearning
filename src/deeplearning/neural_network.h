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
#include <random>
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
  enum HiddenLayerWidenMode {
    WIDEN_RANDOM,
    WIDEN_ZERO_OUTGOING,
    WIDEN_NET2WIDER,
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

  // Expands one hidden layer while retaining all existing parameters. The
  // optimizer state is reset because its tensors depend on the old shape.
  RC WidenHiddenLayer(int layer_index, int new_width,
                      HiddenLayerWidenMode mode,
                      ParamInitType new_param_init = PARAM_INIT_HE);

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

  // Dropout (inverted), 只作用在隐藏层 (1..L-2), rate 取值 [0, 1).
  // rate <= 0 表示关闭 (默认). 开启后:
  //  - Train 的每次前向都为每个隐藏单元重新采样掩码: 以概率 rate 置 0,
  //    幸存输出乘 1/(1-rate) 补量, 反向时 delta 乘同一份掩码;
  //  - Predict / PredictBatch / CalcLoss 等推理路径不受影响.
  // 掩码是 (rand_seed_, 训练步数, 样本在 batch 内的槽位) 的确定函数:
  // 相同 seed + 相同调用序列可复现, 且与线程数 / batch 切分方式无关.
  RC set_dropout_rate(double rate);
  double dropout_rate() const { return dropout_rate_; }

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
      const std::vector<std::vector<double>> &batch_target, int train_step = 0);

  // train_step 是当前 batch 的全局步数, 与 (rand_seed_, 样本槽位) 一起
  // 派生 dropout 掩码种子, 保证掩码与线程数/batch 切分方式无关.
  RC AccumulateGradientsRange(
      const std::vector<std::vector<double>> &batch_data,
      const std::vector<std::vector<double>> &batch_target, int begin, int end,
      const std::shared_ptr<LossFunction> &loss_function,
      const std::shared_ptr<ActivateFunction> &activate_function,
      const std::shared_ptr<SoftmaxFunction> &softmax_function,
      GradientBuffer &gradient, std::string &err_msg,
      int train_step = 0) const;

  // training = true 时按 dropout_rate_ 采样掩码 (Train 路径);
  // 推理调用传 false, 不启用 dropout.
  RC ForwardPropagationBatch(
      const std::vector<std::vector<double>> &batch_data,
      bool training = false);

  RC UpdateNeuronOutputBatchSoftMax();

  RC BackPropagationBatch(
      const std::vector<std::vector<double>> &batch_target);

  RC ApplyGradient(int batch_size);

  // 对累加好的 grad_bias_ / grad_weight_ 做就地裁剪.
  // 调用前 grad 已经按 sum 累加 (未除 batch_size); 这里裁剪的是
  // 平均后的梯度, 故先除 batch_size 再判定.
  void ClipGradients(int batch_size);

  // 由 (rand_seed, 训练步数, 样本槽位) 派生掩码种子 (splitmix64 混合),
  // 让掩码绑定到样本而不是线程/batch 切分方式.
  static unsigned int DropoutSeed(int rand_seed, int step, int slot);

  // 用传入的 gen 填充 factors[0..n): 以 keep 概率填 1/keep, 否则 0.
  // gen 由调用方按样本创建 (跨层连续采样), 保证单/多线程序列一致.
  void FillDropoutMask(std::mt19937 &gen, double *factors, int n) const;

  // 训练前向是否已为 layer l 采样好本 batch 的掩码.
  bool DropoutMaskActive(int layer, int batch_size) const;

  // 按需分配掩码缓冲 (形状不变时不重复分配).
  void ResizeDropoutMask(int batch_size);

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

  // Dropout: rate <= 0 关闭. 掩码存的是乘性因子 (0 或 1/keep),
  // 单线程路径在训练前向时填充, 供同一次反向使用; 多线程路径的掩码
  // 是 AccumulateGradientsRange 内的局部变量 (按槽位派生种子, 两条
  // 路径采出的掩码完全一致).
  double dropout_rate_ = 0.0;
  int dropout_step_ = 0; // 已处理的 batch 数, 跨 Train 调用累计
  // [layer][batch][neuron], 只填隐藏层
  std::vector<std::vector<std::vector<double>>> dropout_mask_;

  std::string err_msg_;
};

} // namespace deeplearning
