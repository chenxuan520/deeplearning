#pragma once
#include <utility>
#include <vector>

namespace deeplearning {

enum OptimizerType {
  OPTIMIZER_SGD,
  OPTIMIZER_MOMENTUM,
  OPTIMIZER_ADAM,
  OPTIMIZER_RMSPROP,
  OPTIMIZER_ADAMW,
};

class OptimizerFunction {
public:
  OptimizerFunction(const std::vector<int> &layer);
  virtual ~OptimizerFunction() = default;

  // 每个 minibatch update 开始时调用一次 (在所有 CalcChangeValue 之前).
  // 默认为 no-op; Adam / AdamW 在此推进 step 计数, 以便 bias correction.
  virtual void BeforeStep() {}

  // 计算单个参数的更新量, 返回值会从参数中扣除 (param -= return_value).
  // - delta: 平均梯度 (已经按 batch_size 取过平均)
  // - learning_rate: 当前学习率
  // - pos: {layer_idx, neuron_idx}
  // - weight_pos: 输入维度索引; -1 表示这是 bias.
  // - param_value: 当前参数值, 用于 weight decay; 不需要时可以忽略.
  virtual double CalcChangeValue(double delta, double learning_rate,
                                 const std::pair<int, int> &pos,
                                 int weight_pos = -1,
                                 double param_value = 0.0) = 0;

  virtual OptimizerType GetOptimizerType() = 0;

  // L2 weight decay (默认 0 = 关闭). 对 weight 应用, bias 不应用.
  // 不同优化器内部对 wd 的耦合方式不同 (Adam vs AdamW), 见各自实现.
  void set_weight_decay(double wd) { weight_decay_ = wd; }
  double weight_decay() const { return weight_decay_; }

protected:
  std::vector<int> layer_;
  double weight_decay_ = 0.0;
};

} // namespace deeplearning
