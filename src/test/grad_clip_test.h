#pragma once

#include "neural_network.h"
#include "test.h"

#include <cmath>
#include <cstdlib>
#include <vector>

namespace grad_clip_test_detail {

using namespace deeplearning;

// 故意构造梯度爆炸场景: tanh + 大学习率 + 离群点, 关掉 clipping
// 训练时容易出现 NaN / Inf; 打开 clip_norm 之后应该稳定.
inline NeuralNetwork::RC BuildBlowupNet(NeuralNetwork &net) {
  auto rc = net.Init(std::vector<int>{2, 8, 8, 2});
  if (rc != NeuralNetwork::SUCCESS) {
    return rc;
  }
  net.set_random_seed(7);
  rc = net.set_param_init_function(ParamInitType::PARAM_INIT_XAVIER);
  if (rc != NeuralNetwork::SUCCESS) {
    return rc;
  }
  rc = net.set_loss_function(LossType::LOSS_MSE);
  if (rc != NeuralNetwork::SUCCESS) {
    return rc;
  }
  rc = net.set_optimizer_function(OptimizerType::OPTIMIZER_SGD);
  if (rc != NeuralNetwork::SUCCESS) {
    return rc;
  }
  net.set_learning_rate(50.0); // 故意非常大
  return NeuralNetwork::SUCCESS;
}

inline bool WeightsAreFinite(NeuralNetwork &net) {
  const auto &w = net.neuron_weight();
  const auto &b = net.neuron_bias();
  for (int l = 1; l < (int)w.size(); l++) {
    for (auto &row : w[l]) {
      for (double v : row) {
        if (!std::isfinite(v)) {
          return false;
        }
      }
    }
    for (double v : b[l]) {
      if (!std::isfinite(v)) {
        return false;
      }
    }
  }
  return true;
}

inline void MakeData(int n,
                     std::vector<std::vector<double>> &data,
                     std::vector<std::vector<double>> &target) {
  std::srand(11);
  data.clear();
  target.clear();
  // 大部分正常数据 + 几个极端离群点
  for (int i = 0; i < n; i++) {
    double x = ((std::rand() % 1000) / 500.0 - 1.0);
    double y = ((std::rand() % 1000) / 500.0 - 1.0);
    if (i < 5) {
      x *= 100.0;
      y *= 100.0;
    }
    data.push_back({x, y});
    if (x + y > 0) {
      target.push_back({1.0, 0.0});
    } else {
      target.push_back({0.0, 1.0});
    }
  }
}

} // namespace grad_clip_test_detail

TEST(GradientClip, ClipNormPreventsBlowup) {
  using namespace grad_clip_test_detail;
  using namespace deeplearning;

  std::vector<std::vector<double>> data, target;
  MakeData(200, data, target);

  // 关掉 clip 时, 极大 lr + 离群点 → 数值爆炸 (允许失败).
  NeuralNetwork net_no_clip;
  MUST_EQUAL(BuildBlowupNet(net_no_clip), NeuralNetwork::SUCCESS);
  net_no_clip.Train(data, target, nullptr, 200, 8);
  bool finite_no_clip = WeightsAreFinite(net_no_clip);
  DEBUG("[no clip]  weights finite = " << finite_no_clip);

  // 打开 clip_norm 之后应该稳定.
  NeuralNetwork net_clip;
  MUST_EQUAL(BuildBlowupNet(net_clip), NeuralNetwork::SUCCESS);
  net_clip.set_gradient_clip_norm(1.0);
  auto rc = net_clip.Train(data, target, nullptr, 200, 8);
  MUST_EQUAL(rc, NeuralNetwork::SUCCESS);

  MUST_TRUE(WeightsAreFinite(net_clip),
            "weights should stay finite with clip_norm enabled");
}

TEST(GradientClip, ClipValueLimitsMagnitude) {
  using namespace grad_clip_test_detail;
  using namespace deeplearning;

  std::vector<std::vector<double>> data, target;
  MakeData(200, data, target);

  NeuralNetwork net;
  MUST_EQUAL(BuildBlowupNet(net), NeuralNetwork::SUCCESS);
  net.set_gradient_clip_value(0.5);
  auto rc = net.Train(data, target, nullptr, 200, 8);
  MUST_EQUAL(rc, NeuralNetwork::SUCCESS);

  MUST_TRUE(WeightsAreFinite(net),
            "weights should stay finite with clip_value enabled");
}
