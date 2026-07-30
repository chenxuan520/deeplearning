#pragma once

#include "neural_network.h"
#include "test.h"

#include <cmath>
#include <cstdlib>
#include <vector>

namespace dropout_test_detail {

using namespace deeplearning;

// 16 个二维样本: 棋盘格标签 (非线性可分), 目标用 one-hot.
inline void MakeData(std::vector<std::vector<double>> &data,
                     std::vector<std::vector<double>> &target) {
  std::srand(3);
  data.clear();
  target.clear();
  for (int i = 0; i < 16; i++) {
    double x = ((std::rand() % 1000) / 500.0 - 1.0);
    double y = ((std::rand() % 1000) / 500.0 - 1.0);
    data.push_back({x, y});
    if (x * y > 0) {
      target.push_back({1.0, 0.0});
    } else {
      target.push_back({0.0, 1.0});
    }
  }
}

// 2 -> 32 -> 32 -> 2, ReLU + MSE + SGD, 固定 seed.
inline NeuralNetwork::RC BuildNet(NeuralNetwork &net) {
  auto rc = net.Init(std::vector<int>{2, 32, 32, 2});
  if (rc != NeuralNetwork::SUCCESS) {
    return rc;
  }
  net.set_random_seed(9);
  rc = net.set_param_init_function(ParamInitType::PARAM_INIT_HE);
  if (rc != NeuralNetwork::SUCCESS) {
    return rc;
  }
  rc = net.set_activate_function(ActivateType::ACTIVATE_RELU);
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
  net.set_learning_rate(0.05);
  return NeuralNetwork::SUCCESS;
}

inline bool SameWeights(NeuralNetwork &a, NeuralNetwork &b) {
  const auto &wa = a.neuron_weight();
  const auto &wb = b.neuron_weight();
  const auto &ba = a.neuron_bias();
  const auto &bb = b.neuron_bias();
  for (size_t l = 0; l < wa.size(); l++) {
    for (size_t o = 0; o < wa[l].size(); o++) {
      for (size_t i = 0; i < wa[l][o].size(); i++) {
        if (wa[l][o][i] != wb[l][o][i]) {
          return false;
        }
      }
    }
    for (size_t o = 0; o < ba[l].size(); o++) {
      if (ba[l][o] != bb[l][o]) {
        return false;
      }
    }
  }
  return true;
}

inline bool WeightsAreFinite(NeuralNetwork &net) {
  for (const auto &layer : net.neuron_weight()) {
    for (const auto &row : layer) {
      for (double v : row) {
        if (!std::isfinite(v)) {
          return false;
        }
      }
    }
  }
  return true;
}

} // namespace dropout_test_detail

TEST(Dropout, SetRateValidation) {
  using namespace deeplearning;

  NeuralNetwork net;
  MUST_EQUAL(net.Init(std::vector<int>{2, 4, 2}), NeuralNetwork::SUCCESS);

  MUST_EQUAL(net.set_dropout_rate(-0.1), NeuralNetwork::INVALID_DATA);
  MUST_EQUAL(net.set_dropout_rate(1.0), NeuralNetwork::INVALID_DATA);
  MUST_EQUAL(net.dropout_rate(), 0.0);

  MUST_EQUAL(net.set_dropout_rate(0.25), NeuralNetwork::SUCCESS);
  MUST_TRUE(net.dropout_rate() == 0.25, "dropout_rate should round-trip");
}

TEST(Dropout, SameSeedReproducible) {
  using namespace dropout_test_detail;
  using namespace deeplearning;

  std::vector<std::vector<double>> data, target;
  MakeData(data, target);

  NeuralNetwork net_a;
  MUST_EQUAL(BuildNet(net_a), NeuralNetwork::SUCCESS);
  MUST_EQUAL(net_a.set_dropout_rate(0.4), NeuralNetwork::SUCCESS);
  MUST_EQUAL(net_a.Train(data, target, nullptr, 50, 4), NeuralNetwork::SUCCESS);

  NeuralNetwork net_b;
  MUST_EQUAL(BuildNet(net_b), NeuralNetwork::SUCCESS);
  MUST_EQUAL(net_b.set_dropout_rate(0.4), NeuralNetwork::SUCCESS);
  MUST_EQUAL(net_b.Train(data, target, nullptr, 50, 4), NeuralNetwork::SUCCESS);

  MUST_TRUE(SameWeights(net_a, net_b),
            "same seed + same config should give identical weights");
}

TEST(Dropout, DropoutChangesTraining) {
  using namespace dropout_test_detail;
  using namespace deeplearning;

  std::vector<std::vector<double>> data, target;
  MakeData(data, target);

  NeuralNetwork net_plain;
  MUST_EQUAL(BuildNet(net_plain), NeuralNetwork::SUCCESS);
  MUST_EQUAL(net_plain.Train(data, target, nullptr, 50, 4),
             NeuralNetwork::SUCCESS);

  NeuralNetwork net_drop;
  MUST_EQUAL(BuildNet(net_drop), NeuralNetwork::SUCCESS);
  MUST_EQUAL(net_drop.set_dropout_rate(0.4), NeuralNetwork::SUCCESS);
  MUST_EQUAL(net_drop.Train(data, target, nullptr, 50, 4),
             NeuralNetwork::SUCCESS);

  MUST_TRUE(!SameWeights(net_plain, net_drop),
            "dropout should change the training trajectory");
}

TEST(Dropout, PredictDeterministicAfterDropoutTraining) {
  using namespace dropout_test_detail;
  using namespace deeplearning;

  std::vector<std::vector<double>> data, target;
  MakeData(data, target);

  NeuralNetwork net;
  MUST_EQUAL(BuildNet(net), NeuralNetwork::SUCCESS);
  MUST_EQUAL(net.set_dropout_rate(0.5), NeuralNetwork::SUCCESS);
  MUST_EQUAL(net.Train(data, target, nullptr, 20, 4), NeuralNetwork::SUCCESS);

  // 推理路径不应再受 dropout 影响: 两次 Predict 必须完全一致,
  // 且输出不能被 1/keep 放大 ( inverted 缩放在训练时就补完了 ).
  std::vector<double> out1, out2;
  MUST_EQUAL(net.Predict(data[0], out1), NeuralNetwork::SUCCESS);
  MUST_EQUAL(net.Predict(data[0], out2), NeuralNetwork::SUCCESS);
  MUST_EQUAL(out1.size(), out2.size());
  for (size_t i = 0; i < out1.size(); i++) {
    MUST_TRUE(out1[i] == out2[i], "inference must be deterministic");
    MUST_TRUE(std::isfinite(out1[i]), "output should stay finite");
  }
}

TEST(Dropout, DropoutHindersMemorization) {
  using namespace dropout_test_detail;
  using namespace deeplearning;

  std::vector<std::vector<double>> data, target;
  MakeData(data, target);

  // 小数据 + 大网络 + 长训练: 无 dropout 几乎能背下来 (train loss 很低),
  // 开 0.5 dropout 后随机掩码会持续干扰, train loss 应显著更高.
  NeuralNetwork net_plain;
  MUST_EQUAL(BuildNet(net_plain), NeuralNetwork::SUCCESS);
  MUST_EQUAL(net_plain.Train(data, target, nullptr, 400, 8),
             NeuralNetwork::SUCCESS);
  double loss_plain = 0;
  MUST_EQUAL(net_plain.CalcLoss(data, target, loss_plain),
             NeuralNetwork::SUCCESS);

  NeuralNetwork net_drop;
  MUST_EQUAL(BuildNet(net_drop), NeuralNetwork::SUCCESS);
  MUST_EQUAL(net_drop.set_dropout_rate(0.5), NeuralNetwork::SUCCESS);
  MUST_EQUAL(net_drop.Train(data, target, nullptr, 400, 8),
             NeuralNetwork::SUCCESS);
  double loss_drop = 0;
  MUST_EQUAL(net_drop.CalcLoss(data, target, loss_drop),
             NeuralNetwork::SUCCESS);

  DEBUG("[dropout] loss plain = " << loss_plain << "  drop = " << loss_drop);
  MUST_TRUE(loss_drop > loss_plain,
            "dropout should make pure memorization harder");
}

TEST(Dropout, ParallelPathSmoke) {
  using namespace dropout_test_detail;
  using namespace deeplearning;

  std::vector<std::vector<double>> data, target;
  MakeData(data, target);

  NeuralNetwork net;
  MUST_EQUAL(BuildNet(net), NeuralNetwork::SUCCESS);
  MUST_EQUAL(net.set_dropout_rate(0.3), NeuralNetwork::SUCCESS);
  net.set_train_thread_num(2);
  MUST_EQUAL(net.Train(data, target, nullptr, 30, 4), NeuralNetwork::SUCCESS);
  MUST_TRUE(WeightsAreFinite(net),
            "parallel path with dropout should keep weights finite");
}

TEST(Dropout, ImportParamKeepsDropoutBehavior) {
  using namespace dropout_test_detail;
  using namespace deeplearning;

  std::vector<std::vector<double>> data, target;
  MakeData(data, target);

  // 直接 Init 训练
  NeuralNetwork net_a;
  MUST_EQUAL(BuildNet(net_a), NeuralNetwork::SUCCESS);
  MUST_EQUAL(net_a.set_dropout_rate(0.4), NeuralNetwork::SUCCESS);
  MUST_EQUAL(net_a.Train(data, target, nullptr, 30, 4), NeuralNetwork::SUCCESS);

  // 导出未训练的初始参数, 走 Import 路径再训 — 应与 Init 路径完全一致
  NeuralNetwork net_src;
  MUST_EQUAL(BuildNet(net_src), NeuralNetwork::SUCCESS);
  NeuralNetwork::NetworkParam param;
  NeuralNetwork::NetworkOption option;
  MUST_EQUAL(net_src.ExportNetworkParam(param, option), NeuralNetwork::SUCCESS);

  NeuralNetwork net_b;
  MUST_EQUAL(net_b.ImportNetworkParam(param, option), NeuralNetwork::SUCCESS);
  MUST_EQUAL(net_b.set_dropout_rate(0.4), NeuralNetwork::SUCCESS);
  MUST_EQUAL(net_b.Train(data, target, nullptr, 30, 4), NeuralNetwork::SUCCESS);

  MUST_TRUE(SameWeights(net_a, net_b),
            "import path should reproduce the exact same dropout sequence");
}

TEST(Dropout, ThreadCountInvariant) {
  using namespace dropout_test_detail;
  using namespace deeplearning;

  std::vector<std::vector<double>> data, target;
  MakeData(data, target);

  NeuralNetwork net_single;
  MUST_EQUAL(BuildNet(net_single), NeuralNetwork::SUCCESS);
  MUST_EQUAL(net_single.set_dropout_rate(0.3), NeuralNetwork::SUCCESS);
  MUST_EQUAL(net_single.Train(data, target, nullptr, 40, 4),
             NeuralNetwork::SUCCESS);

  NeuralNetwork net_multi;
  MUST_EQUAL(BuildNet(net_multi), NeuralNetwork::SUCCESS);
  MUST_EQUAL(net_multi.set_dropout_rate(0.3), NeuralNetwork::SUCCESS);
  net_multi.set_train_thread_num(4);
  MUST_EQUAL(net_multi.Train(data, target, nullptr, 40, 4),
             NeuralNetwork::SUCCESS);

  // 掩码按 (seed, step, 样本槽位) 派生, 与线程数无关; 梯度累加顺序
  // 不同只允许带来浮点 ulp 级差异, 掩码不同则会是 O(1) 级.
  const auto &wa = net_single.neuron_weight();
  const auto &wb = net_multi.neuron_weight();
  double max_diff = 0.0;
  for (size_t l = 0; l < wa.size(); l++) {
    for (size_t o = 0; o < wa[l].size(); o++) {
      for (size_t i = 0; i < wa[l][o].size(); i++) {
        max_diff = std::max(max_diff, std::fabs(wa[l][o][i] - wb[l][o][i]));
      }
    }
  }
  DEBUG("[dropout] thread invariance max weight diff = " << max_diff);
  MUST_TRUE(max_diff < 1e-9,
            "training result should not depend on thread count");
}
