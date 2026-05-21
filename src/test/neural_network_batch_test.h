#pragma once

#include "neural_network.h"
#include "test.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>

namespace mini_batch_test_detail {

using namespace deeplearning;

// 合成一个二分类数据集 (跟 neural_network_test.h 同思路, 但小一点便于回归用)
inline void
MakeData(int seed, int n,
         std::vector<std::vector<double>> &data,
         std::vector<std::vector<double>> &target) {
  std::srand(seed);
  data.clear();
  target.clear();
  data.reserve(n);
  target.reserve(n);
  auto rnd = []() { return (std::rand() % 1000) / 500.0 - 1.0; }; // [-1, 1]
  for (int i = 0; i < n; i++) {
    double x = rnd();
    double y = rnd();
    // 决策边界: y > x^2 - 0.3
    double boundary = x * x - 0.3;
    int label = (y > boundary) ? 0 : 1;
    data.push_back({x, y});
    if (label == 0) {
      target.push_back({1.0, 0.0});
    } else {
      target.push_back({0.0, 1.0});
    }
  }
}

inline double Accuracy(NeuralNetwork &network,
                       const std::vector<std::vector<double>> &data,
                       const std::vector<std::vector<double>> &target) {
  int correct = 0;
  for (int i = 0; i < (int)data.size(); i++) {
    std::vector<double> r;
    auto rc = network.Predict(data[i], r);
    if (rc != NeuralNetwork::SUCCESS) {
      return -1;
    }
    int pred = (r[0] > r[1]) ? 0 : 1;
    int truth = (target[i][0] > target[i][1]) ? 0 : 1;
    if (pred == truth) {
      correct++;
    }
  }
  return correct * 1.0 / (int)data.size();
}

inline NeuralNetwork::RC
BuildNet(NeuralNetwork &net, int seed) {
  auto rc = net.Init(std::vector<int>{2, 8, 8, 2});
  if (rc != NeuralNetwork::SUCCESS) {
    return rc;
  }
  net.set_random_seed(seed);
  rc = net.set_param_init_function(ParamInitType::PARAM_INIT_XAVIER);
  if (rc != NeuralNetwork::SUCCESS) {
    return rc;
  }
  rc = net.set_loss_function(LossType::LOSS_MSE);
  if (rc != NeuralNetwork::SUCCESS) {
    return rc;
  }
  rc = net.set_softmax_function(SoftmaxType::SOFTMAX_STD);
  if (rc != NeuralNetwork::SUCCESS) {
    return rc;
  }
  net.set_learning_rate(0.05);
  return NeuralNetwork::SUCCESS;
}

} // namespace mini_batch_test_detail

TEST(MiniBatch, ConvergeBatch1) {
  using namespace mini_batch_test_detail;
  using namespace deeplearning;
  std::vector<std::vector<double>> tr, tr_t, te, te_t;
  MakeData(1, 2000, tr, tr_t);
  MakeData(2, 500, te, te_t);

  NeuralNetwork net;
  MUST_EQUAL(BuildNet(net, 1), NeuralNetwork::SUCCESS);

  auto rc = net.Train(tr, tr_t, nullptr, 8000, 1);
  MUST_TRUE(rc == NeuralNetwork::SUCCESS, net.err_msg());

  double acc = Accuracy(net, te, te_t);
  DEBUG("batch=1  acc=" << acc);
  MUST_TRUE(acc > 0.85, "batch=1 should converge");
}

TEST(MiniBatch, ConvergeBatch8) {
  using namespace mini_batch_test_detail;
  using namespace deeplearning;
  std::vector<std::vector<double>> tr, tr_t, te, te_t;
  MakeData(1, 2000, tr, tr_t);
  MakeData(2, 500, te, te_t);

  NeuralNetwork net;
  MUST_EQUAL(BuildNet(net, 1), NeuralNetwork::SUCCESS);

  // 同样 8000 step, batch=8 → 处理 64000 sample
  auto rc = net.Train(tr, tr_t, nullptr, 8000, 8);
  MUST_TRUE(rc == NeuralNetwork::SUCCESS, net.err_msg());

  double acc = Accuracy(net, te, te_t);
  DEBUG("batch=8  acc=" << acc);
  MUST_TRUE(acc > 0.85, "batch=8 should converge");
}

TEST(MiniBatch, ConvergeBatch32) {
  using namespace mini_batch_test_detail;
  using namespace deeplearning;
  std::vector<std::vector<double>> tr, tr_t, te, te_t;
  MakeData(1, 2000, tr, tr_t);
  MakeData(2, 500, te, te_t);

  NeuralNetwork net;
  MUST_EQUAL(BuildNet(net, 1), NeuralNetwork::SUCCESS);

  auto rc = net.Train(tr, tr_t, nullptr, 8000, 32);
  MUST_TRUE(rc == NeuralNetwork::SUCCESS, net.err_msg());

  double acc = Accuracy(net, te, te_t);
  DEBUG("batch=32 acc=" << acc);
  MUST_TRUE(acc > 0.85, "batch=32 should converge");
}

// benchmark: 公平对比 (相同总样本数, 不同 batch_size)
// 输出便于人工检查; 不强约束加速比 (避免 CI 抖动)
TEST(MiniBatch, BenchmarkSameSampleCount) {
  using namespace mini_batch_test_detail;
  using namespace deeplearning;
  using clk = std::chrono::high_resolution_clock;

  std::vector<std::vector<double>> tr, tr_t;
  MakeData(1, 2000, tr, tr_t);

  const int total_samples = 32000;

  auto run_once = [&](int batch_num, double &elapsed_ms) -> double {
    NeuralNetwork net;
    if (BuildNet(net, 1) != NeuralNetwork::SUCCESS) {
      return -1;
    }
    int steps = total_samples / batch_num;
    auto t0 = clk::now();
    auto rc = net.Train(tr, tr_t, nullptr, steps, batch_num);
    auto t1 = clk::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (rc != NeuralNetwork::SUCCESS) {
      return -1;
    }
    return Accuracy(net, tr, tr_t);
  };

  double t1_ms = 0, t8_ms = 0, t32_ms = 0;
  double acc1 = run_once(1, t1_ms);
  double acc8 = run_once(8, t8_ms);
  double acc32 = run_once(32, t32_ms);

  DEBUG("[batch=1 ] " << total_samples << " samples in " << t1_ms
                       << " ms, acc=" << acc1);
  DEBUG("[batch=8 ] " << total_samples << " samples in " << t8_ms
                       << " ms, acc=" << acc8);
  DEBUG("[batch=32] " << total_samples << " samples in " << t32_ms
                       << " ms, acc=" << acc32);
  if (t1_ms > 0 && t8_ms > 0 && t32_ms > 0) {
    DEBUG("speedup vs batch=1: batch=8 x" << (t1_ms / t8_ms) << ", batch=32 x"
                                          << (t1_ms / t32_ms));
  }

  MUST_TRUE(acc1 > 0.8, "batch=1 should learn");
  MUST_TRUE(acc8 > 0.8, "batch=8 should learn");
  MUST_TRUE(acc32 > 0.8, "batch=32 should learn");
}
