#pragma once

#include "neural_network.h"
#include "optimizer/adam_optimizer.h"
#include "optimizer/adamw_optimizer.h"
#include "optimizer/rmsprop_optimizer.h"
#include "optimizer/sgd_optimizer.h"
#include "test.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>

namespace optimizer_test_detail {

using namespace deeplearning;

inline void
MakeData(int seed, int n,
         std::vector<std::vector<double>> &data,
         std::vector<std::vector<double>> &target) {
  std::srand(seed);
  data.clear();
  target.clear();
  data.reserve(n);
  target.reserve(n);
  auto rnd = []() { return (std::rand() % 1000) / 500.0 - 1.0; };
  for (int i = 0; i < n; i++) {
    double x = rnd();
    double y = rnd();
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
BuildNet(NeuralNetwork &net, int seed, OptimizerType opt, double lr) {
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
  rc = net.set_optimizer_function(opt);
  if (rc != NeuralNetwork::SUCCESS) {
    return rc;
  }
  net.set_learning_rate(lr);
  return NeuralNetwork::SUCCESS;
}

} // namespace optimizer_test_detail

TEST(Optimizer, AdamConverges) {
  using namespace optimizer_test_detail;
  using namespace deeplearning;
  std::vector<std::vector<double>> tr, tr_t, te, te_t;
  MakeData(1, 2000, tr, tr_t);
  MakeData(2, 500, te, te_t);

  NeuralNetwork net;
  // Adam 一般用较小的 lr (0.001-0.01) 比 SGD 的 0.05 更合适
  MUST_EQUAL(BuildNet(net, 1, OptimizerType::OPTIMIZER_ADAM, 0.01),
             NeuralNetwork::SUCCESS);

  auto rc = net.Train(tr, tr_t, nullptr, 4000, 8);
  MUST_TRUE(rc == NeuralNetwork::SUCCESS, net.err_msg());

  double acc = Accuracy(net, te, te_t);
  DEBUG("Adam acc=" << acc);
  MUST_TRUE(acc > 0.85, "Adam should converge");
}

TEST(Optimizer, AdamWConverges) {
  using namespace optimizer_test_detail;
  using namespace deeplearning;
  std::vector<std::vector<double>> tr, tr_t, te, te_t;
  MakeData(1, 2000, tr, tr_t);
  MakeData(2, 500, te, te_t);

  NeuralNetwork net;
  MUST_EQUAL(BuildNet(net, 1, OptimizerType::OPTIMIZER_ADAMW, 0.01),
             NeuralNetwork::SUCCESS);
  // 启用 weight decay = 1e-4
  net.optimizer_function()->set_weight_decay(1e-4);

  auto rc = net.Train(tr, tr_t, nullptr, 4000, 8);
  MUST_TRUE(rc == NeuralNetwork::SUCCESS, net.err_msg());

  double acc = Accuracy(net, te, te_t);
  DEBUG("AdamW(wd=1e-4) acc=" << acc);
  MUST_TRUE(acc > 0.85, "AdamW should converge");
}

TEST(Optimizer, RMSPropConverges) {
  using namespace optimizer_test_detail;
  using namespace deeplearning;
  std::vector<std::vector<double>> tr, tr_t, te, te_t;
  MakeData(1, 2000, tr, tr_t);
  MakeData(2, 500, te, te_t);

  NeuralNetwork net;
  MUST_EQUAL(BuildNet(net, 1, OptimizerType::OPTIMIZER_RMSPROP, 0.005),
             NeuralNetwork::SUCCESS);

  auto rc = net.Train(tr, tr_t, nullptr, 4000, 8);
  MUST_TRUE(rc == NeuralNetwork::SUCCESS, net.err_msg());

  double acc = Accuracy(net, te, te_t);
  DEBUG("RMSProp acc=" << acc);
  MUST_TRUE(acc > 0.85, "RMSProp should converge");
}

TEST(Optimizer, SGDWithWeightDecayStaysSane) {
  using namespace optimizer_test_detail;
  using namespace deeplearning;
  std::vector<std::vector<double>> tr, tr_t, te, te_t;
  MakeData(1, 2000, tr, tr_t);
  MakeData(2, 500, te, te_t);

  NeuralNetwork net;
  MUST_EQUAL(BuildNet(net, 1, OptimizerType::OPTIMIZER_SGD, 0.05),
             NeuralNetwork::SUCCESS);
  net.optimizer_function()->set_weight_decay(1e-3);

  auto rc = net.Train(tr, tr_t, nullptr, 4000, 8);
  MUST_TRUE(rc == NeuralNetwork::SUCCESS, net.err_msg());

  double acc = Accuracy(net, te, te_t);
  DEBUG("SGD(wd=1e-3) acc=" << acc);
  MUST_TRUE(acc > 0.8, "SGD with weight decay should still converge");
}

// 速度对比, 不强约束加速比, 仅人工观察
TEST(Optimizer, AdamVsSGDOnSameBudget) {
  using namespace optimizer_test_detail;
  using namespace deeplearning;
  using clk = std::chrono::high_resolution_clock;

  std::vector<std::vector<double>> tr, tr_t, te, te_t;
  MakeData(1, 2000, tr, tr_t);
  MakeData(2, 500, te, te_t);

  auto run = [&](OptimizerType opt, double lr, double &acc_out,
                 double &ms_out) {
    NeuralNetwork net;
    if (BuildNet(net, 1, opt, lr) != NeuralNetwork::SUCCESS) {
      return false;
    }
    auto t0 = clk::now();
    auto rc = net.Train(tr, tr_t, nullptr, 2000, 8);
    auto t1 = clk::now();
    ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (rc != NeuralNetwork::SUCCESS) {
      return false;
    }
    acc_out = Accuracy(net, te, te_t);
    return true;
  };

  double sgd_acc = 0, sgd_ms = 0;
  double adam_acc = 0, adam_ms = 0;
  double rmsprop_acc = 0, rmsprop_ms = 0;
  MUST_TRUE(run(OptimizerType::OPTIMIZER_SGD, 0.05, sgd_acc, sgd_ms),
            "sgd run failed");
  MUST_TRUE(run(OptimizerType::OPTIMIZER_ADAM, 0.01, adam_acc, adam_ms),
            "adam run failed");
  MUST_TRUE(run(OptimizerType::OPTIMIZER_RMSPROP, 0.005, rmsprop_acc,
                rmsprop_ms),
            "rmsprop run failed");

  DEBUG("[SGD]      acc=" << sgd_acc << " time=" << sgd_ms << "ms");
  DEBUG("[Adam]     acc=" << adam_acc << " time=" << adam_ms << "ms");
  DEBUG("[RMSProp]  acc=" << rmsprop_acc << " time=" << rmsprop_ms << "ms");
}

// Adam BeforeStep 推进 step 计数, 验证内部状态正确
TEST(Optimizer, AdamStepCounterAdvances) {
  using namespace deeplearning;
  std::vector<int> layer = {2, 3, 2};
  AdamOptimizer opt(layer);
  MUST_EQUAL(opt.step(), 0);
  opt.BeforeStep();
  MUST_EQUAL(opt.step(), 1);
  opt.BeforeStep();
  opt.BeforeStep();
  MUST_EQUAL(opt.step(), 3);
}

// Adam vs AdamW 在相同条件下不应一致 (decoupled wd 让参数更新路径不同)
TEST(Optimizer, AdamAndAdamWDifferWithWeightDecay) {
  using namespace optimizer_test_detail;
  using namespace deeplearning;
  std::vector<std::vector<double>> tr, tr_t;
  MakeData(1, 200, tr, tr_t);

  auto train_one = [&](OptimizerType opt_type,
                       std::vector<std::vector<std::vector<double>>> &w_out) {
    NeuralNetwork net;
    MUST_EQUAL(BuildNet(net, 42, opt_type, 0.01), NeuralNetwork::SUCCESS);
    net.optimizer_function()->set_weight_decay(0.1); // 故意放大 wd
    auto rc = net.Train(tr, tr_t, nullptr, 200, 8);
    MUST_TRUE(rc == NeuralNetwork::SUCCESS, net.err_msg());
    w_out = net.neuron_weight();
  };

  std::vector<std::vector<std::vector<double>>> w_adam, w_adamw;
  train_one(OptimizerType::OPTIMIZER_ADAM, w_adam);
  train_one(OptimizerType::OPTIMIZER_ADAMW, w_adamw);

  double max_diff = 0.0;
  for (int l = 1; l < (int)w_adam.size(); l++) {
    for (int o = 0; o < (int)w_adam[l].size(); o++) {
      for (int i = 0; i < (int)w_adam[l][o].size(); i++) {
        double d = std::abs(w_adam[l][o][i] - w_adamw[l][o][i]);
        if (d > max_diff) {
          max_diff = d;
        }
      }
    }
  }
  DEBUG("max |w_adam - w_adamw| = " << max_diff);
  MUST_TRUE(max_diff > 1e-6, "Adam and AdamW must give different weights "
                              "when weight_decay > 0");
}
