#pragma once

#include "activate/activate_factory.h"
#include "activate/gelu_activate.h"
#include "activate/leaky_relu_activate.h"
#include "neural_network.h"
#include "test.h"

#include <cmath>
#include <cstdlib>
#include <vector>

namespace activate_test_detail {

using namespace deeplearning;

inline void
MakeData(int seed, int n,
         std::vector<std::vector<double>> &data,
         std::vector<std::vector<double>> &target) {
  std::srand(seed);
  data.clear();
  target.clear();
  for (int i = 0; i < n; i++) {
    double x = (std::rand() % 1000) / 500.0 - 1.0;
    double y = (std::rand() % 1000) / 500.0 - 1.0;
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
    network.Predict(data[i], r);
    int pred = (r[0] > r[1]) ? 0 : 1;
    int truth = (target[i][0] > target[i][1]) ? 0 : 1;
    if (pred == truth) {
      correct++;
    }
  }
  return correct * 1.0 / (int)data.size();
}

inline NeuralNetwork::RC
BuildNet(NeuralNetwork &net, ActivateType act, int seed) {
  auto rc = net.Init(std::vector<int>{2, 16, 16, 2});
  if (rc != NeuralNetwork::SUCCESS) {
    return rc;
  }
  net.set_random_seed(seed);
  rc = net.set_param_init_function(ParamInitType::PARAM_INIT_HE);
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
  rc = net.set_activate_function(act);
  if (rc != NeuralNetwork::SUCCESS) {
    return rc;
  }
  rc = net.set_optimizer_function(OptimizerType::OPTIMIZER_ADAM);
  if (rc != NeuralNetwork::SUCCESS) {
    return rc;
  }
  net.set_learning_rate(0.005);
  return NeuralNetwork::SUCCESS;
}

} // namespace activate_test_detail

TEST(Activate, LeakyReluAtSpecificPoints) {
  using namespace deeplearning;
  LeakyReluActivate act;
  MUST_TRUE(std::abs(act.Activate(2.0) - 2.0) < 1e-12, "leaky relu(+)");
  MUST_TRUE(std::abs(act.Activate(-2.0) - (-0.02)) < 1e-12, "leaky relu(-)");
  // 兼容旧接口: DerivActivate(output) — output=正 -> 1, output=负 -> slope
  MUST_TRUE(std::abs(act.DerivActivate(2.0) - 1.0) < 1e-12, "deriv positive");
  MUST_TRUE(std::abs(act.DerivActivate(-0.02) - 0.01) < 1e-12, "deriv negative");
}

TEST(Activate, GeluMatchesKnownValues) {
  using namespace deeplearning;
  GeluActivate act;
  // 已知精确 GELU 值 (用 erf 算):
  // GELU(0)   = 0
  // GELU(1)   ≈ 0.8413447
  // GELU(-1)  ≈ -0.1586553
  MUST_TRUE(std::abs(act.Activate(0.0)) < 1e-9, "gelu(0)=0");
  MUST_TRUE(std::abs(act.Activate(1.0) - 0.8413447460685429) < 1e-6,
            "gelu(1)");
  MUST_TRUE(std::abs(act.Activate(-1.0) - (-0.15865525393145707)) < 1e-6,
            "gelu(-1)");
  // 大 input 时 GELU(x) ≈ x
  MUST_TRUE(std::abs(act.Activate(10.0) - 10.0) < 1e-6, "gelu(+large)≈x");
  MUST_TRUE(std::abs(act.Activate(-10.0)) < 1e-6, "gelu(-large)≈0");
}

TEST(Activate, GeluDerivativeFromInputMatchesNumerical) {
  using namespace deeplearning;
  GeluActivate act;
  // 数值导数 (中心差分) vs 解析:
  for (double x : {-2.0, -0.5, 0.0, 0.3, 1.0, 2.5}) {
    double eps = 1e-5;
    double num = (act.Activate(x + eps) - act.Activate(x - eps)) / (2 * eps);
    double y = act.Activate(x);
    double ana = act.DerivActivate(x, y); // 2-arg version
    MUST_TRUE(std::abs(num - ana) < 1e-4,
              "analytical deriv should match numerical");
  }
}

TEST(Activate, FactoryProducesLeakyReluAndGelu) {
  using namespace deeplearning;
  auto leaky = ActivateFactory::Create(ACTIVATE_LEAKY_RELU);
  auto gelu = ActivateFactory::Create(ACTIVATE_GELU);
  MUST_TRUE(leaky != nullptr, "factory creates leaky relu");
  MUST_TRUE(gelu != nullptr, "factory creates gelu");
  MUST_EQUAL(leaky->GetActivateType(), ACTIVATE_LEAKY_RELU);
  MUST_EQUAL(gelu->GetActivateType(), ACTIVATE_GELU);
}

TEST(Activate, NetworkWithGeluConverges) {
  using namespace activate_test_detail;
  using namespace deeplearning;

  std::vector<std::vector<double>> tr, tr_t, te, te_t;
  MakeData(1, 2000, tr, tr_t);
  MakeData(2, 500, te, te_t);

  NeuralNetwork net;
  MUST_EQUAL(BuildNet(net, ActivateType::ACTIVATE_GELU, 1),
             NeuralNetwork::SUCCESS);
  auto rc = net.Train(tr, tr_t, nullptr, 3000, 16);
  MUST_TRUE(rc == NeuralNetwork::SUCCESS, net.err_msg());

  double acc = Accuracy(net, te, te_t);
  DEBUG("GELU+Adam acc=" << acc);
  MUST_TRUE(acc > 0.82, "GELU activation should learn this dataset");
}

TEST(Activate, NetworkWithLeakyReluConverges) {
  using namespace activate_test_detail;
  using namespace deeplearning;

  std::vector<std::vector<double>> tr, tr_t, te, te_t;
  MakeData(1, 2000, tr, tr_t);
  MakeData(2, 500, te, te_t);

  NeuralNetwork net;
  MUST_EQUAL(BuildNet(net, ActivateType::ACTIVATE_LEAKY_RELU, 1),
             NeuralNetwork::SUCCESS);
  auto rc = net.Train(tr, tr_t, nullptr, 3000, 16);
  MUST_TRUE(rc == NeuralNetwork::SUCCESS, net.err_msg());

  double acc = Accuracy(net, te, te_t);
  DEBUG("LeakyReLU+Adam acc=" << acc);
  MUST_TRUE(acc > 0.82, "LeakyReLU activation should learn this dataset");
}
