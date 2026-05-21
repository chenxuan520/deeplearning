#pragma once

#include "lr_scheduler/cosine_annealing_lr.h"
#include "lr_scheduler/exponential_decay_lr.h"
#include "lr_scheduler/step_decay_lr.h"
#include "lr_scheduler/warmup_cosine_lr.h"
#include "neural_network.h"
#include "test.h"

#include <cmath>
#include <memory>

TEST(LRScheduler, StepDecayBasic) {
  using namespace deeplearning;
  StepDecayLR s(0.1, 10, 0.5);
  MUST_TRUE(std::abs(s.GetLR(0) - 0.1) < 1e-9, "step=0 -> base");
  MUST_TRUE(std::abs(s.GetLR(9) - 0.1) < 1e-9, "step=9 -> base");
  MUST_TRUE(std::abs(s.GetLR(10) - 0.05) < 1e-9, "step=10 -> base*0.5");
  MUST_TRUE(std::abs(s.GetLR(20) - 0.025) < 1e-9, "step=20 -> base*0.25");
}

TEST(LRScheduler, ExponentialDecayBasic) {
  using namespace deeplearning;
  ExponentialDecayLR s(1.0, 0.9);
  MUST_TRUE(std::abs(s.GetLR(0) - 1.0) < 1e-9, "step=0");
  MUST_TRUE(std::abs(s.GetLR(1) - 0.9) < 1e-9, "step=1");
  double expected_10 = std::pow(0.9, 10);
  MUST_TRUE(std::abs(s.GetLR(10) - expected_10) < 1e-9, "step=10");
}

TEST(LRScheduler, CosineAnnealingEnds) {
  using namespace deeplearning;
  CosineAnnealingLR s(1.0, 100, 0.0);
  MUST_TRUE(std::abs(s.GetLR(0) - 1.0) < 1e-9, "step=0 -> base");
  // 中点 = (base + min) / 2 = 0.5
  MUST_TRUE(std::abs(s.GetLR(50) - 0.5) < 1e-6, "step=mid -> avg");
  MUST_TRUE(std::abs(s.GetLR(100) - 0.0) < 1e-9, "step=tmax -> min");
  MUST_TRUE(std::abs(s.GetLR(200) - 0.0) < 1e-9, "step>tmax -> min");
}

TEST(LRScheduler, WarmupCosineRampsUp) {
  using namespace deeplearning;
  WarmupCosineLR s(1.0, 10, 110, 0.0);
  // step=0 warmup 第一步, 1/10 * 1.0 = 0.1
  MUST_TRUE(std::abs(s.GetLR(0) - 0.1) < 1e-9, "warmup step=0");
  MUST_TRUE(std::abs(s.GetLR(9) - 1.0) < 1e-9, "warmup last step");
  // cosine 部分: step=10 -> cos(0)=1 -> lr=base
  MUST_TRUE(std::abs(s.GetLR(10) - 1.0) < 1e-9, "cosine start");
  // step=110 -> end -> 0
  MUST_TRUE(std::abs(s.GetLR(110) - 0.0) < 1e-9, "cosine end");
}

// 端到端: 给 Train 装上 LRScheduler, 训练后 learning_rate_ 应该被
// scheduler 反复覆盖, 最终接近 scheduler 在最后一步的值.
TEST(LRScheduler, IntegrationWithTrain) {
  using namespace deeplearning;
  NeuralNetwork net;
  net.Init(std::vector<int>{2, 3, 2});
  net.set_param_init_function(ParamInitType::PARAM_INIT_XAVIER);

  auto sched = std::make_shared<ExponentialDecayLR>(0.1, 0.5);
  net.set_lr_scheduler(sched);

  std::vector<std::vector<double>> data = {
      {0.0, 0.0}, {1.0, 1.0}, {0.5, 0.5}, {0.2, 0.8}};
  std::vector<std::vector<double>> target = {
      {1.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {0.0, 1.0}};

  auto rc = net.Train(data, target, nullptr, 5, 2);
  MUST_EQUAL(rc, NeuralNetwork::SUCCESS);

  // 第 4 步 (从 0 数, i=4) lr = 0.1 * 0.5^4 = 0.00625
  double expected_last = 0.1 * std::pow(0.5, 4);
  MUST_TRUE(std::abs(net.learning_rate() - expected_last) < 1e-9,
            "lr should match scheduler at last step");
}
