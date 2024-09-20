#pragma once

#include "neural_network.h"
#include "test.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <utility>
#include <vector>

using namespace std;
using namespace deeplearning;

std::string demo_data_file_path;
std::vector<std::vector<double>> demo_data;
std::vector<std::vector<double>> demo_data_target;
std::string demo_test_file_path;
std::vector<std::vector<double>> demo_test;
std::vector<std::vector<double>> demo_test_target;

INIT(NeuralNetwork) {
  srand(time(nullptr));
  demo_data_file_path = "demo.data";
  demo_test_file_path = "demo.test";
  const int train_data_size = 30000;
  const int test_data_size = 1000;
  // create test data, example func y=3*e^(x/10)-x^2-x-ln(x^2+1)+2
  // use random data x from -100.00 to 100.00(double)
  // y is random ,compare with y=3*e^(x/10)-x^2-x-ln(x^2+1)+2
  // create 10000 point
  auto func_calc = [](double x) -> double {
    return 2 * exp(double(x) / 10) - x * x - x - log(x * x + 1) + 2;
    // return 3 * x + 2;
  };

  // func to create data
  auto decimal_point_create = [](int precision) -> double {
    double decimal_point = 0;
    for (int i = 0; i < precision; i++) {
      decimal_point += (rand() % 10) * pow(0.1, i);
    }
    return decimal_point;
  };
  auto func_y_create = [&](const std::pair<int, int> &range) -> double {
    return rand() % (range.second - range.first) + range.first +
           decimal_point_create(3);
  };
  auto func_x_create = [&](const std::pair<int, int> &range) -> double {
    return rand() % (range.second - range.first) + range.first +
           decimal_point_create(3);
  };

  // make x,y range
  std::pair<int, int> y_range = {INT32_MAX, INT16_MIN};
  std::pair<int, int> x_range = {-100, 100};
  for (int i = x_range.first; i < x_range.second; i++) {
    double tmp_y = func_calc(i);
    if (tmp_y > y_range.second) {
      y_range.second = tmp_y;
    }
    if (tmp_y < y_range.first) {
      y_range.first = tmp_y;
    }
  }

  std::ofstream file(demo_data_file_path);
  for (int i = 0; i < train_data_size; i++) {
    double x = func_x_create(x_range);
    double stand_y = func_calc(x);
    double y = func_y_create(y_range);
    demo_data.push_back({x, y});
    demo_data_target.push_back({stand_y > y ? 1.0 : 0.0});

    file << std::fixed << std::setprecision(4) << x << " " << y << " "
         << stand_y << " " << (stand_y > y ? 1.0 : 0.0) << std::endl;
  }
  file.close();

  std::ofstream test_file(demo_test_file_path);
  for (int i = 0; i < test_data_size; i++) {
    double x = func_x_create(x_range);
    double tmp_y = func_calc(x);
    double y = func_y_create(y_range);
    demo_test.push_back({x, y});
    demo_test_target.push_back({tmp_y > y ? 1.0 : 0.0});

    test_file << std::fixed << std::setprecision(4) << x << " " << y << " "
              << tmp_y << " " << (tmp_y > y ? 1.0 : 0.0) << std::endl;
  }
  test_file.close();
};

END(NeuralNetwork) {
  // clean file
  remove(demo_data_file_path.c_str());
  remove(demo_test_file_path.c_str());
}

TEST(NeuralNetwork, TrainAndPredict) {
  NeuralNetwork demo_network;
  demo_network.Init((vector<int>() = {2, 3, 3, 1}), 0.01);
  MUST_EQUAL(demo_network.network_status(), NeuralNetwork::NETWORK_STATUS_INIT);
  auto rc = demo_network.set_loss_function(LossType::LOSS_CROSS_ENTROPY);
  MUST_EQUAL(rc, NeuralNetwork::SUCCESS);

  auto print_func = [](NeuralNetwork &network, int epoch_num) {
    static int count = 0;
    if (count++ % 100 == 0) {
      double train_loss = 0;
      auto rc = network.CalcLoss(demo_data, demo_data_target, train_loss);
      if (rc != NeuralNetwork::SUCCESS) {
        cout << "CalcLoss failed: " << network.err_msg() << endl;
        return;
      }
      double test_loss = 0;
      rc = network.CalcLoss(demo_test, demo_test_target, test_loss);
      if (rc != NeuralNetwork::SUCCESS) {
        cout << "CalcLoss failed: " << network.err_msg() << endl;
        return;
      }
      std::cout << "epoch: " << epoch_num << " train_loss: " << train_loss
                << " test_loss: " << test_loss << std::endl;
    }
  };
  rc = demo_network.Train(demo_data, demo_data_target, print_func);
  MUST_TRUE(rc == NeuralNetwork::SUCCESS, demo_network.err_msg());

  // calc right rate
  int right_count = 0;
  for (int i = 0; i < demo_test.size(); i++) {
    std::vector<double> result;
    rc = demo_network.Predict(demo_test[i], result);
    if (rc != NeuralNetwork::SUCCESS) {
      MUST_EQUAL(rc, NeuralNetwork::SUCCESS);
    }
    if (result[0] > 0.5 && demo_test_target[i][0] > 0.5) {
      right_count++;
    } else if (result[0] < 0.5 && demo_test_target[i][0] < 0.5) {
      right_count++;
    }
  }
  DEBUG("right rate: " << right_count * 1.0 / demo_test.size());
  MUST_TRUE(right_count * 1.0 / demo_test.size() > 0.8,
            "train loss is too high");
}
