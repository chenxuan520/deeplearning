#pragma once

#include "../deeplearning/neural_network.h"
#include "test.h"
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
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
  // create test data, example func y=3*e^(x/10)-x^2-x+2
  // use random data x from -100.00 to 100.00(double)
  // y is random ,compare with y=3*e^(x/10)-x^2-x+2
  // create 10000 point
  auto func_calc = [](double x) -> double {
    return 2 * exp(double(x) / 10) - x * x - x + 2;
    // return 30 * x + 2;
  };
  auto func_y_create = [](double target_y) -> double {
    return target_y + (-1 * (rand() % 2 == 0 ? 1 : -1)) * (rand() % 10000) +
           (rand() % 100) / 100.0;
  };

  std::ofstream file(demo_data_file_path);
  for (int i = 0; i < 10000; i++) {
    double x = rand() % 200 - 100 + (rand() % 100) / 100.0;
    double tmp_y = func_calc(x);
    double y = func_y_create(tmp_y);
    demo_data.push_back({x, y});
    demo_data_target.push_back({tmp_y > y ? 1.0 : 0.0});

    file << std::fixed << std::setprecision(4) << x << " " << y << " " << tmp_y
         << " " << (tmp_y > y ? 1.0 : 0.0) << std::endl;
  }
  file.close();

  std::ofstream test_file(demo_test_file_path);
  for (int i = 0; i < 1000; i++) {
    double x = rand() % 200 - 100 + (rand() % 100) / 100.0;
    double tmp_y = func_calc(x);
    double y = func_y_create(tmp_y);
    demo_test.push_back({x, y});
    demo_test_target.push_back({tmp_y > y ? 1.0 : 0.0});

    test_file << std::fixed << std::setprecision(4) << x << " " << y << " "
              << tmp_y << " " << (tmp_y > y ? 1.0 : 0.0) << std::endl;
  }
  test_file.close();
};

END(NeuralNetwork) {
  // clean file
  // remove(demo_data_file_path.c_str());
  // remove(demo_test_file_path.c_str());
}

TEST(NeuralNetwork, TrainAndPredict) {
  NeuralNetwork demo_network;
  demo_network.Init((vector<int>() = {2, 3, 3, 1}), 0.001);
  MUST_EQUAL(demo_network.network_status(), NeuralNetwork::NETWORK_STATUS_INIT);

  auto print_func = [](const NeuralNetwork &network, double loss_sum) {
    std::cout << "loss: " << loss_sum << std::endl;
  };
  auto rc = demo_network.Train(demo_data, demo_data_target, 100, 1, print_func);
  MUST_TRUE(rc == NeuralNetwork::SUCCESS, demo_network.err_msg());

  std::cout << "Train success" << std::endl;

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
