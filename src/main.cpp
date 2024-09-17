#include "deeplearning/neural_network.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
using namespace deeplearning;

int main() {
  NeuralNetwork network((std::vector<int>() = {2, 1, 1}), 0.2);
  std::vector<std::vector<double>> data = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  std::vector<std::vector<double>> target = {{0}, {0}, {1}, {1}};

  auto print_func = [](const NeuralNetwork &network, double loss_sum) {
    std::cout << loss_sum << std::endl;
  };
  auto rc = network.Train(data, target, 2000, 1, print_func);
  if (rc != NeuralNetwork::SUCCESS) {
    std::cout << "Train failed" << std::endl;
    return -1;
  }
  std::cout << "Train success" << std::endl;
  std::vector<double> test_data = {1, 1.2}, result;
  rc = network.Predict(test_data, result);
  if (rc != NeuralNetwork::SUCCESS) {
    std::cout << "Predict failed" << std::endl;
    return -1;
  }
  std::cout << "Predict: " << result[0] << std::endl;
  return 0;
}
