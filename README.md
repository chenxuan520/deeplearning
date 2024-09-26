# deeplearning
- 简单的C++深度学习框架
## Author
-  **chenxuan**
## 项目结构
-  `src/deeplearning` 为所需的所有头文件,包含即可使用
-  `src/test` 为测试代码
## 使用demo
-  `src/demo` 中有demo代码,可以参考
    - mnist 为 mnist 数据集,使用代码demo默认配置下识别率约为91%
## Quick Start
1. `mkdir build;cmake ..;sudo make install` 安装
```c++
#include "deeplearning/neural_network.h"
using namespace deeplearning;

int main() {
  NeuralNetwork network((std::vector<int>() = {2, 1, 1}));
  std::vector<std::vector<double>> data = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  std::vector<std::vector<double>> target = {{0}, {0}, {1}, {1}};
  
  auto print_func = [](const NeuralNetwork &network, double loss_sum) {
    std::cout << loss_sum << std::endl;
  };
  auto rc = network.Train(data, target, print_func);
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
  return 0;
  std::cout << "Predict: " << result[0] << std::endl;
}
```
# TODO
- [ ] 支持多种优化器
