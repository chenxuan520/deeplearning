#pragma once

#include <string>
#include <vector>

namespace drawtool {

class MatplotDraw {
public:
  MatplotDraw() = delete;

  static void PrintLossResult(const std::string &title,
                              const std::vector<double> &train_losses_x,
                              const std::vector<double> &train_losses_y,
                              const std::vector<double> &test_losses_x,
                              const std::vector<double> &test_losses_y,
                              const std::string &x_label = "",
                              const std::string &y_label = "");

  static void
  PrintWeightBar(const std::string &title,
                 const std::vector<std::vector<std::vector<double>>> &weights);

  static void PrintBiasBar(const std::string &title,
                           const std::vector<std::vector<double>> &biases);

private:
};

} // namespace drawtool
