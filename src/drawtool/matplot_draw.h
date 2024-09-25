#pragma once

#include <string>
#include <vector>

#ifdef _MATPLOTLIB_CPP_LOAD_
#include "matplotlibcpp.h"
#endif

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
                              const std::string &y_label = "") {
#ifdef _MATPLOTLIB_CPP_LOAD_
    matplotlibcpp::named_plot("train", train_losses_x, train_losses_y);
    matplotlibcpp::named_plot("test", test_losses_x, test_losses_y);
    matplotlibcpp::title(title);
    matplotlibcpp::xlabel(x_label);
    matplotlibcpp::ylabel(y_label);
    matplotlibcpp::legend();
    matplotlibcpp::show();
#else
    return;
#endif
  }

  static void
  PrintWeightBar(const std::string &title,
                 const std::vector<std::vector<std::vector<double>>> &weights) {
#ifdef _MATPLOTLIB_CPP_LOAD_
    std::vector<double> weight_arr;
    for (auto weight : weights) {
      for (auto w : weight) {
        for (int i = 0; i < w.size(); i++) {
          weight_arr.push_back(w[i]);
        }
      }
    }
    matplotlibcpp::hist(weight_arr, 10);
    matplotlibcpp::title(title);
    matplotlibcpp::show();
#else
    return;
#endif
  }

  static void PrintBiasBar(const std::string &title,
                           const std::vector<std::vector<double>> &biases) {
    std::vector<double> bias_arr;
    for (auto bias : biases) {
      for (int i = 0; i < bias.size(); i++) {
        bias_arr.push_back(bias[i]);
      }
    }
#ifdef _MATPLOTLIB_CPP_LOAD_
    matplotlibcpp::hist(bias_arr, 10);
    matplotlibcpp::title(title);
    matplotlibcpp::show();
    matplotlibcpp::draw();
#else
    return;
#endif
  }

private:
};

} // namespace drawtool
