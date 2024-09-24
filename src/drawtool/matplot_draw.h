#pragma once

#include <string>
#include <vector>

#ifdef _MATPLOTLIB_CPP_LOAD_
#include "matplotlibcpp.h"
#endif

class MatplotDraw {
public:
  MatplotDraw() = delete;

  static void PrintLossResult(const std::string &title,
                              const std::vector<double> &train_losses,
                              const std::vector<double> &test_losses,
                              const std::string &x_label = "",
                              const std::string &y_label = "") {
#ifdef _MATPLOTLIB_CPP_LOAD_
    matplotlibcpp::plot(train_losses);
    matplotlibcpp::plot(test_losses);
    matplotlibcpp::title(title);
    matplotlibcpp::xlabel(x_label);
    matplotlibcpp::ylabel(y_label);
    matplotlibcpp::show();
#else
    return
#endif
  }

  static void PrintWeightBar(const std::string &title,
                             const std::vector<double> &weight_arr,
                             const std::string &x_label = "",
                             const std::string &y_label = "") {
#ifdef _MATPLOTLIB_CPP_LOAD_
    // TODO: make it work //
#else
    return
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
    matplotlibcpp::hist(bias_arr);
    matplotlibcpp::title(title);
    matplotlibcpp::show();
#else
    return
#endif
  }

private:
};
