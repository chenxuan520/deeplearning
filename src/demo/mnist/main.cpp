#include "lr_scheduler/warmup_cosine_lr.h"
#include "matplot_draw.h"
#include "mnist_data.h"
#include "neural_network.h"
#include "neural_network_loader.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

using namespace std;
using namespace deeplearning;
using namespace drawtool;

namespace {

// 用 PredictBatch 一次性前向, 比逐样本 Predict 快很多.
double Accuracy(NeuralNetwork &net, const vector<vector<double>> &x,
                const vector<int> &labels) {
  vector<vector<double>> pred;
  auto rc = net.PredictBatch(x, pred);
  if (rc != NeuralNetwork::SUCCESS) {
    return -1.0;
  }
  int correct = 0;
  for (size_t i = 0; i < pred.size(); i++) {
    int pi = (int)(std::max_element(pred[i].begin(), pred[i].end()) -
                   pred[i].begin());
    if (pi == labels[i]) {
      correct++;
    }
  }
  return correct * 1.0 / (double)pred.size();
}

} // namespace

int main() {
  // ---- 1. 加载数据 (像素已在 mnist_data.h 内归一化到 [0,1]) ----
  char train_image_name[] = "./demo/mnist/mnist/train-images-idx3-ubyte";
  char train_label_name[] = "./demo/mnist/mnist/train-labels-idx1-ubyte";
  char test_image_name[] = "./demo/mnist/mnist/t10k-images-idx3-ubyte";
  char test_label_name[] = "./demo/mnist/mnist/t10k-labels-idx1-ubyte";

  MnistData mnist_data;
  auto rcMnist = mnist_data.LoadMnistData(train_image_name, train_label_name,
                                          test_image_name, test_label_name);
  if (rcMnist != MnistData::SUCCESS) {
    cout << "LoadMnistData failed: " << mnist_data.err_msg() << endl;
    return -1;
  }
  int N_train = (int)mnist_data.train_data().size();
  int N_test = (int)mnist_data.test_data().size();
  cout << "train_data: " << N_train << "  test_data: " << N_test << endl;

  // one-hot target
  vector<vector<double>> train_target(N_train, vector<double>(10, 0.0));
  for (int i = 0; i < N_train; i++) {
    train_target[i][mnist_data.train_labels()[i]] = 1.0;
  }
  vector<vector<double>> test_target(N_test, vector<double>(10, 0.0));
  for (int i = 0; i < N_test; i++) {
    test_target[i][mnist_data.test_labels()[i]] = 1.0;
  }

  // ---- 2. 构造或加载网络 ----
  // 网络结构: 784 -> 128 -> 64 -> 10
  // 损失/激活/softmax: cross-entropy + ReLU + softmax (分类标配)
  // 初始化: He (配 ReLU)
  // 优化器: Adam
  NeuralNetwork demo_network;
  std::string param_file_name = "./demo/mnist/mnist/demo.v2.param";

  NeuralNetwork::NetworkParam demo_param = {};
  NeuralNetwork::NetworkOption demo_option = {};
  auto loader_rc = NeuralNetworkLoader::ImportParamFromFile(
      demo_param, demo_option, param_file_name);

  double base_lr = 1e-3;
  if (loader_rc == NeuralNetworkLoader::SUCCESS) {
    cout << "Loaded existing model from " << param_file_name << endl;
    auto rc = demo_network.ImportNetworkParam(demo_param, demo_option);
    if (rc != NeuralNetwork::SUCCESS) {
      cout << "ImportNetworkParam failed: " << demo_network.err_msg() << endl;
      return -1;
    }
    base_lr = 1e-4; // 续训用更小 lr 微调
  } else {
    cout << "Init new model: 784 -> 128 -> 64 -> 10" << endl;
    auto rc = demo_network.Init(vector<int>{784, 128, 64, 10});
    if (rc != NeuralNetwork::SUCCESS) {
      cout << "Init failed: " << demo_network.err_msg() << endl;
      return -1;
    }
    demo_network.set_random_seed(42);
    demo_network.set_param_init_function(ParamInitType::PARAM_INIT_HE);
    demo_network.set_activate_function(ActivateType::ACTIVATE_RELU);
    demo_network.set_softmax_function(SoftmaxType::SOFTMAX_STD);
    demo_network.set_loss_function(LossType::LOSS_CROSS_ENTROPY);
    demo_network.set_optimizer_function(OptimizerType::OPTIMIZER_ADAM);
  }

  // ---- 3. 训练配置 + WarmupCosine 调度 ----
  int batch_size = 64;
  int epochs = 5;
  int steps_per_epoch = (N_train + batch_size - 1) / batch_size;
  int total_steps = epochs * steps_per_epoch;
  int warmup_steps = steps_per_epoch; // 1 个 epoch 做 warmup
  double min_lr = base_lr * 0.01;
  demo_network.set_lr_scheduler(std::make_shared<WarmupCosineLR>(
      base_lr, warmup_steps, total_steps, min_lr));

  cout << "Train: batch=" << batch_size << " epochs=" << epochs
       << " steps_per_epoch=" << steps_per_epoch
       << " total_steps=" << total_steps << " base_lr=" << base_lr
       << " (warmup-cosine)" << endl;

  // ---- 4. 训练 + 每 epoch 末评估 ----
  vector<double> train_loss_y, test_loss_y, train_loss_x, test_loss_x;

  auto t0 = chrono::high_resolution_clock::now();

  auto each_step = [&](NeuralNetwork &net, int step, bool &) {
    int next_step = step + 1;
    if (next_step % steps_per_epoch != 0 && next_step != total_steps) {
      return;
    }
    int ep = next_step / steps_per_epoch;
    // train loss 只抽样 5000 个估计 (60000 太慢, CalcLoss 是单样本前向)
    vector<vector<double>> probe_x(mnist_data.train_data().begin(),
                                    mnist_data.train_data().begin() + 5000);
    vector<vector<double>> probe_y(train_target.begin(),
                                    train_target.begin() + 5000);
    double train_loss = 0, test_loss = 0;
    net.CalcLoss(probe_x, probe_y, train_loss);
    net.CalcLoss(mnist_data.test_data(), test_target, test_loss);
    double acc =
        Accuracy(net, mnist_data.test_data(), mnist_data.test_labels());
    train_loss_y.push_back(train_loss);
    test_loss_y.push_back(test_loss);
    train_loss_x.push_back(ep);
    test_loss_x.push_back(ep);
    cout << "epoch " << ep << "/" << epochs << "  train_loss=" << fixed
         << setprecision(4) << train_loss << "  test_loss=" << test_loss
         << "  test_acc=" << acc << "  lr=" << scientific << setprecision(2)
         << net.learning_rate() << fixed << endl;
  };

  auto rc = demo_network.Train(mnist_data.train_data(), train_target,
                                each_step, total_steps, batch_size, base_lr);
  if (rc != NeuralNetwork::SUCCESS) {
    cout << "Train failed: " << demo_network.err_msg() << endl;
    return -1;
  }

  auto t1 = chrono::high_resolution_clock::now();
  double elapsed_sec =
      chrono::duration<double>(t1 - t0).count();
  cout << "Train finished in " << fixed << setprecision(1) << elapsed_sec
       << " sec" << endl;

  // ---- 5. 最终评估 ----
  double train_acc = Accuracy(demo_network, mnist_data.train_data(),
                              mnist_data.train_labels());
  double test_acc = Accuracy(demo_network, mnist_data.test_data(),
                              mnist_data.test_labels());
  cout << "Final accuracy:  train=" << setprecision(4) << train_acc
       << "  test=" << test_acc << endl;

  MatplotDraw::PrintLossResult("Mnist NeuralNetwork v2", train_loss_x,
                                train_loss_y, test_loss_x, test_loss_y,
                                "epoch", "loss");

  if (test_acc < 0.95) {
    cout << "WARN: test acc < 95%" << endl;
  }

  // ---- 6. 保存参数 ----
  NeuralNetwork::NetworkParam param;
  NeuralNetwork::NetworkOption option;
  rc = demo_network.ExportNetworkParam(param, option);
  if (rc != NeuralNetwork::SUCCESS) {
    cout << "ExportNetworkParam failed: " << demo_network.err_msg() << endl;
    return -1;
  }
  auto rcLoader =
      NeuralNetworkLoader::ExportParamToFile(param, option, param_file_name);
  if (rcLoader != NeuralNetworkLoader::SUCCESS) {
    cout << "ExportParamToFile failed: " << demo_network.err_msg() << endl;
    return -1;
  }
  cout << "Saved model to " << param_file_name << endl;

  return 0;
}
