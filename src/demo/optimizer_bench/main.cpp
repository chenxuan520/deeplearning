// optimizer_bench: 同一 MLP 结构下, 比较 SGD / Momentum / Adam / AdamW /
// RMSProp 在 MNIST 上的收敛速度和最终准确率.
//
// 用法:
//   ./bin/optimizer_bench [steps_per_optimizer] [batch_size]
// 默认 steps=8000, batch=32.
//
// 输出每个 optimizer 的:
//   - 训练总耗时 (wall-clock ms)
//   - 训练集 / 测试集 accuracy
//   - 关键 step 的 loss
//
// 通过 --skip-mnist 跑合成数据, 不依赖 MNIST 文件存在.

#include "lr_scheduler/cosine_annealing_lr.h"
#include "mnist_data.h"
#include "neural_network.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace std;
using namespace deeplearning;

namespace {

struct Dataset {
  vector<vector<double>> train_data;
  vector<vector<double>> train_target;
  vector<vector<double>> test_data;
  vector<vector<double>> test_target;
  int input_dim = 0;
  int output_dim = 0;
};

bool LoadMnist(Dataset &ds) {
  MnistData mnist;
  auto rc = mnist.LoadMnistData(
      const_cast<char *>("./demo/mnist/mnist/train-images-idx3-ubyte"),
      const_cast<char *>("./demo/mnist/mnist/train-labels-idx1-ubyte"),
      const_cast<char *>("./demo/mnist/mnist/t10k-images-idx3-ubyte"),
      const_cast<char *>("./demo/mnist/mnist/t10k-labels-idx1-ubyte"));
  if (rc != MnistData::SUCCESS) {
    cerr << "LoadMnistData failed: " << mnist.err_msg() << "\n";
    return false;
  }
  ds.train_data = mnist.train_data();
  ds.test_data = mnist.test_data();
  ds.input_dim = (int)ds.train_data[0].size();
  ds.output_dim = 10;
  ds.train_target.assign(ds.train_data.size(),
                         vector<double>(ds.output_dim, 0.0));
  for (size_t i = 0; i < mnist.train_labels().size(); i++) {
    ds.train_target[i][(int)mnist.train_labels()[i]] = 1.0;
  }
  ds.test_target.assign(ds.test_data.size(),
                        vector<double>(ds.output_dim, 0.0));
  for (size_t i = 0; i < mnist.test_labels().size(); i++) {
    ds.test_target[i][(int)mnist.test_labels()[i]] = 1.0;
  }
  return true;
}

void MakeSynthetic(Dataset &ds, int n_train = 4000, int n_test = 1000) {
  // 4 维输入 + 多类: 2 个高斯团做 4 类 (XOR-ish)
  std::mt19937 gen(42);
  std::normal_distribution<double> ng(0.0, 0.5);
  ds.input_dim = 4;
  ds.output_dim = 4;

  auto sample = [&](vector<vector<double>> &x,
                    vector<vector<double>> &y, int n) {
    x.assign(n, vector<double>(4, 0.0));
    y.assign(n, vector<double>(4, 0.0));
    for (int i = 0; i < n; i++) {
      int cls = i % 4;
      double cx = (cls & 1) ? 1.0 : -1.0;
      double cy = (cls & 2) ? 1.0 : -1.0;
      x[i] = {cx + ng(gen), cy + ng(gen), cx * cy + ng(gen), ng(gen)};
      y[i][cls] = 1.0;
    }
  };
  sample(ds.train_data, ds.train_target, n_train);
  sample(ds.test_data, ds.test_target, n_test);
}

double Accuracy(NeuralNetwork &net,
                const vector<vector<double>> &x,
                const vector<vector<double>> &y) {
  vector<vector<double>> pred;
  auto rc = net.PredictBatch(x, pred);
  if (rc != NeuralNetwork::SUCCESS) {
    return -1;
  }
  int correct = 0;
  for (size_t i = 0; i < pred.size(); i++) {
    int pi = (int)(std::max_element(pred[i].begin(), pred[i].end()) -
                   pred[i].begin());
    int ti = (int)(std::max_element(y[i].begin(), y[i].end()) - y[i].begin());
    if (pi == ti) {
      correct++;
    }
  }
  return correct * 1.0 / (double)pred.size();
}

struct RunConfig {
  string name;
  OptimizerType type;
  double lr;
  double weight_decay = 0.0;
  bool use_cosine = false;
};

struct RunResult {
  string name;
  double ms;
  double train_acc;
  double test_acc;
  vector<pair<int, double>> loss_trace;
};

RunResult RunOne(const Dataset &ds, const RunConfig &cfg, int steps,
                 int batch) {
  using clk = chrono::high_resolution_clock;
  NeuralNetwork net;
  net.Init(vector<int>{ds.input_dim, 64, 32, ds.output_dim});
  net.set_random_seed(123);
  net.set_param_init_function(ParamInitType::PARAM_INIT_HE);
  net.set_loss_function(LossType::LOSS_CROSS_ENTROPY);
  net.set_softmax_function(SoftmaxType::SOFTMAX_STD);
  net.set_activate_function(ActivateType::ACTIVATE_RELU);
  net.set_optimizer_function(cfg.type);
  net.optimizer_function()->set_weight_decay(cfg.weight_decay);
  net.set_learning_rate(cfg.lr);

  if (cfg.use_cosine) {
    net.set_lr_scheduler(
        std::make_shared<CosineAnnealingLR>(cfg.lr, steps, cfg.lr * 0.01));
  }

  RunResult r;
  r.name = cfg.name;

  // 用回调收集 loss 曲线 (粗略, 不要每步都算 — 太慢)
  int log_every = max(1, steps / 8);
  auto cb = [&](NeuralNetwork &n, int step, bool &) {
    if (step % log_every != 0 && step != steps - 1) {
      return;
    }
    // 评 200 个样本快速过一下 loss, 不全量算 (省时)
    int probe = min<int>(200, ds.train_data.size());
    vector<vector<double>> sub_x(ds.train_data.begin(),
                                  ds.train_data.begin() + probe);
    vector<vector<double>> sub_y(ds.train_target.begin(),
                                  ds.train_target.begin() + probe);
    double loss = 0;
    n.CalcLoss(sub_x, sub_y, loss);
    r.loss_trace.push_back({step, loss});
  };

  auto t0 = clk::now();
  auto rc = net.Train(ds.train_data, ds.train_target, cb, steps, batch);
  auto t1 = clk::now();
  if (rc != NeuralNetwork::SUCCESS) {
    cerr << "  Train failed: " << net.err_msg() << "\n";
    r.ms = -1;
    return r;
  }
  r.ms = chrono::duration<double, milli>(t1 - t0).count();
  r.train_acc = Accuracy(net, ds.train_data, ds.train_target);
  r.test_acc = Accuracy(net, ds.test_data, ds.test_target);
  return r;
}

void PrintTable(const vector<RunResult> &runs) {
  cout << "\n=== Summary ===\n";
  cout << left << setw(22) << "optimizer" << right << setw(12) << "time(ms)"
       << setw(14) << "train_acc" << setw(14) << "test_acc" << "\n";
  cout << string(62, '-') << "\n";
  for (auto &r : runs) {
    cout << left << setw(22) << r.name << right << setw(12) << fixed
         << setprecision(1) << r.ms << setw(14) << setprecision(4)
         << r.train_acc << setw(14) << r.test_acc << "\n";
  }
  cout << "\n=== Loss traces ===\n";
  for (auto &r : runs) {
    cout << r.name << ": ";
    for (auto &p : r.loss_trace) {
      cout << "step=" << p.first << " loss=" << fixed << setprecision(3)
           << p.second << "  ";
    }
    cout << "\n";
  }
}

} // namespace

int main(int argc, char **argv) {
  int steps = 8000;
  int batch = 32;
  bool skip_mnist = false;

  for (int i = 1; i < argc; i++) {
    string a = argv[i];
    if (a == "--skip-mnist") {
      skip_mnist = true;
    } else if (a == "--steps" && i + 1 < argc) {
      steps = atoi(argv[++i]);
    } else if (a == "--batch" && i + 1 < argc) {
      batch = atoi(argv[++i]);
    } else if (a == "-h" || a == "--help") {
      cout << "optimizer_bench [--steps N] [--batch N] [--skip-mnist]\n";
      return 0;
    }
  }

  Dataset ds;
  if (!skip_mnist && LoadMnist(ds)) {
    cout << "Loaded MNIST: train=" << ds.train_data.size()
         << " test=" << ds.test_data.size() << "\n";
  } else {
    cout << "Using synthetic 4-class dataset (--skip-mnist or MNIST missing)\n";
    MakeSynthetic(ds);
  }
  cout << "input_dim=" << ds.input_dim << " output_dim=" << ds.output_dim
       << " steps=" << steps << " batch=" << batch << "\n\n";

  vector<RunConfig> configs = {
      {"SGD lr=0.05", OptimizerType::OPTIMIZER_SGD, 0.05},
      {"Momentum lr=0.05", OptimizerType::OPTIMIZER_MOMENTUM, 0.05},
      {"RMSProp lr=0.001", OptimizerType::OPTIMIZER_RMSPROP, 0.001},
      {"Adam lr=0.001", OptimizerType::OPTIMIZER_ADAM, 0.001},
      {"AdamW lr=0.001 wd=1e-4", OptimizerType::OPTIMIZER_ADAMW, 0.001, 1e-4},
      {"Adam+CosineLR lr=0.001", OptimizerType::OPTIMIZER_ADAM, 0.001, 0.0,
       true},
  };

  vector<RunResult> runs;
  for (auto &c : configs) {
    cout << "Training [" << c.name << "] ..." << flush;
    auto r = RunOne(ds, c, steps, batch);
    cout << " done (" << fixed << setprecision(1) << r.ms << "ms, test_acc="
         << setprecision(4) << r.test_acc << ")\n";
    runs.push_back(r);
  }

  PrintTable(runs);
  return 0;
}
