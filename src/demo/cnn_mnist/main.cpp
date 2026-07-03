#include "../mnist/mnist_data.h"
#include "cnn/mini_cnn_classifier.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace deeplearning;

namespace {

struct DemoOption {
  int epoch_num = 3;
  int train_limit = 2000;
  int test_limit = 1000;
  int rand_seed = 0;
  int conv_channels = 8;
  int kernel_size = 5;
  double learning_rate = 0.01;
};

void PrintUsage(const char *prog) {
  cout << "Usage: " << prog << " [options]\n"
       << "  --epochs <int>\n"
       << "  --learning-rate <double>\n"
       << "  --train-limit <int>\n"
       << "  --test-limit <int>\n"
       << "  --rand-seed <int>\n"
       << "  --conv-channels <int>\n"
       << "  --kernel-size <int>\n"
       << "  --help\n";
}

bool ParseArgs(int argc, char **argv, DemoOption &option) {
  for (int i = 1; i < argc; i++) {
    string arg = argv[i];
    auto need_value = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        throw std::runtime_error(string("Missing value for ") + name);
      }
      i += 1;
      return argv[i];
    };

    if (arg == "--epochs") {
      option.epoch_num = std::stoi(need_value("--epochs"));
    } else if (arg == "--learning-rate") {
      option.learning_rate = std::stod(need_value("--learning-rate"));
    } else if (arg == "--train-limit") {
      option.train_limit = std::stoi(need_value("--train-limit"));
    } else if (arg == "--test-limit") {
      option.test_limit = std::stoi(need_value("--test-limit"));
    } else if (arg == "--rand-seed") {
      option.rand_seed = std::stoi(need_value("--rand-seed"));
    } else if (arg == "--conv-channels") {
      option.conv_channels = std::stoi(need_value("--conv-channels"));
    } else if (arg == "--kernel-size") {
      option.kernel_size = std::stoi(need_value("--kernel-size"));
    } else if (arg == "--help") {
      return false;
    } else {
      throw std::runtime_error(string("Unknown option: ") + arg);
    }
  }
  return true;
}

MiniCNNClassifier::Tensor3D ToImageTensor(const vector<double> &image) {
  MiniCNNClassifier::Tensor3D tensor(
      1, vector<vector<double>>(28, vector<double>(28, 0.0)));
  for (int row = 0; row < 28; row++) {
    for (int col = 0; col < 28; col++) {
      tensor[0][row][col] = image[row * 28 + col];
    }
  }
  return tensor;
}

void BuildDataset(const vector<vector<double>> &flat_images,
                  const vector<int> &flat_labels, int limit,
                  vector<MiniCNNClassifier::Tensor3D> &images,
                  vector<int> &labels) {
  int count = limit <= 0 ? static_cast<int>(flat_images.size())
                         : std::min(limit, static_cast<int>(flat_images.size()));
  images.clear();
  labels.clear();
  images.reserve(count);
  labels.reserve(count);
  for (int i = 0; i < count; i++) {
    images.push_back(ToImageTensor(flat_images[i]));
    labels.push_back(flat_labels[i]);
  }
}

double Accuracy(MiniCNNClassifier &model,
                const vector<MiniCNNClassifier::Tensor3D> &images,
                const vector<int> &labels) {
  if (images.empty()) {
    return 0.0;
  }
  int correct = 0;
  for (int i = 0; i < static_cast<int>(images.size()); i++) {
    int pred = -1;
    auto rc = model.Predict(images[i], pred);
    if (rc != MiniCNNClassifier::SUCCESS) {
      return -1.0;
    }
    if (pred == labels[i]) {
      correct++;
    }
  }
  return correct * 1.0 / images.size();
}

} // namespace

int main(int argc, char **argv) {
  DemoOption option;
  try {
    if (!ParseArgs(argc, argv, option)) {
      PrintUsage(argv[0]);
      return 0;
    }
  } catch (const std::exception &ex) {
    cout << ex.what() << endl;
    PrintUsage(argv[0]);
    return -1;
  }

  if (option.epoch_num <= 0 || option.learning_rate <= 0.0 ||
      option.conv_channels <= 0 || option.kernel_size <= 0 ||
      option.kernel_size % 2 == 0) {
    cout << "Invalid option: epochs/lr/conv-channels must be > 0, kernel-size "
            "must be positive odd"
         << endl;
    return -1;
  }

  MnistData mnist_data;
  auto mnist_rc = mnist_data.LoadMnistData(
      "./demo/mnist/mnist/train-images-idx3-ubyte",
      "./demo/mnist/mnist/train-labels-idx1-ubyte",
      "./demo/mnist/mnist/t10k-images-idx3-ubyte",
      "./demo/mnist/mnist/t10k-labels-idx1-ubyte");
  if (mnist_rc != MnistData::SUCCESS) {
    cout << "LoadMnistData failed: " << mnist_data.err_msg() << endl;
    return -1;
  }

  vector<MiniCNNClassifier::Tensor3D> train_images, test_images;
  vector<int> train_labels, test_labels;
  BuildDataset(mnist_data.train_data(), mnist_data.train_labels(), option.train_limit,
               train_images, train_labels);
  BuildDataset(mnist_data.test_data(), mnist_data.test_labels(), option.test_limit,
               test_images, test_labels);

  MiniCNNClassifier model;
  MiniCNNClassifier::Config config;
  config.input_channels_ = 1;
  config.input_height_ = 28;
  config.input_width_ = 28;
  config.conv_channels_ = option.conv_channels;
  config.kernel_height_ = option.kernel_size;
  config.kernel_width_ = option.kernel_size;
  config.conv_stride_ = 1;
  config.conv_padding_ = option.kernel_size / 2;
  config.pool_height_ = 2;
  config.pool_width_ = 2;
  config.pool_stride_ = 2;
  config.class_num_ = 10;
  config.rand_seed_ = option.rand_seed;
  auto init_rc = model.Init(config);
  if (init_rc != MiniCNNClassifier::SUCCESS) {
    cout << "Init failed: " << model.err_msg() << endl;
    return -1;
  }

  cout << "Train CNN on MNIST subset: train=" << train_images.size()
       << " test=" << test_images.size() << " epochs=" << option.epoch_num
       << " lr=" << option.learning_rate
       << " conv_channels=" << option.conv_channels
       << " kernel=" << option.kernel_size << "x" << option.kernel_size << endl;

  auto t0 = chrono::high_resolution_clock::now();
  auto callback = [&](int epoch, double average_loss, bool &) {
    double train_acc = Accuracy(model, train_images, train_labels);
    double test_acc = Accuracy(model, test_images, test_labels);
    cout << "epoch " << (epoch + 1) << "/" << option.epoch_num
         << " loss=" << fixed << setprecision(4) << average_loss
         << " train_acc=" << train_acc << " test_acc=" << test_acc << endl;
  };
  auto train_rc =
      model.Train(train_images, train_labels, callback, option.epoch_num,
                  option.learning_rate);
  if (train_rc != MiniCNNClassifier::SUCCESS) {
    cout << "Train failed: " << model.err_msg() << endl;
    return -1;
  }

  auto t1 = chrono::high_resolution_clock::now();
  double elapsed_sec = chrono::duration<double>(t1 - t0).count();
  double train_acc = Accuracy(model, train_images, train_labels);
  double test_acc = Accuracy(model, test_images, test_labels);
  cout << "Final accuracy: train=" << fixed << setprecision(4) << train_acc
       << " test=" << test_acc << endl;
  cout << "Elapsed: " << fixed << setprecision(1) << elapsed_sec << " sec" << endl;
  cout << "This demo trains a minimal CNN (conv + ReLU + max-pool + linear) "
          "on MNIST. Increase --train-limit/--epochs for a stronger model."
       << endl;
  return 0;
}
