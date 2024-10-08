#include "matplot_draw.h"
#include "mnist_data.h"
#include "neural_network.h"
#include "neural_network_loader.h"
#include <iostream>
#include <utility>
#include <vector>

using namespace std;
using namespace deeplearning;
using namespace drawtool;

int main() {
  // step 1 load data
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
  // print all size of data
  cout << "train_data size: " << mnist_data.train_data().size() << endl;
  cout << "train_labels size: " << mnist_data.train_labels().size() << endl;
  cout << "test_data size: " << mnist_data.test_data().size() << endl;
  cout << "test_labels size: " << mnist_data.test_labels().size() << endl;

  // step 2 create network
  NeuralNetwork demo_network;
  auto rc = NeuralNetwork::SUCCESS;
  std::string param_file_name = "./demo/mnist/mnist/demo.param";

  // if exist param file, read from it ,or init
  NeuralNetwork::NetworkParam demo_param = {};
  NeuralNetwork::NetworkOption demo_option = {};
  auto loader_rc = NeuralNetworkLoader::ImportParamFromFile(
      demo_param, demo_option, param_file_name);

  if (loader_rc == NeuralNetworkLoader::SUCCESS) {
    auto rc = demo_network.ImportNetworkParam(demo_param, demo_option);
    if (rc != NeuralNetwork::SUCCESS) {
      cout << "ImportNetworkParam failed: " << demo_network.err_msg() << endl;
      return -1;
    }

    demo_network.set_learning_rate(0.05);

  } else {
    auto rc = demo_network.Init(vector<int>{784, 20, 10});

    if (rc != NeuralNetwork::SUCCESS) {
      cout << "Init failed: " << demo_network.err_msg() << endl;
      return -1;
    }
    cout << "Init success" << endl;

    rc = demo_network.set_param_init_function(
        ParamInitType::PARAM_INIT_UNIFORM_RANDOM);
    if (rc != NeuralNetwork::SUCCESS) {
      cout << "set_loss_function failed: " << demo_network.err_msg() << endl;
      return -1;
    }
    rc = demo_network.set_loss_function(LossType::LOSS_CROSS_ENTROPY);
    if (rc != NeuralNetwork::SUCCESS) {
      cout << "set_loss_function failed: " << demo_network.err_msg() << endl;
      return -1;
    }
  }

  cout << "Init success begin train" << endl;
  vector<vector<double>> train_target(mnist_data.train_data().size(),
                                      vector<double>(10, 0));
  for (int i = 0; i < mnist_data.train_labels().size(); i++) {
    train_target[i][int(mnist_data.train_labels()[i])] = 1;
  }
  vector<vector<double>> test_target(mnist_data.test_data().size(),
                                     vector<double>(10, 0));
  for (int i = 0; i < mnist_data.test_labels().size(); i++) {
    test_target[i][int(mnist_data.test_labels()[i])] = 1.0;
  }

  // step 3 train data
  vector<double> train_loss_y, test_loss_y, train_loss_x, test_loss_x;
  auto print_func = [&](NeuralNetwork &network, int epoch_num, bool &) {
    static int count = 0;
    if (count++ % 10000 == 0) {
      double train_loss = 0;
      rc = network.CalcLoss(mnist_data.train_data(), train_target, train_loss);
      if (rc != NeuralNetwork::SUCCESS) {
        cout << "CalcLoss failed: " << demo_network.err_msg() << endl;
        return;
      }
      double test_loss = 0;
      rc = network.CalcLoss(mnist_data.test_data(), test_target, test_loss);
      if (rc != NeuralNetwork::SUCCESS) {
        cout << "CalcLoss failed: " << demo_network.err_msg() << endl;
        return;
      }
      train_loss_y.push_back(train_loss);
      test_loss_y.push_back(test_loss);
      train_loss_x.push_back(epoch_num);
      test_loss_x.push_back(epoch_num);
      std::cout << "epoch: " << epoch_num << " train_loss: " << train_loss
                << " test_loss: " << test_loss << std::endl;
    }
  };

  // demo_network.set_optimizer_function(OptimizerType::OPTIMIZER_MOMENTUM);
  rc = demo_network.Train(mnist_data.train_data(), train_target, print_func,
                          1.5 * mnist_data.train_data().size(), 1, 0.2);
  if (rc != NeuralNetwork::SUCCESS) {
    cout << "Train failed: " << demo_network.err_msg() << endl;
    return -1;
  }
  cout << "Train success begin predict" << endl;

  // step 4 test data
  vector<pair<vector<double>, pair<int, int>>> test_error_data;
  bool save_error_data = false;
  int right_count = 0, test_date_size = mnist_data.test_data().size();

  for (int i = 0; i < test_date_size; i++) {
    vector<double> result(10, 0);
    rc = demo_network.Predict(mnist_data.test_data()[i], result);

    if (rc != NeuralNetwork::SUCCESS) {
      cout << "Predict failed: " << demo_network.err_msg() << endl;
    }

    int max_index = 0;
    double max_value = result[0];
    for (int j = 1; j < result.size(); j++) {
      if (result[j] > max_value) {
        max_value = result[j];
        max_index = j;
      }
    }
    if (max_index == mnist_data.test_labels()[i]) {
      right_count++;
    } else {
      if (save_error_data) {
        test_error_data.push_back(
            make_pair(mnist_data.test_data()[i],
                      make_pair(mnist_data.test_labels()[i], max_index)));
      }
    }
  }
  cout << " right count:" << right_count
       << " right rate: " << right_count * 1.0 / test_date_size << endl;

  // draw pic
  MatplotDraw::PrintLossResult("Mnist NeuralNetwork", train_loss_x,
                               train_loss_y, test_loss_x, test_loss_y, "epoch",
                               "loss");

  if (right_count * 1.0 / test_date_size < 0.8) {
    cout << "right rate is too low" << endl;
    return -1;
  }

  // save param
  NeuralNetwork::NetworkParam param;
  NeuralNetwork::NetworkOption option;
  rc = demo_network.ExportNetworkParam(param, option);
  if (rc != NeuralNetwork::SUCCESS) {
    cout << "ExportNetworkParam failed: " << demo_network.err_msg() << endl;
    return -1;
  }
  // auto rcLoader =
  //     NeuralNetworkLoader::ExportParamToFile(param, option, param_file_name);
  // if (rcLoader != NeuralNetworkLoader::SUCCESS) {
  //   cout << "ExportParamToFile failed: " << demo_network.err_msg() << endl;
  //   return -1;
  // }

  // save errorimg2file
  if (save_error_data) {
    std::string filename = "./demo/mnist/mnist/error_data.txt";
    std::string result;
    for (int i = 0; i < test_error_data.size(); i++) {
      auto &data = test_error_data[i].first;
      auto &label = test_error_data[i].second;
      string data_str;
      MnistData::DrawMnistImage(test_error_data[i].first, data_str,
                                test_error_data[i].second.first,
                                test_error_data[i].second.second);
      result += data_str;
    }
    ofstream outfile(filename);
    outfile << result;
    outfile.close();
  }
  return 0;
}
