#pragma once
#include "activate/activate_factory.h"
#include "loss/loss_factory.h"
#include "optimizer/optimizer_factory.h"
#include "param_init/param_init_factory.h"
#include "softmax/softmax_factory.h"
#include "util/random.h"
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
namespace deeplearning {

class NeuralNetwork {
public:
  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
  };
  enum NetworkStatus {
    NETWORK_STATUS_UNINIT,
    NETWORK_STATUS_INIT,
  };
  struct NetworkParam {
    std::vector<int> layer_;
    std::vector<std::vector<double>> neuron_bias_;
    std::vector<std::vector<std::vector<double>>> neuron_weight_;
  };
  struct NetworkOption {
    double learning_rate_;
    int rand_seed_;
    LossType loss_type_;
    ActivateType activate_type_;
    SoftmaxType softmax_type_;
    OptimizerType optimizer_type_;
  };

public:
  NeuralNetwork();
  ~NeuralNetwork();
  NeuralNetwork(const NeuralNetwork &) = delete;
  NeuralNetwork &operator=(const NeuralNetwork &) = delete;

  NeuralNetwork(const std::vector<int> &layer);

  RC Init(const std::vector<int> &layer);

  RC Train(const std::vector<std::vector<double>> &data,
           const std::vector<std::vector<double>> &target,
           std::function<void(NeuralNetwork &network, int epoch_num,
                               bool &early_stop)>
               each_epoch_call = nullptr,
           int epoch_num = 0, int batch_num = 1, double learning_rate = 0);

  RC Predict(const std::vector<double> &data, std::vector<double> &result);

  RC CalcLoss(const std::vector<std::vector<double>> &data,
              const std::vector<std::vector<double>> &target, double &loss);

  RC ExportNetworkParam(NetworkParam &param, NetworkOption &option);

  RC ImportNetworkParam(const NetworkParam &param,
                        const NetworkOption &option);

  RC Clone(const NeuralNetwork &old);

public:
  std::string err_msg();
  double learning_rate();
  int rand_seed();
  NetworkStatus network_status();
  const std::vector<std::vector<std::vector<double>>> &neuron_weight();
  const std::vector<std::vector<double>> &neuron_bias();

  void set_learning_rate(double rate);
  void set_random_seed(int seed);
  RC set_loss_function(LossType type);
  RC set_activate_function(ActivateType type);
  RC set_softmax_function(SoftmaxType type);
  RC set_param_init_function(ParamInitType type);
  RC set_optimizer_function(OptimizerType type);

private:
  double CalcDelta(const double deriv_target, const double out);

  void InitParamWithLayer(const std::vector<int> &layer);

  RC UpdateNeuronOutput(const std::pair<int, int> &neuron_pos,
                        const std::vector<double> &input);

  RC UpdateNeuronOutputSoftMax();

  void ClearNeuronDelta();

  RC UpdateNeuronDelta(const std::pair<int, int> &neuron_pos,
                       const std::vector<double> &target);

  RC UpdateAllNeuron();

  RC UpdateSingleNeuron(const std::pair<int, int> &neuron_pos);

  RC ForwardPropagation(const std::vector<double> &data);

  RC BackPropagation(const std::vector<double> &input,
                     const std::vector<double> &target);

private:
  std::shared_ptr<LossFunction> loss_function_ = nullptr;
  std::shared_ptr<ActivateFunction> activate_function_ = nullptr;
  std::shared_ptr<SoftmaxFunction> softmax_function_ = nullptr;
  std::shared_ptr<ParamInitFunction> param_init_function_ = nullptr;
  std::shared_ptr<OptimizerFunction> optimizer_function_ = nullptr;

  NetworkStatus network_status_ = NETWORK_STATUS_UNINIT;
  int rand_seed_ = 0;
  double learning_rate_ = 0.1;
  std::vector<int> layer_;
  std::vector<std::vector<double>> neuron_bias_;
  std::vector<std::vector<std::vector<double>>> neuron_weight_;
  std::vector<std::vector<double>> neuron_output_;
  std::vector<std::vector<double>> neuron_delta_;
  std::string err_msg_;
};

} // namespace deeplearning
