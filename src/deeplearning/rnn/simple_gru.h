#pragma once

#include "transformer/tensor_optimizer.h"

#include <string>
#include <vector>

namespace deeplearning {

class SimpleGRU {
public:
  using Matrix = std::vector<std::vector<double>>;

  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
  };

  RC Init(int input_dim, int hidden_dim);
  RC Forward(const Matrix &input_sequence, Matrix &hidden_sequence);
  RC Backward(const Matrix &grad_hidden_sequence, Matrix &grad_input_sequence);
  void ApplyGradient(double learning_rate);

  void set_random_seed(int seed);
  RC set_weights(const Matrix &wz, const Matrix &uz,
                 const std::vector<double> &bz, const Matrix &wr,
                 const Matrix &ur, const std::vector<double> &br,
                 const Matrix &wn, const Matrix &un,
                 const std::vector<double> &bn);
  RC SetGradients(const Matrix &gwz, const Matrix &guz,
                  const std::vector<double> &gbz, const Matrix &gwr,
                  const Matrix &gur, const std::vector<double> &gbr,
                  const Matrix &gwn, const Matrix &gun,
                  const std::vector<double> &gbn);

  std::string err_msg();
  int input_dim() const;
  int hidden_dim() const;
  const Matrix &wz() const;
  const Matrix &uz() const;
  const std::vector<double> &bz() const;
  const Matrix &wr() const;
  const Matrix &ur() const;
  const std::vector<double> &br() const;
  const Matrix &wn() const;
  const Matrix &un() const;
  const std::vector<double> &bn() const;
  const Matrix &grad_wz() const;
  const Matrix &grad_uz() const;
  const std::vector<double> &grad_bz() const;
  const Matrix &grad_wr() const;
  const Matrix &grad_ur() const;
  const std::vector<double> &grad_br() const;
  const Matrix &grad_wn() const;
  const Matrix &grad_un() const;
  const std::vector<double> &grad_bn() const;

private:
  void ResetGradients();
  bool CheckWeightShapes(const Matrix &wz, const Matrix &uz,
                         const std::vector<double> &bz, const Matrix &wr,
                         const Matrix &ur, const std::vector<double> &br,
                         const Matrix &wn, const Matrix &un,
                         const std::vector<double> &bn);

private:
  int input_dim_ = 0;
  int hidden_dim_ = 0;
  int rand_seed_ = 0;
  Matrix wz_, uz_, wr_, ur_, wn_, un_;
  std::vector<double> bz_, br_, bn_;
  Matrix grad_wz_, grad_uz_, grad_wr_, grad_ur_, grad_wn_, grad_un_;
  std::vector<double> grad_bz_, grad_br_, grad_bn_;
  Matrix last_input_sequence_;
  Matrix last_hidden_sequence_;
  Matrix last_z_sequence_;
  Matrix last_r_sequence_;
  Matrix last_n_sequence_;
  TensorOptimizer wz_optimizer_, uz_optimizer_, bz_optimizer_;
  TensorOptimizer wr_optimizer_, ur_optimizer_, br_optimizer_;
  TensorOptimizer wn_optimizer_, un_optimizer_, bn_optimizer_;
  std::string err_msg_;
  bool is_init_ = false;
  bool has_forward_cache_ = false;
};

} // namespace deeplearning
