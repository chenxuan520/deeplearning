#include "rnn/simple_gru.h"

#include <cmath>
#include <random>

namespace deeplearning {

namespace {
double Sigmoid(double x) {
  if (x >= 0.0) {
    double e = std::exp(-x);
    return 1.0 / (1.0 + e);
  }
  double e = std::exp(x);
  return e / (1.0 + e);
}
} // namespace

SimpleGRU::RC SimpleGRU::Init(int input_dim, int hidden_dim) {
  if (is_init_) {
    err_msg_ = "[SimpleGRU::Init] SimpleGRU has init";
    return ALREADY_INIT;
  }
  if (input_dim <= 0 || hidden_dim <= 0) {
    err_msg_ = "[SimpleGRU::Init] Invalid config";
    return INVALID_DATA;
  }
  input_dim_ = input_dim;
  hidden_dim_ = hidden_dim;
  auto make_w = [&](int rows, int cols) {
    return Matrix(rows, std::vector<double>(cols, 0.0));
  };
  wz_ = make_w(hidden_dim_, input_dim_);
  wr_ = make_w(hidden_dim_, input_dim_);
  wn_ = make_w(hidden_dim_, input_dim_);
  uz_ = make_w(hidden_dim_, hidden_dim_);
  ur_ = make_w(hidden_dim_, hidden_dim_);
  un_ = make_w(hidden_dim_, hidden_dim_);
  bz_.assign(hidden_dim_, 0.0);
  br_.assign(hidden_dim_, 0.0);
  bn_.assign(hidden_dim_, 0.0);
  ResetGradients();

  std::mt19937 gen(static_cast<std::mt19937::result_type>(rand_seed_));
  const double input_limit =
      std::sqrt(6.0 / static_cast<double>(input_dim_ + hidden_dim_));
  const double hidden_limit =
      std::sqrt(6.0 / static_cast<double>(hidden_dim_ + hidden_dim_));
  std::uniform_real_distribution<double> input_dist(-input_limit, input_limit);
  std::uniform_real_distribution<double> hidden_dist(-hidden_limit, hidden_limit);
  for (auto *m : {&wz_, &wr_, &wn_}) {
    for (auto &row : *m) {
      for (double &v : row) v = input_dist(gen);
    }
  }
  for (auto *m : {&uz_, &ur_, &un_}) {
    for (auto &row : *m) {
      for (double &v : row) v = hidden_dist(gen);
    }
  }
  is_init_ = true;
  return SUCCESS;
}

SimpleGRU::RC SimpleGRU::Forward(const Matrix &input_sequence,
                                 Matrix &hidden_sequence) {
  if (!is_init_) {
    err_msg_ = "[SimpleGRU::Forward] SimpleGRU not init";
    return NOT_INIT;
  }
  if (input_sequence.empty()) {
    err_msg_ = "[SimpleGRU::Forward] Invalid input sequence";
    return INVALID_DATA;
  }
  for (const auto &input : input_sequence) {
    if (static_cast<int>(input.size()) != input_dim_) {
      err_msg_ = "[SimpleGRU::Forward] Invalid input sequence";
      return INVALID_DATA;
    }
  }

  last_input_sequence_ = input_sequence;
  last_hidden_sequence_.assign(input_sequence.size() + 1,
                               std::vector<double>(hidden_dim_, 0.0));
  last_z_sequence_.assign(input_sequence.size(),
                          std::vector<double>(hidden_dim_, 0.0));
  last_r_sequence_.assign(input_sequence.size(),
                          std::vector<double>(hidden_dim_, 0.0));
  last_n_sequence_.assign(input_sequence.size(),
                          std::vector<double>(hidden_dim_, 0.0));
  hidden_sequence.assign(input_sequence.size(),
                         std::vector<double>(hidden_dim_, 0.0));

  for (int t = 0; t < static_cast<int>(input_sequence.size()); ++t) {
    const auto &x = input_sequence[t];
    const auto &prev = last_hidden_sequence_[t];
    auto &z = last_z_sequence_[t];
    auto &r = last_r_sequence_[t];
    auto &n = last_n_sequence_[t];
    auto &h = hidden_sequence[t];
    for (int j = 0; j < hidden_dim_; ++j) {
      double az = bz_[j];
      double ar = br_[j];
      for (int i = 0; i < input_dim_; ++i) {
        az += wz_[j][i] * x[i];
        ar += wr_[j][i] * x[i];
      }
      for (int i = 0; i < hidden_dim_; ++i) {
        az += uz_[j][i] * prev[i];
        ar += ur_[j][i] * prev[i];
      }
      z[j] = Sigmoid(az);
      r[j] = Sigmoid(ar);
    }
    for (int j = 0; j < hidden_dim_; ++j) {
      double an = bn_[j];
      for (int i = 0; i < input_dim_; ++i) {
        an += wn_[j][i] * x[i];
      }
      for (int i = 0; i < hidden_dim_; ++i) {
        an += un_[j][i] * (r[i] * prev[i]);
      }
      n[j] = std::tanh(an);
      h[j] = (1.0 - z[j]) * n[j] + z[j] * prev[j];
    }
    last_hidden_sequence_[t + 1] = h;
  }
  has_forward_cache_ = true;
  return SUCCESS;
}

SimpleGRU::RC SimpleGRU::Backward(const Matrix &grad_hidden_sequence,
                                  Matrix &grad_input_sequence) {
  if (!is_init_) {
    err_msg_ = "[SimpleGRU::Backward] SimpleGRU not init";
    return NOT_INIT;
  }
  if (!has_forward_cache_ ||
      grad_hidden_sequence.size() + 1 != last_hidden_sequence_.size()) {
    err_msg_ = "[SimpleGRU::Backward] Invalid grad sequence";
    return INVALID_DATA;
  }
  for (const auto &g : grad_hidden_sequence) {
    if (static_cast<int>(g.size()) != hidden_dim_) {
      err_msg_ = "[SimpleGRU::Backward] Invalid grad sequence";
      return INVALID_DATA;
    }
  }

  ResetGradients();
  grad_input_sequence.assign(grad_hidden_sequence.size(),
                             std::vector<double>(input_dim_, 0.0));
  std::vector<double> grad_future(hidden_dim_, 0.0);

  for (int t = static_cast<int>(grad_hidden_sequence.size()) - 1; t >= 0; --t) {
    const auto &x = last_input_sequence_[t];
    const auto &prev = last_hidden_sequence_[t];
    const auto &z = last_z_sequence_[t];
    const auto &r = last_r_sequence_[t];
    const auto &n = last_n_sequence_[t];
    std::vector<double> grad_h = grad_hidden_sequence[t];
    for (int j = 0; j < hidden_dim_; ++j) grad_h[j] += grad_future[j];

    std::vector<double> grad_prev(hidden_dim_, 0.0);
    std::vector<double> grad_z(hidden_dim_, 0.0);
    std::vector<double> grad_n(hidden_dim_, 0.0);
    std::vector<double> grad_rprev(hidden_dim_, 0.0);
    std::vector<double> grad_r(hidden_dim_, 0.0);

    for (int j = 0; j < hidden_dim_; ++j) {
      grad_z[j] += grad_h[j] * (prev[j] - n[j]);
      grad_n[j] += grad_h[j] * (1.0 - z[j]);
      grad_prev[j] += grad_h[j] * z[j];
    }

    for (int j = 0; j < hidden_dim_; ++j) {
      double da = grad_n[j] * (1.0 - n[j] * n[j]);
      grad_bn_[j] += da;
      for (int i = 0; i < input_dim_; ++i) {
        grad_wn_[j][i] += da * x[i];
        grad_input_sequence[t][i] += wn_[j][i] * da;
      }
      for (int i = 0; i < hidden_dim_; ++i) {
        double rp = r[i] * prev[i];
        grad_un_[j][i] += da * rp;
        grad_rprev[i] += un_[j][i] * da;
      }
    }
    for (int i = 0; i < hidden_dim_; ++i) {
      grad_r[i] += grad_rprev[i] * prev[i];
      grad_prev[i] += grad_rprev[i] * r[i];
    }

    for (int j = 0; j < hidden_dim_; ++j) {
      double dz = grad_z[j] * z[j] * (1.0 - z[j]);
      grad_bz_[j] += dz;
      for (int i = 0; i < input_dim_; ++i) {
        grad_wz_[j][i] += dz * x[i];
        grad_input_sequence[t][i] += wz_[j][i] * dz;
      }
      for (int i = 0; i < hidden_dim_; ++i) {
        grad_uz_[j][i] += dz * prev[i];
        grad_prev[i] += uz_[j][i] * dz;
      }
    }

    for (int j = 0; j < hidden_dim_; ++j) {
      double dr = grad_r[j] * r[j] * (1.0 - r[j]);
      grad_br_[j] += dr;
      for (int i = 0; i < input_dim_; ++i) {
        grad_wr_[j][i] += dr * x[i];
        grad_input_sequence[t][i] += wr_[j][i] * dr;
      }
      for (int i = 0; i < hidden_dim_; ++i) {
        grad_ur_[j][i] += dr * prev[i];
        grad_prev[i] += ur_[j][i] * dr;
      }
    }
    grad_future = std::move(grad_prev);
  }
  return SUCCESS;
}

void SimpleGRU::ApplyGradient(double learning_rate) {
  if (learning_rate <= 0.0) return;
  wz_optimizer_.Apply(wz_, grad_wz_, learning_rate);
  uz_optimizer_.Apply(uz_, grad_uz_, learning_rate);
  bz_optimizer_.Apply(bz_, grad_bz_, learning_rate);
  wr_optimizer_.Apply(wr_, grad_wr_, learning_rate);
  ur_optimizer_.Apply(ur_, grad_ur_, learning_rate);
  br_optimizer_.Apply(br_, grad_br_, learning_rate);
  wn_optimizer_.Apply(wn_, grad_wn_, learning_rate);
  un_optimizer_.Apply(un_, grad_un_, learning_rate);
  bn_optimizer_.Apply(bn_, grad_bn_, learning_rate);
}

void SimpleGRU::set_random_seed(int seed) { rand_seed_ = seed; }

bool SimpleGRU::CheckWeightShapes(const Matrix &wz, const Matrix &uz,
                                  const std::vector<double> &bz,
                                  const Matrix &wr, const Matrix &ur,
                                  const std::vector<double> &br,
                                  const Matrix &wn, const Matrix &un,
                                  const std::vector<double> &bn) {
  auto check = [&](const Matrix &m, int rows, int cols) {
    if (static_cast<int>(m.size()) != rows) return false;
    for (const auto &row : m) {
      if (static_cast<int>(row.size()) != cols) return false;
    }
    return true;
  };
  return check(wz, hidden_dim_, input_dim_) &&
         check(wr, hidden_dim_, input_dim_) &&
         check(wn, hidden_dim_, input_dim_) &&
         check(uz, hidden_dim_, hidden_dim_) &&
         check(ur, hidden_dim_, hidden_dim_) &&
         check(un, hidden_dim_, hidden_dim_) &&
         static_cast<int>(bz.size()) == hidden_dim_ &&
         static_cast<int>(br.size()) == hidden_dim_ &&
         static_cast<int>(bn.size()) == hidden_dim_;
}

SimpleGRU::RC SimpleGRU::set_weights(const Matrix &wz, const Matrix &uz,
                                     const std::vector<double> &bz,
                                     const Matrix &wr, const Matrix &ur,
                                     const std::vector<double> &br,
                                     const Matrix &wn, const Matrix &un,
                                     const std::vector<double> &bn) {
  if (!is_init_) {
    err_msg_ = "[SimpleGRU::set_weights] SimpleGRU not init";
    return NOT_INIT;
  }
  if (!CheckWeightShapes(wz, uz, bz, wr, ur, br, wn, un, bn)) {
    err_msg_ = "[SimpleGRU::set_weights] Invalid weight size";
    return INVALID_DATA;
  }
  wz_ = wz; uz_ = uz; bz_ = bz;
  wr_ = wr; ur_ = ur; br_ = br;
  wn_ = wn; un_ = un; bn_ = bn;
  return SUCCESS;
}

SimpleGRU::RC SimpleGRU::SetGradients(
    const Matrix &gwz, const Matrix &guz, const std::vector<double> &gbz,
    const Matrix &gwr, const Matrix &gur, const std::vector<double> &gbr,
    const Matrix &gwn, const Matrix &gun, const std::vector<double> &gbn) {
  if (!is_init_) return NOT_INIT;
  if (!CheckWeightShapes(gwz, guz, gbz, gwr, gur, gbr, gwn, gun, gbn)) {
    err_msg_ = "[SimpleGRU::SetGradients] Invalid gradient size";
    return INVALID_DATA;
  }
  grad_wz_ = gwz; grad_uz_ = guz; grad_bz_ = gbz;
  grad_wr_ = gwr; grad_ur_ = gur; grad_br_ = gbr;
  grad_wn_ = gwn; grad_un_ = gun; grad_bn_ = gbn;
  return SUCCESS;
}

std::string SimpleGRU::err_msg() { return err_msg_; }
int SimpleGRU::input_dim() const { return input_dim_; }
int SimpleGRU::hidden_dim() const { return hidden_dim_; }
const SimpleGRU::Matrix &SimpleGRU::wz() const { return wz_; }
const SimpleGRU::Matrix &SimpleGRU::uz() const { return uz_; }
const std::vector<double> &SimpleGRU::bz() const { return bz_; }
const SimpleGRU::Matrix &SimpleGRU::wr() const { return wr_; }
const SimpleGRU::Matrix &SimpleGRU::ur() const { return ur_; }
const std::vector<double> &SimpleGRU::br() const { return br_; }
const SimpleGRU::Matrix &SimpleGRU::wn() const { return wn_; }
const SimpleGRU::Matrix &SimpleGRU::un() const { return un_; }
const std::vector<double> &SimpleGRU::bn() const { return bn_; }
const SimpleGRU::Matrix &SimpleGRU::grad_wz() const { return grad_wz_; }
const SimpleGRU::Matrix &SimpleGRU::grad_uz() const { return grad_uz_; }
const std::vector<double> &SimpleGRU::grad_bz() const { return grad_bz_; }
const SimpleGRU::Matrix &SimpleGRU::grad_wr() const { return grad_wr_; }
const SimpleGRU::Matrix &SimpleGRU::grad_ur() const { return grad_ur_; }
const std::vector<double> &SimpleGRU::grad_br() const { return grad_br_; }
const SimpleGRU::Matrix &SimpleGRU::grad_wn() const { return grad_wn_; }
const SimpleGRU::Matrix &SimpleGRU::grad_un() const { return grad_un_; }
const std::vector<double> &SimpleGRU::grad_bn() const { return grad_bn_; }

void SimpleGRU::ResetGradients() {
  grad_wz_.assign(hidden_dim_, std::vector<double>(input_dim_, 0.0));
  grad_wr_.assign(hidden_dim_, std::vector<double>(input_dim_, 0.0));
  grad_wn_.assign(hidden_dim_, std::vector<double>(input_dim_, 0.0));
  grad_uz_.assign(hidden_dim_, std::vector<double>(hidden_dim_, 0.0));
  grad_ur_.assign(hidden_dim_, std::vector<double>(hidden_dim_, 0.0));
  grad_un_.assign(hidden_dim_, std::vector<double>(hidden_dim_, 0.0));
  grad_bz_.assign(hidden_dim_, 0.0);
  grad_br_.assign(hidden_dim_, 0.0);
  grad_bn_.assign(hidden_dim_, 0.0);
}

} // namespace deeplearning
