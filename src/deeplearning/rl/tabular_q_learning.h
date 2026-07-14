#pragma once

#include "rl/tic_tac_toe_env.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace deeplearning {

class TabularQLearning {
public:
  struct Config {
    double alpha = 0.5;
    double gamma = 0.99;
    double epsilon = 1.0;
    double epsilon_min = 0.05;
    double epsilon_decay = 0.9995;
    int rand_seed = 0;
    int action_num = 9;
    bool random_tie_break = false;
  };

  enum class RC {
    SUCCESS,
    NOT_INIT,
    INVALID_ACTION,
  };

  enum class OpponentType { RANDOM, OPTIMAL };

  struct EpisodeStats {
    int win_num = 0;
    int draw_num = 0;
    int loss_num = 0;
    double total_reward = 0.0;
  };

  struct EvalStats {
    int win_num = 0;
    int draw_num = 0;
    int loss_num = 0;
  };

public:
  void Init(const Config &config);
  void set_random_seed(int seed);

  int SelectAction(int state_key, const std::vector<int> &legal_actions,
                   bool explore = true);
  void Update(int state_key, int action, double reward, int next_state_key,
              const std::vector<int> &legal_actions_next, bool terminal);
  void DecayEpsilon();

  RC RunEpisode(TicTacToeEnv &env, OpponentType opponent, EpisodeStats &stats);
  EvalStats Evaluate(TicTacToeEnv &env, OpponentType opponent, int game_num);

  double epsilon() const;
  size_t q_table_size() const;
  const std::unordered_map<int, std::vector<double>> &q_table() const;
  double QValue(int state_key, int action) const;
  std::string err_msg() const;

private:
  std::vector<double> &QRow(int state_key);
  int ArgmaxQ(int state_key, const std::vector<int> &legal_actions);
  double MaxQ(int state_key, const std::vector<int> &legal_actions);

private:
  Config config_;
  std::unordered_map<int, std::vector<double>> q_table_;
  int random_counter_ = 0;
  std::string err_msg_;
  bool is_init_ = false;
};

} // namespace deeplearning
