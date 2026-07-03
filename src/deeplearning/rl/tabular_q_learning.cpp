#include "rl/tabular_q_learning.h"

#include "util/random.h"

#include <algorithm>
#include <cmath>

namespace deeplearning {

void TabularQLearning::Init(const Config &config) {
  config_ = config;
  q_table_.clear();
  random_counter_ = config.rand_seed;
  err_msg_.clear();
  is_init_ = true;
}

void TabularQLearning::set_random_seed(int seed) {
  config_.rand_seed = seed;
  random_counter_ = seed;
}

std::vector<double> &TabularQLearning::QRow(int state_key) {
  auto it = q_table_.find(state_key);
  if (it == q_table_.end()) {
    it = q_table_.emplace(state_key, std::vector<double>(config_.action_num, 0.0))
             .first;
  }
  return it->second;
}

double TabularQLearning::QValue(int state_key, int action) const {
  auto it = q_table_.find(state_key);
  if (it == q_table_.end()) {
    return 0.0;
  }
  return it->second[static_cast<size_t>(action)];
}

int TabularQLearning::ArgmaxQ(int state_key,
                              const std::vector<int> &legal_actions) const {
  if (legal_actions.empty()) {
    return -1;
  }
  int best_action = legal_actions.front();
  double best_value = QValue(state_key, best_action);
  for (size_t i = 1; i < legal_actions.size(); i++) {
    int action = legal_actions[i];
    double value = QValue(state_key, action);
    if (value > best_value) {
      best_value = value;
      best_action = action;
    }
  }
  return best_action;
}

double TabularQLearning::MaxQ(int state_key,
                              const std::vector<int> &legal_actions) const {
  int best_action = ArgmaxQ(state_key, legal_actions);
  if (best_action < 0) {
    return 0.0;
  }
  return QValue(state_key, best_action);
}

int TabularQLearning::SelectAction(int state_key,
                                   const std::vector<int> &legal_actions,
                                   bool explore) {
  if (!is_init_) {
    err_msg_ = "SelectAction: not init";
    return -1;
  }
  if (legal_actions.empty()) {
    err_msg_ = "SelectAction: no legal action";
    return -1;
  }

  if (explore && config_.epsilon > 0.0) {
    Random random(0, 1000000, random_counter_++);
    double sample = random.CreateRandom() / 1000000.0;
    if (sample < config_.epsilon) {
      return TicTacToeEnv::ChooseRandomAction(legal_actions, random_counter_++);
    }
  }
  return ArgmaxQ(state_key, legal_actions);
}

void TabularQLearning::Update(int state_key, int action, double reward,
                              int next_state_key,
                              const std::vector<int> &legal_actions_next,
                              bool terminal) {
  if (!is_init_) {
    err_msg_ = "Update: not init";
    return;
  }
  if (action < 0 || action >= config_.action_num) {
    err_msg_ = "Update: invalid action";
    return;
  }

  double old_q = QRow(state_key)[static_cast<size_t>(action)];
  double target = reward;
  if (!terminal) {
    target += config_.gamma * MaxQ(next_state_key, legal_actions_next);
  }
  QRow(state_key)[static_cast<size_t>(action)] =
      old_q + config_.alpha * (target - old_q);
}

void TabularQLearning::DecayEpsilon() {
  config_.epsilon = std::max(config_.epsilon_min,
                             config_.epsilon * config_.epsilon_decay);
}

TabularQLearning::RC TabularQLearning::RunEpisode(TicTacToeEnv &env,
                                                  OpponentType opponent,
                                                  EpisodeStats &stats) {
  stats = EpisodeStats{};
  if (!is_init_) {
    err_msg_ = "RunEpisode: not init";
    return RC::NOT_INIT;
  }

  env.Reset();
  while (!env.IsTerminal()) {
    int state_key = env.AgentStateKey();
    std::vector<int> legal_actions = env.LegalActions();
    int action = SelectAction(state_key, legal_actions, true);
    if (action < 0) {
      err_msg_ = "RunEpisode: agent action failed";
      return RC::INVALID_ACTION;
    }

    TicTacToeEnv::StepResult agent_step;
    if (env.StepAgent(action, agent_step) != TicTacToeEnv::RC::SUCCESS) {
      err_msg_ = "RunEpisode: " + env.err_msg();
      return RC::INVALID_ACTION;
    }
    if (agent_step.done) {
      Update(state_key, action, agent_step.reward, -1, {}, true);
      stats.total_reward += agent_step.reward;
      break;
    }

    int opponent_action = -1;
    TicTacToeEnv::StepResult opponent_step;
    TicTacToeEnv::RC opp_rc = TicTacToeEnv::RC::SUCCESS;
    if (opponent == OpponentType::RANDOM) {
      opp_rc = env.StepOpponentRandom(opponent_action, opponent_step);
    } else {
      opp_rc = env.StepOpponentOptimal(opponent_action, opponent_step);
    }
    if (opp_rc != TicTacToeEnv::RC::SUCCESS) {
      err_msg_ = "RunEpisode: " + env.err_msg();
      return RC::INVALID_ACTION;
    }

    if (opponent_step.done) {
      Update(state_key, action, opponent_step.reward, -1, {}, true);
      stats.total_reward += opponent_step.reward;
      break;
    }

    std::vector<int> next_legal = env.LegalActions();
    Update(state_key, action, 0.0, opponent_step.agent_state_key, next_legal,
           false);
  }

  TicTacToeEnv::GameResult result = env.Result();
  if (result == TicTacToeEnv::GameResult::X_WIN) {
    stats.win_num = 1;
  } else if (result == TicTacToeEnv::GameResult::DRAW) {
    stats.draw_num = 1;
  } else if (result == TicTacToeEnv::GameResult::O_WIN) {
    stats.loss_num = 1;
  }
  return RC::SUCCESS;
}

TabularQLearning::EvalStats
TabularQLearning::Evaluate(TicTacToeEnv &env, OpponentType opponent,
                           int game_num) {
  EvalStats stats;
  for (int i = 0; i < game_num; i++) {
    env.set_opponent_random_seed(config_.rand_seed + i * 17 + 1000);
    env.Reset();
    while (!env.IsTerminal()) {
      int state_key = env.AgentStateKey();
      std::vector<int> legal_actions = env.LegalActions();
      int action = SelectAction(state_key, legal_actions, false);
      TicTacToeEnv::StepResult agent_step;
      env.StepAgent(action, agent_step);
      if (agent_step.done) {
        break;
      }

      int opponent_action = -1;
      TicTacToeEnv::StepResult opponent_step;
      if (opponent == OpponentType::RANDOM) {
        env.StepOpponentRandom(opponent_action, opponent_step);
      } else {
        env.StepOpponentOptimal(opponent_action, opponent_step);
      }
    }

    TicTacToeEnv::GameResult result = env.Result();
    if (result == TicTacToeEnv::GameResult::X_WIN) {
      stats.win_num++;
    } else if (result == TicTacToeEnv::GameResult::DRAW) {
      stats.draw_num++;
    } else if (result == TicTacToeEnv::GameResult::O_WIN) {
      stats.loss_num++;
    }
  }
  return stats;
}

double TabularQLearning::epsilon() const { return config_.epsilon; }

size_t TabularQLearning::q_table_size() const { return q_table_.size(); }

std::string TabularQLearning::err_msg() const { return err_msg_; }

} // namespace deeplearning
