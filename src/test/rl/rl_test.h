#pragma once

#include "rl/tabular_q_learning.h"
#include "rl/tic_tac_toe_env.h"
#include "test.h"

TEST(TicTacToeEnv, WinDrawAndIllegalMove) {
  TicTacToeEnv env;
  MUST_TRUE(env.Reset() == TicTacToeEnv::RC::SUCCESS, env.err_msg());

  TicTacToeEnv::StepResult step;
  MUST_TRUE(env.StepAgent(0, step) == TicTacToeEnv::RC::SUCCESS, env.err_msg());
  MUST_TRUE(!step.done, "game should continue");

  int opp_action = -1;
  MUST_TRUE(env.StepOpponentRandom(opp_action, step) == TicTacToeEnv::RC::SUCCESS,
            env.err_msg());

  MUST_TRUE(env.StepAgent(0, step) == TicTacToeEnv::RC::INVALID_ACTION,
            env.err_msg());

  env.Reset();
  MUST_TRUE(env.StepAgent(0, step) == TicTacToeEnv::RC::SUCCESS, env.err_msg());
  MUST_TRUE(env.StepOpponent(4, step) == TicTacToeEnv::RC::SUCCESS, env.err_msg());
  MUST_TRUE(env.StepAgent(1, step) == TicTacToeEnv::RC::SUCCESS, env.err_msg());
  MUST_TRUE(env.StepOpponent(5, step) == TicTacToeEnv::RC::SUCCESS, env.err_msg());
  MUST_TRUE(env.StepAgent(2, step) == TicTacToeEnv::RC::SUCCESS, env.err_msg());
  MUST_TRUE(step.done, "agent should win top row");
  MUST_TRUE(env.Result() == TicTacToeEnv::GameResult::X_WIN, "agent should win");
}

TEST(TicTacToeEnv, OptimalPlayerBlocksWin) {
  TicTacToeEnv env;
  env.Reset();
  TicTacToeEnv::StepResult step;
  MUST_TRUE(env.StepAgent(0, step) == TicTacToeEnv::RC::SUCCESS, env.err_msg());
  int action = -1;
  MUST_TRUE(env.StepOpponentOptimal(action, step) == TicTacToeEnv::RC::SUCCESS,
            env.err_msg());
  MUST_TRUE(action == 4, "optimal O should take center");
}

TEST(TabularQLearning, LearnsAgainstRandomOpponent) {
  TabularQLearning agent;
  TabularQLearning::Config config;
  config.alpha = 0.5;
  config.gamma = 0.99;
  config.epsilon = 1.0;
  config.epsilon_min = 0.05;
  config.epsilon_decay = 0.999;
  config.rand_seed = 7;
  agent.Init(config);

  TicTacToeEnv env;
  for (int episode = 0; episode < 25000; episode++) {
    env.set_opponent_random_seed(episode);
    TabularQLearning::EpisodeStats stats;
    MUST_TRUE(agent.RunEpisode(env, TabularQLearning::OpponentType::RANDOM,
                               stats) == TabularQLearning::RC::SUCCESS,
              agent.err_msg());
    agent.DecayEpsilon();
  }

  TabularQLearning::EvalStats eval =
      agent.Evaluate(env, TabularQLearning::OpponentType::RANDOM, 500);
  int unbeaten = eval.win_num + eval.draw_num;
  MUST_TRUE(unbeaten >= 450,
            "agent should mostly win or draw vs random after training");
  MUST_TRUE(agent.q_table_size() > 100, "q-table should cover many states");
}

TEST(TabularQLearning, RandomTieBreakAvoidsAlwaysPickingFirstAction) {
  TabularQLearning agent;
  TabularQLearning::Config config;
  config.rand_seed = 123;
  config.random_tie_break = true;
  agent.Init(config);

  std::vector<int> legal_actions = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  bool saw_nonzero = false;
  for (int i = 0; i < 32; i++) {
    int action = agent.SelectAction(0, legal_actions, false);
    MUST_TRUE(action >= 0 && action <= 8, "action should stay legal");
    if (action != 0) {
      saw_nonzero = true;
      break;
    }
  }
  MUST_TRUE(saw_nonzero,
            "random tie break should not always choose the first legal action");
}
