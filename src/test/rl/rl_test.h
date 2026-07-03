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
