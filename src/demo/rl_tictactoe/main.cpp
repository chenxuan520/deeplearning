#include "rl/tabular_q_learning.h"
#include "rl/tic_tac_toe_env.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace deeplearning;

namespace {

struct DemoOption {
  int episodes = 30000;
  int eval_games = 1000;
  int rand_seed = 0;
  double alpha = 0.5;
  double gamma = 0.99;
  double epsilon = 1.0;
  double epsilon_min = 0.05;
  double epsilon_decay = 0.9995;
  string opponent = "random";
  string export_json;
  bool play = false;
  bool show_sample = false;
};

void PrintUsage(const char *prog) {
  cout << "Usage: " << prog << " [options]\n"
       << "  --episodes <int>           training episodes (default 30000)\n"
       << "  --eval-games <int>         evaluation games (default 1000)\n"
       << "  --rand-seed <int>\n"
       << "  --alpha <double>\n"
       << "  --gamma <double>\n"
       << "  --epsilon <double>\n"
       << "  --epsilon-min <double>\n"
       << "  --epsilon-decay <double>\n"
       << "  --opponent random|optimal   training opponent (default random)\n"
       << "  --export-json <path>        export trained Q table and metrics\n"
       << "  --play                      play one game vs trained agent (stdin)\n"
       << "  --show-sample               print one greedy eval game\n"
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

    if (arg == "--episodes") {
      option.episodes = std::stoi(need_value("--episodes"));
    } else if (arg == "--eval-games") {
      option.eval_games = std::stoi(need_value("--eval-games"));
    } else if (arg == "--rand-seed") {
      option.rand_seed = std::stoi(need_value("--rand-seed"));
    } else if (arg == "--alpha") {
      option.alpha = std::stod(need_value("--alpha"));
    } else if (arg == "--gamma") {
      option.gamma = std::stod(need_value("--gamma"));
    } else if (arg == "--epsilon") {
      option.epsilon = std::stod(need_value("--epsilon"));
    } else if (arg == "--epsilon-min") {
      option.epsilon_min = std::stod(need_value("--epsilon-min"));
    } else if (arg == "--epsilon-decay") {
      option.epsilon_decay = std::stod(need_value("--epsilon-decay"));
    } else if (arg == "--opponent") {
      option.opponent = need_value("--opponent");
    } else if (arg == "--export-json") {
      option.export_json = need_value("--export-json");
    } else if (arg == "--play") {
      option.play = true;
    } else if (arg == "--show-sample") {
      option.show_sample = true;
    } else if (arg == "--help") {
      return false;
    } else {
      throw std::runtime_error(string("Unknown option: ") + arg);
    }
  }
  return true;
}

TabularQLearning::OpponentType ParseOpponent(const string &name) {
  if (name == "random") {
    return TabularQLearning::OpponentType::RANDOM;
  }
  if (name == "optimal") {
    return TabularQLearning::OpponentType::OPTIMAL;
  }
  throw std::runtime_error("Unknown opponent: " + name);
}

void PrintBoardWithIndex(const TicTacToeEnv &env) {
  cout << env.Render() << "\n";
  cout << "positions: 0|1|2 / 3|4|5 / 6|7|8\n";
}

void PrintResult(const TicTacToeEnv &env) {
  PrintBoardWithIndex(env);
  switch (env.Result()) {
  case TicTacToeEnv::GameResult::X_WIN:
    cout << "Result: agent(X) wins\n";
    break;
  case TicTacToeEnv::GameResult::O_WIN:
    cout << "Result: opponent(O) wins\n";
    break;
  case TicTacToeEnv::GameResult::DRAW:
    cout << "Result: draw\n";
    break;
  default:
    break;
  }
}

void PlayInteractive(TicTacToeEnv &env, TabularQLearning &agent) {
  env.Reset();
  cout << "You are O, agent is X. Enter position 0-8.\n";
  while (!env.IsTerminal()) {
    PrintBoardWithIndex(env);
    if (env.IsAgentTurn()) {
      int action =
          agent.SelectAction(env.AgentStateKey(), env.LegalActions(), false);
      TicTacToeEnv::StepResult step;
      env.StepAgent(action, step);
      cout << "Agent(X) plays " << action << "\n";
      if (step.done) {
        break;
      }
      continue;
    }

    int action = -1;
    while (true) {
      cout << "Your move: ";
      if (!(cin >> action)) {
        cout << "Invalid input.\n";
        return;
      }
      TicTacToeEnv::StepResult step;
      if (env.StepOpponent(action, step) == TicTacToeEnv::RC::SUCCESS) {
        if (step.done) {
          break;
        }
        break;
      }
      cout << "Illegal move (" << env.err_msg() << "), try again.\n";
    }
  }
  PrintResult(env);
}

void ShowSampleGame(TicTacToeEnv &env, TabularQLearning &agent) {
  env.Reset();
  env.set_opponent_random_seed(42);
  cout << "Sample greedy game vs random opponent:\n";
  while (!env.IsTerminal()) {
    PrintBoardWithIndex(env);
    if (env.IsAgentTurn()) {
      int action =
          agent.SelectAction(env.AgentStateKey(), env.LegalActions(), false);
      TicTacToeEnv::StepResult step;
      env.StepAgent(action, step);
      cout << "Agent(X) -> " << action << "\n\n";
      if (step.done) {
        break;
      }
    } else {
      int action = -1;
      TicTacToeEnv::StepResult step;
      env.StepOpponentRandom(action, step);
      cout << "Opponent(O) -> " << action << "\n\n";
    }
  }
  PrintResult(env);
}

void PrintEval(const char *title, const TabularQLearning::EvalStats &stats,
               int game_num) {
  cout << title << ": win=" << stats.win_num << " draw=" << stats.draw_num
       << " loss=" << stats.loss_num << " / " << game_num;
  if (game_num > 0) {
    double win_rate = stats.win_num * 100.0 / game_num;
    double unbeaten = (stats.win_num + stats.draw_num) * 100.0 / game_num;
    cout << " (win " << win_rate << "%, unbeaten " << unbeaten << "%)";
  }
  cout << "\n";
}

void WriteEvalJson(ostream &os, const TabularQLearning::EvalStats &stats,
                   int game_num) {
  os << "{\"games\":" << game_num << ",\"win\":" << stats.win_num
     << ",\"draw\":" << stats.draw_num << ",\"loss\":" << stats.loss_num
     << ",\"winRate\":" << setprecision(12)
     << (game_num > 0 ? stats.win_num * 1.0 / game_num : 0.0)
     << ",\"unbeatenRate\":"
     << (game_num > 0 ? (stats.win_num + stats.draw_num) * 1.0 / game_num
                      : 0.0)
     << "}";
}

bool ExportTrainingJson(const string &filename, const DemoOption &option,
                        const TabularQLearning &agent, int train_win,
                        int train_draw, int train_loss,
                        const TabularQLearning::EvalStats &random_eval,
                        const TabularQLearning::EvalStats &optimal_eval) {
  std::ofstream os(filename, std::ios::binary | std::ios::trunc);
  if (!os.is_open()) {
    cout << "Open export json failed: " << filename << "\n";
    return false;
  }

  vector<int> keys;
  keys.reserve(agent.q_table().size());
  for (const auto &item : agent.q_table()) {
    keys.push_back(item.first);
  }
  std::sort(keys.begin(), keys.end());

  os << "{\"format\":\"deeplearning.tictactoe_demo\",\"version\":1,";
  os << "\"training\":{\"episodes\":" << option.episodes
     << ",\"randSeed\":" << option.rand_seed << ",\"opponent\":\""
     << option.opponent << "\",\"alpha\":" << setprecision(12) << option.alpha
     << ",\"gamma\":" << option.gamma << ",\"epsilonStart\":"
     << option.epsilon << ",\"epsilonMin\":" << option.epsilon_min
     << ",\"epsilonDecay\":" << option.epsilon_decay << ",\"epsilonFinal\":"
     << agent.epsilon() << ",\"trainWin\":" << train_win
     << ",\"trainDraw\":" << train_draw << ",\"trainLoss\":" << train_loss
     << ",\"qStates\":" << keys.size() << "},";
  os << "\"evaluation\":{\"random\":";
  WriteEvalJson(os, random_eval, option.eval_games);
  os << ",\"optimal\":";
  WriteEvalJson(os, optimal_eval, option.eval_games);
  os << "},\"qTable\":{";
  for (int i = 0; i < static_cast<int>(keys.size()); i++) {
    if (i != 0) {
      os << ",";
    }
    const int key = keys[i];
    const auto &row = agent.q_table().at(key);
    os << "\"" << key << "\":[";
    for (int action = 0; action < static_cast<int>(row.size()); action++) {
      if (action != 0) {
        os << ",";
      }
      os << setprecision(12) << row[action];
    }
    os << "]";
  }
  os << "}}\n";
  return os.good();
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
    cout << ex.what() << "\n";
    PrintUsage(argv[0]);
    return 1;
  }

  TabularQLearning agent;
  TabularQLearning::Config config;
  config.alpha = option.alpha;
  config.gamma = option.gamma;
  config.epsilon = option.epsilon;
  config.epsilon_min = option.epsilon_min;
  config.epsilon_decay = option.epsilon_decay;
  config.rand_seed = option.rand_seed;
  config.random_tie_break = true;
  agent.Init(config);

  TicTacToeEnv env;
  env.set_opponent_random_seed(option.rand_seed);
  TabularQLearning::OpponentType train_opponent =
      ParseOpponent(option.opponent);

  cout << "Q-learning tic-tac-toe (agent=X, opponent during train="
       << option.opponent << ")\n";
  cout << "episodes=" << option.episodes << " alpha=" << option.alpha
       << " gamma=" << option.gamma << " epsilon=" << option.epsilon << "\n";

  int win_total = 0;
  int draw_total = 0;
  int loss_total = 0;
  for (int episode = 0; episode < option.episodes; episode++) {
    env.set_opponent_random_seed(option.rand_seed + episode);
    TabularQLearning::EpisodeStats stats;
    if (agent.RunEpisode(env, train_opponent, stats) !=
        TabularQLearning::RC::SUCCESS) {
      cout << "Train failed: " << agent.err_msg() << "\n";
      return 1;
    }
    win_total += stats.win_num;
    draw_total += stats.draw_num;
    loss_total += stats.loss_num;
    agent.DecayEpsilon();

    if ((episode + 1) % 5000 == 0 || episode + 1 == option.episodes) {
      cout << "episode " << (episode + 1) << "/" << option.episodes
           << "  train(win/draw/loss)=" << win_total << "/" << draw_total
           << "/" << loss_total << "  q_states=" << agent.q_table_size()
           << "  epsilon=" << agent.epsilon() << "\n";
    }
  }

  TabularQLearning::EvalStats random_eval =
      agent.Evaluate(env, TabularQLearning::OpponentType::RANDOM,
                     option.eval_games);
  TabularQLearning::EvalStats optimal_eval =
      agent.Evaluate(env, TabularQLearning::OpponentType::OPTIMAL,
                     option.eval_games);
  PrintEval("Eval vs random (greedy)", random_eval, option.eval_games);
  PrintEval("Eval vs optimal (greedy)", optimal_eval, option.eval_games);

  if (!option.export_json.empty()) {
    if (!ExportTrainingJson(option.export_json, option, agent, win_total,
                            draw_total, loss_total, random_eval,
                            optimal_eval)) {
      return 1;
    }
    cout << "Exported demo json: " << option.export_json << "\n";
  }

  if (option.show_sample) {
    cout << "\n";
    ShowSampleGame(env, agent);
  }
  if (option.play) {
    cout << "\n";
    PlayInteractive(env, agent);
  }

  return 0;
}
