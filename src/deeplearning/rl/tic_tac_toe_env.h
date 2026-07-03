#pragma once

#include <array>
#include <string>
#include <vector>

namespace deeplearning {

class TicTacToeEnv {
public:
  enum class Cell : int { EMPTY = 0, X = 1, O = 2 };

  enum class RC {
    SUCCESS,
    INVALID_ACTION,
    GAME_OVER,
    NOT_AGENT_TURN,
  };

  enum class GameResult { ONGOING, X_WIN, O_WIN, DRAW };

  struct StepResult {
    double reward = 0.0;
    bool done = false;
    GameResult result = GameResult::ONGOING;
    int agent_state_key = 0;
  };

public:
  RC Reset();
  void set_opponent_random_seed(int seed);
  RC StepAgent(int action, StepResult &out);
  RC StepOpponentRandom(int &action, StepResult &out);
  RC StepOpponentOptimal(int &action, StepResult &out);
  RC StepOpponent(int action, StepResult &out);

  std::vector<int> LegalActions() const;
  int AgentStateKey() const;
  bool IsAgentTurn() const;
  bool IsTerminal() const;
  GameResult Result() const;
  std::string Render() const;
  std::string err_msg() const;

  static int EncodeBoardKey(const std::array<Cell, 9> &board);
  static int ChooseRandomAction(const std::vector<int> &legal_actions, int seed);
  static int ChooseOptimalAction(const std::array<Cell, 9> &board, Cell player);

private:
  RC ApplyMove(int action, Cell player, StepResult &out);
  GameResult CheckResult() const;
  bool IsLegalAction(int action) const;

private:
  std::array<Cell, 9> board_{};
  bool agent_turn_ = true;
  GameResult result_ = GameResult::ONGOING;
  int opponent_random_seed_ = 0;
  std::string err_msg_;
};

} // namespace deeplearning
