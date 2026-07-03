#include "rl/tic_tac_toe_env.h"

#include "util/random.h"

#include <algorithm>
#include <sstream>

namespace deeplearning {

namespace {

constexpr int kBoardSize = 9;

int CellValue(TicTacToeEnv::Cell cell) { return static_cast<int>(cell); }

TicTacToeEnv::GameResult CheckBoardResult(const std::array<TicTacToeEnv::Cell, 9> &board) {
  for (int line = 0; line < 3; line++) {
    const int base = line * 3;
    if (board[base] != TicTacToeEnv::Cell::EMPTY &&
        board[base] == board[base + 1] && board[base] == board[base + 2]) {
      return board[base] == TicTacToeEnv::Cell::X ? TicTacToeEnv::GameResult::X_WIN
                                                 : TicTacToeEnv::GameResult::O_WIN;
    }
  }
  for (int col = 0; col < 3; col++) {
    if (board[col] != TicTacToeEnv::Cell::EMPTY &&
        board[col] == board[col + 3] && board[col] == board[col + 6]) {
      return board[col] == TicTacToeEnv::Cell::X ? TicTacToeEnv::GameResult::X_WIN
                                                 : TicTacToeEnv::GameResult::O_WIN;
    }
  }
  if (board[0] != TicTacToeEnv::Cell::EMPTY && board[0] == board[4] &&
      board[0] == board[8]) {
    return board[0] == TicTacToeEnv::Cell::X ? TicTacToeEnv::GameResult::X_WIN
                                             : TicTacToeEnv::GameResult::O_WIN;
  }
  if (board[2] != TicTacToeEnv::Cell::EMPTY && board[2] == board[4] &&
      board[2] == board[6]) {
    return board[2] == TicTacToeEnv::Cell::X ? TicTacToeEnv::GameResult::X_WIN
                                             : TicTacToeEnv::GameResult::O_WIN;
  }

  for (TicTacToeEnv::Cell cell : board) {
    if (cell == TicTacToeEnv::Cell::EMPTY) {
      return TicTacToeEnv::GameResult::ONGOING;
    }
  }
  return TicTacToeEnv::GameResult::DRAW;
}

std::vector<int> LegalActionsForBoard(const std::array<TicTacToeEnv::Cell, 9> &board) {
  std::vector<int> actions;
  for (int i = 0; i < kBoardSize; i++) {
    if (board[static_cast<size_t>(i)] == TicTacToeEnv::Cell::EMPTY) {
      actions.push_back(i);
    }
  }
  return actions;
}

int MinimaxScore(const std::array<TicTacToeEnv::Cell, 9> &board,
                 TicTacToeEnv::Cell player) {
  TicTacToeEnv::GameResult result = CheckBoardResult(board);
  if (result == TicTacToeEnv::GameResult::X_WIN) {
    return 1;
  }
  if (result == TicTacToeEnv::GameResult::O_WIN) {
    return -1;
  }
  if (result == TicTacToeEnv::GameResult::DRAW) {
    return 0;
  }

  std::vector<int> legal_actions = LegalActionsForBoard(board);
  TicTacToeEnv::Cell opponent =
      player == TicTacToeEnv::Cell::X ? TicTacToeEnv::Cell::O : TicTacToeEnv::Cell::X;

  if (player == TicTacToeEnv::Cell::X) {
    int best = -1000;
    for (int action : legal_actions) {
      std::array<TicTacToeEnv::Cell, 9> next_board = board;
      next_board[static_cast<size_t>(action)] = player;
      best = std::max(best, MinimaxScore(next_board, opponent));
    }
    return best;
  }

  int best = 1000;
  for (int action : legal_actions) {
    std::array<TicTacToeEnv::Cell, 9> next_board = board;
    next_board[static_cast<size_t>(action)] = player;
    best = std::min(best, MinimaxScore(next_board, opponent));
  }
  return best;
}

} // namespace

int TicTacToeEnv::EncodeBoardKey(const std::array<Cell, 9> &board) {
  int key = 0;
  int base = 1;
  for (int i = 0; i < kBoardSize; i++) {
    key += CellValue(board[i]) * base;
    base *= 3;
  }
  return key;
}

int TicTacToeEnv::ChooseRandomAction(const std::vector<int> &legal_actions,
                                     int seed) {
  if (legal_actions.empty()) {
    return -1;
  }
  Random random(0, static_cast<int>(legal_actions.size()), seed);
  return legal_actions[static_cast<size_t>(random.CreateRandom())];
}

int TicTacToeEnv::ChooseOptimalAction(const std::array<Cell, 9> &board,
                                      Cell player) {
  std::vector<int> legal_actions = LegalActionsForBoard(board);
  if (legal_actions.empty()) {
    return -1;
  }

  int best_action = legal_actions.front();
  if (player == Cell::X) {
    int best_score = -1000;
    for (int action : legal_actions) {
      std::array<Cell, 9> next_board = board;
      next_board[static_cast<size_t>(action)] = player;
      int score = MinimaxScore(next_board, Cell::O);
      if (score > best_score) {
        best_score = score;
        best_action = action;
      }
    }
    return best_action;
  }

  int best_score = 1000;
  for (int action : legal_actions) {
    std::array<Cell, 9> next_board = board;
    next_board[static_cast<size_t>(action)] = player;
    int score = MinimaxScore(next_board, Cell::X);
    if (score < best_score) {
      best_score = score;
      best_action = action;
    }
  }
  return best_action;
}

TicTacToeEnv::RC TicTacToeEnv::Reset() {
  board_.fill(Cell::EMPTY);
  agent_turn_ = true;
  result_ = GameResult::ONGOING;
  err_msg_.clear();
  return RC::SUCCESS;
}

void TicTacToeEnv::set_opponent_random_seed(int seed) {
  opponent_random_seed_ = seed;
}

bool TicTacToeEnv::IsLegalAction(int action) const {
  if (action < 0 || action >= kBoardSize) {
    return false;
  }
  return board_[static_cast<size_t>(action)] == Cell::EMPTY;
}

bool TicTacToeEnv::IsAgentTurn() const { return agent_turn_; }

bool TicTacToeEnv::IsTerminal() const { return result_ != GameResult::ONGOING; }

TicTacToeEnv::GameResult TicTacToeEnv::Result() const { return result_; }

std::vector<int> TicTacToeEnv::LegalActions() const {
  if (!agent_turn_ || IsTerminal()) {
    return {};
  }
  return LegalActionsForBoard(board_);
}

int TicTacToeEnv::AgentStateKey() const {
  if (!agent_turn_ || IsTerminal()) {
    return -1;
  }
  return EncodeBoardKey(board_);
}

TicTacToeEnv::GameResult TicTacToeEnv::CheckResult() const {
  return CheckBoardResult(board_);
}

TicTacToeEnv::RC TicTacToeEnv::ApplyMove(int action, Cell player, StepResult &out) {
  out = StepResult{};
  if (IsTerminal()) {
    err_msg_ = "ApplyMove: game already over";
    return RC::GAME_OVER;
  }
  if (!IsLegalAction(action)) {
    err_msg_ = "ApplyMove: illegal action";
    return RC::INVALID_ACTION;
  }

  board_[static_cast<size_t>(action)] = player;
  result_ = CheckResult();
  out.done = IsTerminal();
  out.result = result_;

  if (result_ == GameResult::X_WIN) {
    out.reward = 1.0;
  } else if (result_ == GameResult::O_WIN) {
    out.reward = -1.0;
  } else if (result_ == GameResult::DRAW) {
    out.reward = 0.0;
  }

  if (!out.done) {
    agent_turn_ = !agent_turn_;
    out.agent_state_key = agent_turn_ ? AgentStateKey() : -1;
  }
  return RC::SUCCESS;
}

TicTacToeEnv::RC TicTacToeEnv::StepAgent(int action, StepResult &out) {
  if (!agent_turn_) {
    err_msg_ = "StepAgent: not agent turn";
    return RC::NOT_AGENT_TURN;
  }
  return ApplyMove(action, Cell::X, out);
}

TicTacToeEnv::RC TicTacToeEnv::StepOpponentRandom(int &action, StepResult &out) {
  if (agent_turn_) {
    err_msg_ = "StepOpponentRandom: still agent turn";
    return RC::NOT_AGENT_TURN;
  }
  if (IsTerminal()) {
    err_msg_ = "StepOpponentRandom: game already over";
    return RC::GAME_OVER;
  }

  std::vector<int> legal_actions = LegalActionsForBoard(board_);
  action = ChooseRandomAction(legal_actions, opponent_random_seed_++);
  return ApplyMove(action, Cell::O, out);
}

TicTacToeEnv::RC TicTacToeEnv::StepOpponent(int action, StepResult &out) {
  if (agent_turn_) {
    err_msg_ = "StepOpponent: still agent turn";
    return RC::NOT_AGENT_TURN;
  }
  if (IsTerminal()) {
    err_msg_ = "StepOpponent: game already over";
    return RC::GAME_OVER;
  }
  return ApplyMove(action, Cell::O, out);
}

TicTacToeEnv::RC TicTacToeEnv::StepOpponentOptimal(int &action, StepResult &out) {
  if (agent_turn_) {
    err_msg_ = "StepOpponentOptimal: still agent turn";
    return RC::NOT_AGENT_TURN;
  }
  if (IsTerminal()) {
    err_msg_ = "StepOpponentOptimal: game already over";
    return RC::GAME_OVER;
  }

  action = ChooseOptimalAction(board_, Cell::O);
  if (action < 0) {
    err_msg_ = "StepOpponentOptimal: no legal action";
    return RC::INVALID_ACTION;
  }
  return ApplyMove(action, Cell::O, out);
}

std::string TicTacToeEnv::Render() const {
  std::ostringstream oss;
  for (int row = 0; row < 3; row++) {
    for (int col = 0; col < 3; col++) {
      Cell cell = board_[static_cast<size_t>(row * 3 + col)];
      char mark = '.';
      if (cell == Cell::X) {
        mark = 'X';
      } else if (cell == Cell::O) {
        mark = 'O';
      }
      oss << mark;
      if (col < 2) {
        oss << '|';
      }
    }
    if (row < 2) {
      oss << "\n-+-+-\n";
    }
  }
  return oss.str();
}

std::string TicTacToeEnv::err_msg() const { return err_msg_; }

} // namespace deeplearning
