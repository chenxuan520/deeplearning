// 第 21 章 §7: 用三词表 Markov 模型隔离验证采样算法,不需要训练或 GPU。
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
constexpr int kVocab = 3;
constexpr int kLength = 6;
constexpr int kSamples = 100000;
constexpr int kMaxDraft = 5;
constexpr double kRoundCost = 1.0;
constexpr double kTargetRowCost = 0.15;
constexpr double kDraftTokenCost = 0.03;
using Probs = std::array<double, kVocab>;
using Model = std::array<Probs, kVocab>;
const Model kTarget{{{0.6, 0.3, 0.1}, {0.15, 0.7, 0.15}, {0.3, 0.1, 0.6}}};
const Model kDraft{{{0.2, 0.3, 0.5}, {0.1, 0.8, 0.1}, {0.4, 0.2, 0.4}}};

enum class Mode { Plain, Fixed, Adaptive };
struct Stats {
  uint64_t rounds = 0;
  uint64_t target_rows = 0;
  uint64_t draft_tokens = 0;
  uint64_t accepted = 0;
};

void Check(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}

double Uniform(std::mt19937_64 &rng) {
  return static_cast<double>(rng() >> 11) / 9007199254740992.0;
}

int Sample(const Probs &probs, std::mt19937_64 &rng) {
  const double u = Uniform(rng);
  double cumulative = 0.0;
  int last_positive = 0;
  for (int x = 0; x < kVocab; ++x) {
    if (probs[x] > 0.0) last_positive = x;
    cumulative += probs[x];
    if (u < cumulative) return x;
  }
  return last_positive;  // 仅兜住浮点求和略小于 1 的尾差。
}

Probs Residual(const Probs &p, const Probs &q) {
  Probs r{};
  double mass = 0.0;
  for (int x = 0; x < kVocab; ++x) {
    r[x] = std::max(0.0, p[x] - q[x]);
    mass += r[x];
  }
  Check(mass > 0.0, "Residual is only defined when rejection is possible");
  for (double &value : r) value /= mass;
  return r;
}

// 单位是人为指定的模拟成本,不是时间。target_rows 包括全部接受时的 bonus 分布。
double Cost(int draft_length) {
  return kRoundCost + kTargetRowCost * (draft_length + 1) + kDraftTokenCost * draft_length;
}

int ChooseLength(int state, int limit, const Model &p, const Model &q) {
  // 必须在抽取本轮草稿前选择长度。这里只用当前已提交的前缀,
  // 不根据抽到的候选内容挑选,从而可直接沿用固定长度算法的分布保证。
  Probs alive{};
  alive[state] = 1.0;
  double expected_tokens = 1.0;
  double best_score = expected_tokens / Cost(0);
  int best_length = 0;
  for (int length = 1; length <= limit; ++length) {
    Probs next{};
    for (int previous = 0; previous < kVocab; ++previous) {
      for (int x = 0; x < kVocab; ++x) {
        next[x] += alive[previous] * std::min(p[previous][x], q[previous][x]);
      }
    }
    alive = next;
    expected_tokens += std::accumulate(alive.begin(), alive.end(), 0.0);
    const double score = expected_tokens / Cost(length);
    if (score > best_score) {
      best_score = score;
      best_length = length;
    }
  }
  return best_length;
}

std::vector<int> Generate(Mode mode, const Model &p, const Model &q,
                          std::mt19937_64 &rng, Stats &stats) {
  std::vector<int> output;
  int state = 0;  // 所有实验都从同一个已知前缀 A 开始。
  while (output.size() < kLength) {
    const int remaining = kLength - static_cast<int>(output.size());
    const int limit = std::min(kMaxDraft, remaining - 1);
    const int length = mode == Mode::Plain ? 0
                     : mode == Mode::Fixed ? limit : ChooseLength(state, limit, p, q);
    ++stats.rounds;
    stats.target_rows += length + 1;
    stats.draft_tokens += length;

    std::vector<int> draft;
    std::vector<Probs> proposals;
    std::vector<Probs> targets;
    int temporary = state;
    for (int i = 0; i < length; ++i) {
      proposals.push_back(q[temporary]);
      draft.push_back(Sample(proposals.back(), rng));
      temporary = draft.back();
    }
    // 真模型可用一次 causal forward 并行取得这些分布;
    // 这里逐行查概率表,一轮只记作一次目标模型调用。
    temporary = state;
    for (int token : draft) {
      targets.push_back(p[temporary]);
      temporary = token;
    }
    targets.push_back(p[temporary]);  // 全部接受后的额外一个 token。

    bool rejected = false;
    for (int i = 0; i < length; ++i) {
      const int y = draft[i];
      const double accept = std::min(1.0, targets[i][y] / proposals[i][y]);
      if (Uniform(rng) < accept) {
        output.push_back(y);
        state = y;
        ++stats.accepted;
      } else {
        // 不能直接从 p 再抽一次;要补回草稿未覆盖的概率质量。
        state = Sample(Residual(targets[i], proposals[i]), rng);
        output.push_back(state);
        rejected = true;
        break;  // 后续草稿及其条件分布作废。
      }
    }
    if (!rejected) {
      state = Sample(targets.back(), rng);
      output.push_back(state);
    }
  }
  return output;
}

void CheckIdentity(const Probs &p, const Probs &q) {
  double accepted_mass = 0.0;
  for (int x = 0; x < kVocab; ++x) accepted_mass += std::min(p[x], q[x]);
  const double rejected_mass = 1.0 - accepted_mass;
  const Probs r = rejected_mass > 1e-14 ? Residual(p, q) : Probs{};
  for (int x = 0; x < kVocab; ++x) {
    Check(std::abs(std::min(p[x], q[x]) + rejected_mass * r[x] - p[x]) < 1e-12,
          "Acceptance + correction does not restore target distribution");
  }
}

void SelfCheck() {
  for (int state = 0; state < kVocab; ++state) CheckIdentity(kTarget[state], kDraft[state]);
  CheckIdentity({0.6, 0.3, 0.1}, {0.6, 0.3, 0.1});  // 从不拒绝。
  CheckIdentity({1.0, 0.0, 0.0}, {0.0, 1.0, 0.0});  // 支持集不相交。
  CheckIdentity({0.5, 0.5, 0.0}, {0.0, 0.5, 0.5});  // 零概率项。
  const Model only_a{{{1, 0, 0}, {1, 0, 0}, {1, 0, 0}}};
  const Model only_b{{{0, 1, 0}, {0, 1, 0}, {0, 1, 0}}};
  for (Mode mode : {Mode::Fixed, Mode::Adaptive}) {
    std::mt19937_64 rng(7);
    Stats stats;
    const auto output = Generate(mode, only_a, only_b, rng, stats);
    Check(output == std::vector<int>(kLength, 0), "Rejected suffix was committed");
    Check(stats.accepted == 0, "A zero-probability target token was accepted");
  }
  std::mt19937_64 rng(11);
  Stats stats;
  Generate(Mode::Fixed, kTarget, kTarget, rng, stats);
  Check(stats.rounds == 1 && stats.accepted == kMaxDraft, "All-accepted/bonus path failed");
}

// 六个 token 的完整联合分布有 3^6=729 种,逐条相乘得到独立的理论答案。
std::vector<double> ExactJoint() {
  int count = 1;
  for (int i = 0; i < kLength; ++i) count *= kVocab;
  std::vector<double> exact(count);
  for (int code = 0; code < count; ++code) {
    int rest = code;
    int state = 0;
    double probability = 1.0;
    for (int i = 0; i < kLength; ++i) {
      const int token = rest % kVocab;
      rest /= kVocab;
      probability *= kTarget[state][token];
      state = token;
    }
    exact[code] = probability;
  }
  return exact;
}

void Run(Mode mode, const char *name, uint64_t seed) {
  const auto exact = ExactJoint();
  std::vector<int> counts(exact.size(), 0);
  std::mt19937_64 rng(seed);
  Stats stats;
  for (int sample = 0; sample < kSamples; ++sample) {
    const auto output = Generate(mode, kTarget, kDraft, rng, stats);
    int code = 0;
    int place = 1;
    for (int token : output) {
      code += token * place;
      place *= kVocab;
    }
    ++counts[code];
  }
  double tv = 0.0;
  for (size_t i = 0; i < exact.size(); ++i) {
    tv += std::abs(static_cast<double>(counts[i]) / kSamples - exact[i]) * 0.5;
  }
  // 这只是固定种子下的统计回归检查;分布保证来自概率恒等式与正确的条件前缀。
  Check(tv < 0.04, "Joint distribution sanity check failed");
  const double total_tokens = kSamples * kLength;
  const double simulated_cost = kRoundCost * stats.rounds
                              + kTargetRowCost * stats.target_rows
                              + kDraftTokenCost * stats.draft_tokens;
  std::cout << std::left << std::setw(10) << name << std::right
            << std::setw(12) << stats.rounds
            << std::setw(13) << stats.target_rows
            << std::setw(13) << stats.draft_tokens
            << std::setw(12) << stats.accepted
            << std::setw(12) << std::fixed << std::setprecision(5) << tv
            << std::setw(14) << simulated_cost / total_tokens << '\n';
}
}  // namespace

int main() {
  try {
    SelfCheck();
    const Probs p = kTarget[0], q = kDraft[0], r = Residual(p, q);
    std::cout << "One-step example: columns A B C\n"
              << "target p       0.60 0.30 0.10\n"
              << "draft q        0.20 0.30 0.50\n"
              << "accept chance  1.00 1.00 0.20\n"
              << "residual r     " << r[0] << ' ' << r[1] << ' ' << r[2] << '\n'
              << "wrong resample-p result: ";
    for (int x = 0; x < kVocab; ++x) std::cout << std::min(p[x], q[x]) + 0.4 * p[x] << ' ';
    std::cout << "\n\n" << kSamples << " sequences, " << kLength
              << " tokens each; same target, different RNG streams.\n"
              << "mode            rounds  target_rows  draft_tokens    accepted    joint_TV    cost/token\n";
    Run(Mode::Plain, "plain", 20261029);
    Run(Mode::Fixed, "fixed-5", 20261030);
    Run(Mode::Adaptive, "adaptive", 20261031);
    std::cout << "Cost = " << kRoundCost << " * rounds + " << kTargetRowCost
              << " * target_rows + " << kDraftTokenCost
              << " * draft_tokens (simulation, not GPU time).\n"
              << "Adaptive chooses length before drafting; this is not a DSpark reproduction.\n"
              << "PASS: probability identity, zero support, reject/all-accept paths and joint distribution\n";
  } catch (const std::exception &e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
