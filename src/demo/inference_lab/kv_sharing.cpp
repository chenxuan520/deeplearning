// 第 19 章 §9.4: 固定权重的四层单头注意力实验,只依赖 C++17 标准库。
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
constexpr int kLayers = 4;
constexpr int kDim = 4;
using Vec = std::array<double, kDim>;
using Sequence = std::vector<Vec>;
using Owners = std::array<int, kLayers>;
const Owners kIndependent{0, 1, 2, 3};
const Owners kShared{0, 0, 0, 0};

struct KV {
  Sequence keys;
  Sequence values;
};

struct Result {
  Sequence outputs;
  size_t cached_scalars = 0;
  size_t projected_pairs = 0;
};

void Check(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}

// 权重是可复现的小数,没有训练;不同层、不同投影使用不同矩阵。
Vec Project(const Vec &x, int layer, int kind) {
  Vec y{};
  for (int i = 0; i < kDim; ++i) {
    for (int j = 0; j < kDim; ++j) {
      const double w = 0.3 * std::sin((layer + 1) * 11 + kind * 7 + i * 5 + j * 3);
      y[i] += w * x[j];
    }
  }
  return y;
}

Vec Embed(int token, size_t position) {
  Vec x{};
  for (int d = 0; d < kDim; ++d) {
    x[d] = std::sin((token + 1) * (d + 1) * 0.3)
         + 0.1 * std::cos((position + 1) * (d + 1) * 0.2);
  }
  return x;
}

Vec Attend(const Vec &query, const KV &cache, size_t visible) {
  Check(visible > 0 && visible <= cache.keys.size(), "Invalid causal prefix");
  std::vector<double> scores(visible, 0.0);
  for (size_t t = 0; t < visible; ++t) {
    for (int d = 0; d < kDim; ++d) scores[t] += query[d] * cache.keys[t][d];
    scores[t] /= std::sqrt(static_cast<double>(kDim));
  }
  const double maximum = *std::max_element(scores.begin(), scores.end());
  double sum = 0.0;
  for (double &score : scores) {
    score = std::exp(score - maximum);
    sum += score;
  }
  Vec context{};
  for (size_t t = 0; t < visible; ++t) {
    for (int d = 0; d < kDim; ++d) {
      context[d] += scores[t] / sum * cache.values[t][d];
    }
  }
  return context;
}

Vec FinishLayer(const Vec &input, const Vec &context, int layer) {
  const Vec projected = Project(context, layer, 3);
  Vec hidden = input;
  for (int d = 0; d < kDim; ++d) hidden[d] += 0.25 * projected[d];
  const Vec feed_forward = Project(hidden, layer, 4);
  for (int d = 0; d < kDim; ++d) hidden[d] += 0.1 * std::tanh(feed_forward[d]);
  return hidden;
}

// 参考路径: 给定整段输入,逐层重算所有位置;显式限制每个位置只能看前缀。
Sequence FullForward(const std::vector<int> &tokens, const Owners &owners) {
  Sequence hidden;
  for (size_t t = 0; t < tokens.size(); ++t) hidden.push_back(Embed(tokens[t], t));
  std::array<KV, kLayers> caches;
  for (int layer = 0; layer < kLayers; ++layer) {
    const int owner = owners[layer];
    Check(owner >= 0 && owner <= layer && owners[owner] == owner, "Invalid KV owner");
    if (owner == layer) {
      for (const Vec &x : hidden) {
        caches[layer].keys.push_back(Project(x, layer, 1));
        caches[layer].values.push_back(Project(x, layer, 2));
      }
    }
    Sequence next;
    for (size_t t = 0; t < tokens.size(); ++t) {
      const Vec query = Project(hidden[t], layer, 0);
      next.push_back(FinishLayer(hidden[t], Attend(query, caches[owner], t + 1), layer));
    }
    hidden = std::move(next);
  }
  return hidden;
}

// 增量路径: 每次只输入一个新 token。owner 层追加一次,其余层只读同一份 K/V。
Result CachedForward(const std::vector<int> &tokens, const Owners &owners) {
  std::array<KV, kLayers> caches;
  Result result;
  for (size_t t = 0; t < tokens.size(); ++t) {
    Vec hidden = Embed(tokens[t], t);
    for (int layer = 0; layer < kLayers; ++layer) {
      const int owner = owners[layer];
      Check(owner >= 0 && owner <= layer && owners[owner] == owner, "Invalid KV owner");
      if (owner == layer) {
        caches[layer].keys.push_back(Project(hidden, layer, 1));
        caches[layer].values.push_back(Project(hidden, layer, 2));
        ++result.projected_pairs;
      }
      const KV &source = caches[owner];  // 引用,不会复制向量。
      const Vec query = Project(hidden, layer, 0);
      hidden = FinishLayer(hidden, Attend(query, source, t + 1), layer);
    }
    result.outputs.push_back(hidden);
  }
  for (const KV &cache : caches) {
    result.cached_scalars += (cache.keys.size() + cache.values.size()) * kDim;
  }
  return result;
}

double MaxError(const Sequence &a, const Sequence &b) {
  Check(a.size() == b.size(), "Output lengths differ");
  double error = 0.0;
  for (size_t t = 0; t < a.size(); ++t) {
    for (int d = 0; d < kDim; ++d) error = std::max(error, std::abs(a[t][d] - b[t][d]));
  }
  return error;
}

std::vector<int> Tokens(int length) {
  std::vector<int> tokens;
  for (int t = 0; t < length; ++t) tokens.push_back((t * 3 + 1) % 7);
  return tokens;
}

void SelfCheck() {
  for (const Owners owners : {kIndependent, kShared, Owners{0, 0, 2, 2}}) {
    for (int length : {1, 2, 8, 17}) {
      const auto tokens = Tokens(length);
      const auto cached = CachedForward(tokens, owners);
      Check(MaxError(cached.outputs, FullForward(tokens, owners)) < 1e-12,
            "Cached/full forward mismatch");
      // 每个前缀单独重算,避免只比较最后一步而漏掉中途错误。
      for (int n = 1; n <= length; ++n) {
        const std::vector<int> prefix(tokens.begin(), tokens.begin() + n);
        Check(MaxError(Sequence{cached.outputs[n - 1]},
                       Sequence{FullForward(prefix, owners).back()}) < 1e-12,
              "Prefix recomputation mismatch");
      }
      int sources = 0;
      for (int layer = 0; layer < kLayers; ++layer) sources += owners[layer] == layer;
      Check(cached.cached_scalars == static_cast<size_t>(sources * 2 * length * kDim),
            "Unexpected retained KV size");
      Check(cached.projected_pairs == static_cast<size_t>(sources * length),
            "A consumer projected or appended extra KV");
    }
    auto changed = Tokens(8);
    const auto original = FullForward(changed, owners);
    changed[6] += 11;
    changed[7] += 19;
    const auto future_changed = FullForward(changed, owners);
    Check(MaxError(Sequence(original.begin(), original.begin() + 6),
                   Sequence(future_changed.begin(), future_changed.begin() + 6)) < 1e-12,
          "Future tokens leaked through the causal mask");
  }
}
}  // namespace

int main() {
  try {
    SelfCheck();
    std::cout << "Four layers, one head, dimension=4, fixed untrained weights\n"
              << "Retained K/V payload only; excludes weights, activations and allocator overhead.\n"
              << "tokens  independent_bytes  shared_bytes  independent_KV_pairs  shared_KV_pairs\n";
    for (int length : {8, 32, 128}) {
      const auto independent = CachedForward(Tokens(length), kIndependent);
      const auto shared = CachedForward(Tokens(length), kShared);
      std::cout << std::setw(6) << length
                << std::setw(19) << independent.cached_scalars * sizeof(double)
                << std::setw(14) << shared.cached_scalars * sizeof(double)
                << std::setw(22) << independent.projected_pairs
                << std::setw(17) << shared.projected_pairs << '\n';
    }
    const auto tokens = Tokens(8);
    const auto independent = CachedForward(tokens, kIndependent);
    const auto shared = CachedForward(tokens, kShared);
    std::cout << std::scientific
              << "Shared cached vs shared recomputed max error: "
              << MaxError(shared.outputs, FullForward(tokens, kShared)) << '\n'
              << "Independent vs shared output difference (different architectures): "
              << MaxError(independent.outputs, shared.outputs) << '\n'
              << "PASS: full/prefix equivalence, causal mask, grouped sharing and KV accounting\n";
  } catch (const std::exception &e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
