#pragma once
#include <algorithm>
#include <random>

namespace deeplearning {

class Random {
public:
  Random(int min_num, int max_num_not_include, int seed)
      : gen(seed), distr(min_num, max_num_not_include - 1) {}
  inline int CreateRandom() { return distr(gen); }

public:
  template <typename T> static void RandomShuffle(T &vec) {
    std::random_device rand_dev;
    std::mt19937 rand_gen(rand_dev());
    std::shuffle(vec.begin(), vec.end(), rand_gen);
  }

private:
  std::mt19937_64 gen;
  std::uniform_int_distribution<int> distr;
};

} // namespace deeplearning
