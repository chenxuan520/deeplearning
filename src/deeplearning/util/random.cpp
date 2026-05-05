#include "random.h"

namespace deeplearning {

Random::Random(int min_num, int max_num_not_include, int seed)
    : gen(seed), distr(min_num, max_num_not_include - 1) {}

int Random::CreateRandom() { return distr(gen); }

} // namespace deeplearning
