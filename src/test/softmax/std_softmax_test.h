#pragma once

#include "softmax/softmax_factory.h"
#include "test.h"
#include <vector>

TEST(SoftmaxStd, Demo) {
  auto softmax = deeplearning::SoftmaxFactory::Create(
      deeplearning::SoftmaxType::SOFTMAX_STD);
  MUST_TRUE(softmax != nullptr, "nullptr");
  std::vector<double> arr = {3, 3};
  softmax->Normalize(arr, arr);
  MUST_EQUAL(arr[0], 0.5);
  MUST_EQUAL(arr[1], 0.5);
}
