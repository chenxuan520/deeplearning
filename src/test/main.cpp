// this file is to include all test header
#include "neural_network_loader_test.h"
#include "neural_network_test.h"
#include "test.h"

ARGC_FUNC {
  if (argc == 2) {
    REGEX_FILT_TEST(argv[1]);
  }
}
