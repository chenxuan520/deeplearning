// this file is to include all test header
#include "activate_test.h"
#include "cnn/cnn_test.h"
#include "dropout_test.h"
#include "embedding/word2vec_test.h"
#include "grad_clip_test.h"
#include "lr_scheduler_test.h"
#include "neural_network_batch_test.h"
#include "neural_network_loader_test.h"
#include "neural_network_test.h"
#include "optimizer_test.h"
#include "rl/rl_test.h"
#include "rnn/rnn_test.h"
#include "softmax/std_softmax_test.h"
#include "test.h"
#include "transformer/transformer_test.h"

ARGC_FUNC {
  if (argc == 2) {
    REGEX_FILT_TEST(argv[1]);
  }
}
