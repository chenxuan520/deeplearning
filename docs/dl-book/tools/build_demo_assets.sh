#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC_DIR="${ROOT_DIR}/src"
OUT_DIR="${ROOT_DIR}/docs/dl-book/assets/demos"

retry() {
  local max=3
  local attempt=1
  until "$@"; do
    if [ "${attempt}" -ge "${max}" ]; then
      return 1
    fi
    echo "Attempt ${attempt} failed; retrying in 10s..."
    attempt=$((attempt + 1))
    sleep 10
  done
}

download_mnist_data() {
  mkdir -p "${SRC_DIR}/demo/mnist/mnist"
  retry wget https://gitee.com/chenxuan520/deeplearning/releases/download/v0.0.1-beta/t10k-labels-idx1-ubyte -O "${SRC_DIR}/demo/mnist/mnist/t10k-labels-idx1-ubyte"
  retry wget https://gitee.com/chenxuan520/deeplearning/releases/download/v0.0.1-beta/train-labels-idx1-ubyte -O "${SRC_DIR}/demo/mnist/mnist/train-labels-idx1-ubyte"
  retry wget https://gitee.com/chenxuan520/deeplearning/releases/download/v0.0.1-beta/t10k-images-idx3-ubyte -O "${SRC_DIR}/demo/mnist/mnist/t10k-images-idx3-ubyte"
  retry wget https://gitee.com/chenxuan520/deeplearning/releases/download/v0.0.1-beta/train-images-idx3-ubyte -O "${SRC_DIR}/demo/mnist/mnist/train-images-idx3-ubyte"
}

build_binaries() {
  (cd "${SRC_DIR}" && ./build.sh false Release)
}

build_mnist_asset() {
  mkdir -p "${OUT_DIR}/mnist"
  if [ ! -f "${SRC_DIR}/demo/mnist/mnist/demo.v2.param" ]; then
    download_mnist_data
    (cd "${SRC_DIR}" && ./bin/mnist)
  fi
  (cd "${SRC_DIR}" && ./bin/web_model_export \
    --type mlp \
    --model demo/mnist/mnist/demo.v2.param \
    --out ../docs/dl-book/assets/demos/mnist/model.json)
}

build_tictactoe_asset() {
  mkdir -p "${OUT_DIR}/tictactoe"
  (cd "${SRC_DIR}" && ./bin/rl_tictactoe \
    --episodes 30000 \
    --eval-games 1000 \
    --rand-seed 0 \
    --export-json ../docs/dl-book/assets/demos/tictactoe/q_table.json)
}

build_binaries
build_mnist_asset
build_tictactoe_asset

echo "Generated demo assets:"
find "${OUT_DIR}" -maxdepth 3 -type f -print
