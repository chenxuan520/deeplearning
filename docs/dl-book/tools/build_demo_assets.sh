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

cleanup_failed_download() {
  local target="$1"
  if [ -f "${target}" ] && [ ! -s "${target}" ]; then
    rm -f "${target}"
  fi
}

download_mnist_data() {
  mkdir -p "${SRC_DIR}/demo/mnist/mnist"
  retry wget https://gitee.com/chenxuan520/deeplearning/releases/download/v0.0.1-beta/t10k-labels-idx1-ubyte -O "${SRC_DIR}/demo/mnist/mnist/t10k-labels-idx1-ubyte"
  retry wget https://gitee.com/chenxuan520/deeplearning/releases/download/v0.0.1-beta/train-labels-idx1-ubyte -O "${SRC_DIR}/demo/mnist/mnist/train-labels-idx1-ubyte"
  retry wget https://gitee.com/chenxuan520/deeplearning/releases/download/v0.0.1-beta/t10k-images-idx3-ubyte -O "${SRC_DIR}/demo/mnist/mnist/t10k-images-idx3-ubyte"
  retry wget https://gitee.com/chenxuan520/deeplearning/releases/download/v0.0.1-beta/train-images-idx3-ubyte -O "${SRC_DIR}/demo/mnist/mnist/train-images-idx3-ubyte"
}

download_mnist_model() {
  mkdir -p "${SRC_DIR}/demo/mnist/mnist"
  retry wget https://github.com/chenxuan520/deeplearning/releases/download/v0.0.1-beta/demo.v2.param -O "${SRC_DIR}/demo/mnist/mnist/demo.v2.param" || cleanup_failed_download "${SRC_DIR}/demo/mnist/mnist/demo.v2.param"
  if [ ! -s "${SRC_DIR}/demo/mnist/mnist/demo.v2.param" ]; then
    retry wget https://gitee.com/chenxuan520/deeplearning/releases/download/v0.0.1-beta/demo.v2.param -O "${SRC_DIR}/demo/mnist/mnist/demo.v2.param" || cleanup_failed_download "${SRC_DIR}/demo/mnist/mnist/demo.v2.param"
  fi
  if [ ! -s "${SRC_DIR}/demo/mnist/mnist/demo.v2.param" ] && [ ! -s "${SRC_DIR}/demo/mnist/mnist/demo.param" ]; then
    retry wget https://github.com/chenxuan520/deeplearning/releases/download/v0.0.1-beta/demo.param -O "${SRC_DIR}/demo/mnist/mnist/demo.param" || cleanup_failed_download "${SRC_DIR}/demo/mnist/mnist/demo.param"
  fi
  if [ ! -s "${SRC_DIR}/demo/mnist/mnist/demo.v2.param" ] && [ ! -s "${SRC_DIR}/demo/mnist/mnist/demo.param" ]; then
    retry wget https://gitee.com/chenxuan520/deeplearning/releases/download/v0.0.1-beta/demo.param -O "${SRC_DIR}/demo/mnist/mnist/demo.param" || cleanup_failed_download "${SRC_DIR}/demo/mnist/mnist/demo.param"
  fi
}

build_binaries() {
  (cd "${SRC_DIR}" && ./build.sh false Release)
}

build_mnist_asset() {
  mkdir -p "${OUT_DIR}/mnist"
  if [ ! -s "${SRC_DIR}/demo/mnist/mnist/demo.v2.param" ] && [ ! -s "${SRC_DIR}/demo/mnist/mnist/demo.param" ]; then
    download_mnist_model
  fi
  local model_file=""
  if [ -s "${SRC_DIR}/demo/mnist/mnist/demo.v2.param" ]; then
    model_file="demo/mnist/mnist/demo.v2.param"
  elif [ -s "${SRC_DIR}/demo/mnist/mnist/demo.param" ]; then
    model_file="demo/mnist/mnist/demo.param"
  else
    echo "MNIST model snapshot not found" >&2
    exit 1
  fi
  echo "Using MNIST model snapshot: ${model_file}"
  (cd "${SRC_DIR}" && ./bin/web_model_export \
    --type mlp \
    --model "${model_file}" \
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
