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

RELEASE_BASE="https://github.com/chenxuan520/deeplearning/releases/download/v0.0.7"
# 旧版纯 gitee 渠道(仅兜底;GitHub Release 已有全部文件)
GITEE_RELEASE_BASE="https://gitee.com/chenxuan520/deeplearning/releases/download/v0.0.1-beta"

download_mnist_data() {
  mkdir -p "${SRC_DIR}/demo/mnist/mnist"
  retry wget "${RELEASE_BASE}/t10k-labels-idx1-ubyte" -O "${SRC_DIR}/demo/mnist/mnist/t10k-labels-idx1-ubyte"
  retry wget "${RELEASE_BASE}/train-labels-idx1-ubyte" -O "${SRC_DIR}/demo/mnist/mnist/train-labels-idx1-ubyte"
  retry wget "${RELEASE_BASE}/t10k-images-idx3-ubyte" -O "${SRC_DIR}/demo/mnist/mnist/t10k-images-idx3-ubyte"
  retry wget "${RELEASE_BASE}/train-images-idx3-ubyte" -O "${SRC_DIR}/demo/mnist/mnist/train-images-idx3-ubyte"
}

download_mnist_model() {
  mkdir -p "${SRC_DIR}/demo/mnist/mnist"
  retry wget "${RELEASE_BASE}/demo.param" -O "${SRC_DIR}/demo/mnist/mnist/demo.param" || cleanup_failed_download "${SRC_DIR}/demo/mnist/mnist/demo.param"
  if [ ! -s "${SRC_DIR}/demo/mnist/mnist/demo.param" ]; then
    retry wget "${GITEE_RELEASE_BASE}/demo.param" -O "${SRC_DIR}/demo/mnist/mnist/demo.param" || cleanup_failed_download "${SRC_DIR}/demo/mnist/mnist/demo.param"
  fi
}

build_binaries() {
  mkdir -p "${SRC_DIR}/build"
  (
    cd "${SRC_DIR}/build"
    cmake -DENABLE_DRAW=false -DCMAKE_BUILD_TYPE=Release ..
    cmake --build . --target mnist web_model_export -j2
  )
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

validate_alphazero_assets() {
  python3 - <<'PY'
import hashlib
import json
import urllib.request

base = "https://azgomoku.011203.xyz"
def open_url(url, timeout):
    request = urllib.request.Request(url, headers={"User-Agent": "dl-book-assets-check/1.0"})
    return urllib.request.urlopen(request, timeout=timeout)

with open_url(base + "/model.json", 30) as response:
    manifest = json.load(response)
assert manifest["format"] == "XQPVRN01"
assert manifest["parameter_count"] == 191853
for key in ("file", "engine", "training_curve"):
    assert manifest[key].startswith(base + "/"), (key, manifest[key])

with open_url(manifest["file"], 60) as response:
    weights = response.read()
assert len(weights) == manifest["size_bytes"]
assert hashlib.sha256(weights).hexdigest() == manifest["sha256"]

with open_url(manifest["engine"], 30) as response:
    engine = response.read()
assert (b"XQPVRN01" in engine and b"AlphaZeroGomoku" in engine
        and b"SearchSession" in engine)

with open_url(manifest["training_curve"], 30) as response:
    curve = response.read(8)
assert curve == b"\x89PNG\r\n\x1a\n"
print("Validated AlphaZero browser assets:", manifest["file"])
PY
}

build_binaries
build_mnist_asset
validate_alphazero_assets

echo "Generated demo assets:"
find "${OUT_DIR}" -maxdepth 3 -type f -print
