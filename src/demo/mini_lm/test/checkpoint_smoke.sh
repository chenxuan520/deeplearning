#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MINI_LM="${ROOT_DIR}/bin/mini_lm"

if [[ ! -x "${MINI_LM}" ]]; then
  echo "mini_lm binary not found: ${MINI_LM}" >&2
  echo "Build first, for example: cd ${ROOT_DIR} && ./build.sh false Debug" >&2
  exit 1
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

MODEL="${WORK_DIR}/smoke.param"
CHECKPOINT="${WORK_DIR}/smoke.param.ckpt"
CORPUS="${WORK_DIR}/corpus.txt"

cat > "${CORPUS}" <<'EOF'
alice was beginning to get very tired of sitting by her sister on the bank
alice was thinking that the world was curious and bright
the rabbit said hello to alice and alice said hello to the rabbit
EOF

"${MINI_LM}" init \
  --model "${MODEL}" \
  --tokenizer word \
  --max-vocab-size 12 \
  --corpus-file "${CORPUS}" \
  --model-dim 8 \
  --head-num 1 \
  --feed-forward-dim 16 \
  --block-num 0 \
  --context-size 3 \
  --rand-seed 7 >/tmp/mini_lm_checkpoint_init.log

"${MINI_LM}" train \
  --model "${MODEL}" \
  --corpus-file "${CORPUS}" \
  --epochs 2 \
  --learning-rate 0.005 \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint-every 1 \
  --progress-every-sec 0 \
  --early-stop-loss 0 \
  --log-every 1 >/tmp/mini_lm_checkpoint_train.log

test -s "${CHECKPOINT}"
test -s "${CHECKPOINT}.vocab"
test -s "${CHECKPOINT}.train"
grep -q '^completed_epoch=2$' "${CHECKPOINT}.train"

"${MINI_LM}" train \
  --model "${MODEL}" \
  --corpus-file "${CORPUS}" \
  --epochs 3 \
  --learning-rate 0.005 \
  --checkpoint "${CHECKPOINT}" \
  --resume-checkpoint \
  --progress-every-sec 0 \
  --early-stop-loss 0 \
  --log-every 1 >/tmp/mini_lm_checkpoint_resume.log

grep -q '^completed_epoch=3$' "${CHECKPOINT}.train"
grep -q 'Resumed checkpoint: completed_epoch=2' /tmp/mini_lm_checkpoint_resume.log

rm -f "${MODEL}" "${MODEL}.vocab"
"${MINI_LM}" train \
  --model "${MODEL}" \
  --corpus-file "${CORPUS}" \
  --epochs 3 \
  --learning-rate 0.005 \
  --checkpoint "${CHECKPOINT}" \
  --resume-checkpoint \
  --progress-every-sec 0 >/tmp/mini_lm_checkpoint_sync.log

test -s "${MODEL}"
test -s "${MODEL}.vocab"
grep -q 'Saved checkpoint weights to model' /tmp/mini_lm_checkpoint_sync.log

"${MINI_LM}" generate \
  --model "${MODEL}" \
  --prompt "alice" \
  --generate-num 4 >/tmp/mini_lm_checkpoint_generate.log

grep -q '^Generated:' /tmp/mini_lm_checkpoint_generate.log
echo "mini_lm checkpoint smoke passed"
