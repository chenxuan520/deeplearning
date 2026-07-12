#!/usr/bin/env bash
#
# Fetch public-domain training text for the mini_lm demo and clean it into a
# ready-to-train corpus. Data comes from Project Gutenberg (public domain).
#
# The downloaded raw text and the cleaned corpus are TRAINING DATA and are NOT
# committed to git (see src/demo/mini_lm/.gitignore). Only this script and
# clean_text.py are tracked, so anyone can regenerate the data on demand.
#
# Usage:
#   ./fetch_data.sh                 # download the default book (Alice) into ./data
#   ./fetch_data.sh --out-dir DIR   # choose a different output directory
#   ./fetch_data.sh --book alice|sherlock|all
#
# Output (default): ./data/<name>.raw.txt (raw) and ./data/<name>.txt (cleaned).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/data"
BOOK="alice"

# Small, plain-text, public-domain books. Kept as a simple case table instead of
# a bash-4 associative array so the script also runs on macOS's bash 3.2.
KNOWN_BOOKS="alice sherlock"

book_url() {
  case "$1" in
    alice)    echo "https://www.gutenberg.org/files/11/11-0.txt" ;;
    sherlock) echo "https://www.gutenberg.org/files/1661/1661-0.txt" ;;
    *)        echo "" ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      OUT_DIR="$2"; shift 2 ;;
    --book)
      BOOK="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

download_one() {
  local name="$1"
  local url
  url="$(book_url "$name")"
  if [[ -z "$url" ]]; then
    echo "Unknown book: $name (known: ${KNOWN_BOOKS})" >&2
    exit 1
  fi
  local raw="${OUT_DIR}/${name}.raw.txt"
  local clean="${OUT_DIR}/${name}.txt"
  echo "Downloading ${name} from ${url}"
  curl -fsSL --retry 3 --max-time 60 "$url" -o "$raw"
  echo "Cleaning -> ${clean}"
  python3 "${SCRIPT_DIR}/clean_text.py" -i "$raw" -o "$clean"
}

mkdir -p "$OUT_DIR"

if [[ "$BOOK" == "all" ]]; then
  for name in ${KNOWN_BOOKS}; do
    download_one "$name"
  done
else
  download_one "$BOOK"
fi

echo "Done. Cleaned corpus is in ${OUT_DIR}/"
