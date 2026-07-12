#!/usr/bin/env python3
"""Clean raw text into a corpus suitable for the mini_lm character model.

This is a tiny, dependency-free helper (Python 3 standard library only) used by
`docs/mini-lm-demo.md`. It turns messy downloaded text (e.g. a Project Gutenberg
book) into a plain-ASCII corpus so the character vocabulary stays small and the
model spends its capacity on language structure instead of rare glyphs.

What it does:
  1. (optional) strip Project Gutenberg header/footer markers
  2. normalize common Unicode punctuation (curly quotes, dashes, ellipsis) to ASCII
  3. optionally drop every remaining non-ASCII byte
  4. collapse 3+ blank lines into one and trim surrounding whitespace

Usage:
  python3 clean_text.py --input raw.txt --output corpus.txt
  python3 clean_text.py -i raw.txt -o corpus.txt --keep-gutenberg-markers
  cat raw.txt | python3 clean_text.py > corpus.txt
"""

import argparse
import re
import sys

# Map common Unicode punctuation to ASCII equivalents so the char vocab stays
# small. Extend this table if your source text uses other special glyphs.
UNICODE_TO_ASCII = {
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote / apostrophe
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\u2014": "-",   # em dash
    "\u2013": "-",   # en dash
    "\u2026": "...",  # ellipsis
    "\u00a0": " ",   # non-breaking space
    "\ufeff": "",    # BOM
}

GUTENBERG_START = "*** START OF THE PROJECT GUTENBERG EBOOK"
GUTENBERG_END = "*** END OF THE PROJECT GUTENBERG EBOOK"


def strip_gutenberg(text):
    """Return only the body between the Gutenberg start/end markers, if present."""
    start = text.find(GUTENBERG_START)
    if start != -1:
        newline = text.find("\n", start)
        text = text[newline + 1:] if newline != -1 else text[start:]
    end = text.find(GUTENBERG_END)
    if end != -1:
        text = text[:end]
    return text


def clean(text, keep_gutenberg_markers=False, keep_non_ascii=False):
    if not keep_gutenberg_markers:
        text = strip_gutenberg(text)
    for src, dst in UNICODE_TO_ASCII.items():
        text = text.replace(src, dst)
    if not keep_non_ascii:
        text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", help="input file (default: stdin)")
    parser.add_argument("-o", "--output", help="output file (default: stdout)")
    parser.add_argument("--keep-gutenberg-markers", action="store_true",
                        help="do not strip Project Gutenberg header/footer")
    parser.add_argument("--keep-non-ascii", action="store_true",
                        help="keep non-ASCII bytes instead of dropping them")
    args = parser.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
    else:
        raw = sys.stdin.read()

    result = clean(raw, args.keep_gutenberg_markers, args.keep_non_ascii)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        distinct = len(set(result))
        sys.stderr.write(
            "wrote {} chars ({} distinct) to {}\n".format(
                len(result), distinct, args.output))
    else:
        sys.stdout.write(result)


if __name__ == "__main__":
    main()
