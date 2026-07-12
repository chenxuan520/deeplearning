#!/usr/bin/env python3
"""
count_chars.py — 统计电子书正文规模 (docs/dl-book)

从 HTML 去掉标签 / script / style / 注释后, 统计:
  - 汉字 (CJK 统一汉字)
  - 去空白后的可见字符 (含英文、数字、标点)
  - 含空格的字符数

用法:
    cd docs/dl-book
    python3 tools/count_chars.py              # 汇总
    python3 tools/count_chars.py -v           # 按章节明细 (每章标题 + 字数)
    python3 tools/count_chars.py --pages 400  # 自定义「每页汉字数」估页数

默认扫描 BASE_DIR 下所有 *.html (含 index / glossary / 各章)。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RE_SCRIPT = re.compile(r"<script[\s\S]*?</script>", re.IGNORECASE)
RE_STYLE = re.compile(r"<style[\s\S]*?</style>", re.IGNORECASE)
RE_COMMENT = re.compile(r"<!--[\s\S]*?-->")
RE_TAG = re.compile(r"<[^>]+>")
RE_HAN = re.compile(r"[\u4e00-\u9fff]")
RE_WS = re.compile(r"\s+")
RE_TITLE = re.compile(r"<title[^>]*>([\s\S]*?)</title>", re.IGNORECASE)
RE_H1 = re.compile(r"<h1[^>]*>([\s\S]*?)</h1>", re.IGNORECASE)


def strip_html(raw: str) -> str:
    text = RE_SCRIPT.sub(" ", raw)
    text = RE_STYLE.sub(" ", text)
    text = RE_COMMENT.sub(" ", text)
    text = RE_TAG.sub(" ", text)
    return RE_WS.sub(" ", text).strip()


def extract_title(raw: str, fallback: str) -> str:
    """取章节名: 优先 <title> 竖线前那段, 退回 <h1>, 再退回文件名。"""
    m = RE_TITLE.search(raw)
    if m:
        title = RE_TAG.sub("", m.group(1)).split("|")[0].strip()
        if title:
            return title
    m = RE_H1.search(raw)
    if m:
        title = RE_WS.sub(" ", RE_TAG.sub(" ", m.group(1))).strip()
        if title:
            return title
    return fallback


def count_text(text: str) -> dict[str, int]:
    han = len(RE_HAN.findall(text))
    with_space = len(text)
    no_space = len(RE_WS.sub("", text))
    return {"han": han, "no_space": no_space, "with_space": with_space}


def disp_width(text: str) -> int:
    """终端显示宽度: 全角 / 宽字符按 2 计, 其余按 1。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def ljust_disp(text: str, width: int) -> str:
    """按显示宽度左对齐到 width; 超长则截断加省略号。"""
    if disp_width(text) <= width:
        return text + " " * (width - disp_width(text))
    out, used = "", 0
    for ch in text:
        cw = 2 if unicodedata.east_asian_width(ch) in "WF" else 1
        if used + cw > width - 1:
            break
        out += ch
        used += cw
    out += "…"
    return out + " " * max(0, width - disp_width(out))


def rjust_disp(text: str, width: int) -> str:
    """按显示宽度右对齐, 使中文表头能和数字列对齐。"""
    return " " * max(0, width - disp_width(text)) + text


def iter_html_files(base: Path) -> list[Path]:
    return sorted(base.glob("*.html"))


def main() -> int:
    parser = argparse.ArgumentParser(description="统计 dl-book 电子书字数")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="按章节输出明细 (每章标题 + 字数)",
    )
    parser.add_argument(
        "--pages",
        type=float,
        default=350.0,
        help="估算页数时采用的每页汉字数 (默认 350)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 输出 (便于 CI / 脚本读取)",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=BASE_DIR,
        help=f"HTML 根目录 (默认 {BASE_DIR})",
    )
    args = parser.parse_args()

    files = iter_html_files(args.dir)
    if not files:
        print(f"未找到 HTML: {args.dir}/*.html", file=sys.stderr)
        return 1

    rows: list[dict] = []
    total = {"han": 0, "no_space": 0, "with_space": 0}

    for path in files:
        raw = path.read_text(encoding="utf-8")
        title = extract_title(raw, path.stem)
        stats = count_text(strip_html(raw))
        row = {"file": path.name, "title": title, **stats}
        rows.append(row)
        for k in total:
            total[k] += stats[k]

    pages = total["han"] / args.pages if args.pages > 0 else 0.0

    if args.json:
        payload = {
            "dir": str(args.dir),
            "files": len(rows),
            "total": total,
            "pages_estimate": round(pages, 1),
            "pages_per_han": args.pages,
            "by_file": rows if args.verbose else None,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.verbose:
        print(f"目录: {args.dir}")
        print(f"文件: {len(rows)} 个 HTML\n")
        header = (
            f"{ljust_disp('章节', 34)}{ljust_disp('文件', 20)}"
            f"{rjust_disp('汉字', 9)}{rjust_disp('去空白', 11)}"
            f"{rjust_disp('含空白', 11)}"
        )
        print(header)
        print("-" * disp_width(header))
        for row in rows:
            print(
                f"{ljust_disp(row['title'], 34)}{ljust_disp(row['file'], 20)}"
                f"{row['han']:>9,d}{row['no_space']:>11,d}{row['with_space']:>11,d}"
            )
        print("-" * disp_width(header))

    print(f"汉字总数:       {total['han']:,}")
    print(f"去空白字符:     {total['no_space']:,}")
    print(f"含空白字符:     {total['with_space']:,}")
    print(f"约 {args.pages:g} 字/页 → {pages:.0f} 页")
    return 0


if __name__ == "__main__":
    sys.exit(main())
