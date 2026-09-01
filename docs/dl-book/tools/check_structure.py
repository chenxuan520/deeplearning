#!/usr/bin/env python3
"""
check_structure.py — 电子书(docs/dl-book)结构自查

检查两件事,任一不通过即退出码 1:
    1. HTML 标签良构:<div>/<h2> 等成对配平、无错位/多余闭合(基于栈,忽略空元素)。
    2. h2 主编号连续:每章形如 "1. xxx" 的 h2 编号应是 1,2,3,… 连续
       (objectives / 小结 / 动手与思考 等无编号 h2 不计入)。

用法
    cd docs/dl-book
    python3 tools/check_structure.py

它和 check_anchors.py 是一对:一个查"链接指向"、一个查"骨架完整"。
"""

import glob
import os
import re
import sys
from html.parser import HTMLParser

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 空元素 + SVG 里常见的自足标签,不参与配平
VOID = {
    "br", "img", "meta", "link", "input", "hr", "area", "base", "col", "embed",
    "param", "source", "track", "wbr",
    "path", "line", "circle", "rect", "polyline", "polygon", "marker", "text",
    "stop", "use", "defs", "g",
}


class Balancer(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        if tag in VOID:
            return
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        elif tag in self.stack:
            self.errors.append(f"错位 </{tag}>")
            while self.stack and self.stack[-1] != tag:
                self.stack.pop()
            if self.stack:
                self.stack.pop()
        else:
            self.errors.append(f"多余 </{tag}>")


def check_file(path):
    html = open(path, encoding="utf-8").read()
    problems = []

    b = Balancer()
    b.feed(html)
    leftover = [t for t in b.stack if t not in ("html", "body", "main")]
    if b.errors:
        problems.append("标签错位: " + "; ".join(b.errors[:5]))
    if leftover:
        problems.append("未闭合残留: " + ", ".join(leftover))

    nums = [int(m) for m in re.findall(r"<h2>(\d+)\.", html)]
    if nums and nums != list(range(1, len(nums) + 1)):
        problems.append(f"h2 编号不连续: {nums}")

    return problems


def main():
    os.chdir(BASE_DIR)
    files = sorted(glob.glob("chapter-*.html")) + [
        "glossary.html", "about.html", "index.html",
        "sand-to-mythos.html", "machine-learning-history.html",
    ]
    bad = 0
    for path in files:
        if not os.path.exists(path):
            continue
        problems = check_file(path)
        if problems:
            bad += 1
            print(f"{path}:")
            for p in problems:
                print(f"    - {p}")
    if bad:
        print(f"--- {bad} 个文件有结构问题 ---")
        return 1
    print(f"结构校验通过: {len([f for f in files if os.path.exists(f)])} 个文件全部良构、编号连续")
    return 0


if __name__ == "__main__":
    sys.exit(main())
