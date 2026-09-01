#!/usr/bin/env python3
"""
check_anchors.py — 电子书(docs/dl-book)锚点失效自查

用途
    章节里大量用 <a href="chapter-xx.html#锚点"> 做跨节/跨章跳转。锚点 id 由
    assets/book.js 的 slugify() 在运行时按 h2/h3 标题自动生成(不写进 HTML)。
    改标题、加小节、调编号后,引用方很容易和新锚点对不上(尤其标题里的
    冒号 ":" 会转成连字符,手写引用常漏掉),这个脚本把全书扫一遍,列出所有
    指向不存在锚点的链接。

用法
    cd docs/dl-book
    python3 tools/check_anchors.py
    # 全部有效 -> 退出码 0;存在失效 -> 打印清单并退出码 1(可接 CI / pre-commit)

实现要点(必须和 assets/book.js 保持一致,否则会误报)
    - slugify: 小写 -> 冒号[:：]变空格 -> 空白变连字符 -> 删除其余非
      (字母数字/连字符/CJK) 字符。
    - 只给 h2/h3 生成 id;位于 .quiz 容器内的标题会被 book.js 跳过,这里同样跳过。
    - 也纳入 HTML 里显式写死的 id="..."(如 <main id="toc">),它们是静态锚点。
    - 只校验指向本地 .html 的带 # 的链接;外部链接、无 # 链接忽略。
"""

import glob
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def slugify(text):
    """复刻 assets/book.js 的 slugify()。"""
    s = text.lower()
    s = re.sub(r"[：:]", " ", s)
    s = s.strip()
    s = re.sub(r"\s+", "-", s)
    out = []
    for ch in s:
        if ch == "-" or ch.isalnum() or "\u4e00" <= ch <= "\u9fff":
            out.append(ch)
    return "".join(out)


def collect_anchor_ids(html):
    """返回该页面 book.js 会生成的全部 h2/h3 锚点 id 集合(跳过 .quiz 内标题)。"""
    ids = set()
    depth = 0            # <div> 嵌套深度
    quiz_depth = None    # 进入 quiz 容器时记下深度,退出时清空
    for m in re.finditer(r"<(/?)(div|h2|h3)([^>]*)>", html):
        closing, tag, attrs = m.group(1), m.group(2), m.group(3)
        if tag == "div":
            if closing:
                if quiz_depth is not None and depth == quiz_depth:
                    quiz_depth = None
                depth -= 1
            else:
                depth += 1
                if quiz_depth is None and "quiz" in attrs:
                    quiz_depth = depth
        elif tag in ("h2", "h3") and not closing:
            if quiz_depth is not None:
                continue
            end = html.find("</" + tag + ">", m.end())
            text = re.sub(r"<[^>]+>", "", html[m.end():end])
            text = text.replace("&amp;", "&")
            ids.add(slugify(text))
    # 再纳入 HTML 里显式写死的 id="...",如 index.html 的 <main id="toc">、
    # glossary 的 <li id="g-...">。这些是静态锚点,不经 slugify。
    for m in re.finditer(r'\bid="([^"]+)"', html):
        ids.add(m.group(1))
    return ids


def main():
    os.chdir(BASE_DIR)
    files = sorted(glob.glob("chapter-*.html")) + [
        "glossary.html", "about.html", "index.html",
        "sand-to-mythos.html", "machine-learning-history.html",
    ]
    file_anchors = {f: collect_anchor_ids(open(f, encoding="utf-8").read())
                    for f in files if os.path.exists(f)}

    total = 0
    bad = []
    for path in files:
        if not os.path.exists(path):
            continue
        html = open(path, encoding="utf-8").read()
        for m in re.finditer(r'href="([^"]*#[^"]+)"', html):
            href = m.group(1)
            total += 1
            target, anchor = href.split("#", 1)
            tf = target if target else path
            if not tf.endswith(".html"):
                continue
            if tf not in file_anchors:
                if not os.path.exists(tf):
                    bad.append((path, href, "file-missing"))
                continue
            if anchor not in file_anchors[tf]:
                bad.append((path, href, "anchor-missing"))

    print(f"总锚点引用: {total}")
    print(f"失效: {len(bad)}")
    for src, href, why in bad:
        print(f"  {src} -> {href} [{why}]")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
