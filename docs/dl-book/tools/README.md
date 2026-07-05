# 电子书维护脚本 (`docs/dl-book/tools/`)

改章节 HTML 后，可在 **`docs/dl-book/` 目录下**运行这些脚本（仅依赖 Python 3 标准库）。

```bash
cd docs/dl-book
```

## 脚本一览

| 脚本 | 作用 | CI |
|------|------|-----|
| `check_anchors.py` | 跨章/跨节 `#锚点` 是否指向存在的 h2/h3 | ✅ [dl-book-check.yml](../../../.github/workflows/dl-book-check.yml) |
| `check_structure.py` | HTML 标签配平 + 每章 h2 主编号是否连续 | ✅ 同上 |
| `count_chars.py` | 统计全书汉字与篇幅（本地自查用） | — |

---

## `check_anchors.py`

章节里大量 `<a href="chapter-xx.html#锚点">`；锚点 id 由 `assets/book.js` 的 `slugify()` **运行时**按 h2/h3 标题生成，改标题后引用容易失效。

```bash
python3 tools/check_anchors.py
```

- 全部有效 → 退出码 `0`
- 存在失效链接 → 打印清单，退出码 `1`

**注意：** 脚本里的 slugify 规则必须与 `assets/book.js` 一致，否则会误报。

---

## `check_structure.py`

```bash
python3 tools/check_structure.py
```

检查：

1. HTML 标签是否成对闭合（基于栈，忽略空元素 / 常见 SVG 标签）
2. 每章形如 `1. xxx` 的 **h2 主编号**是否 1、2、3… 连续（「小结」「动手与思考」等无编号 h2 不计）

任一不通过 → 退出码 `1`。

---

## `count_chars.py`

去掉 HTML 标签、`script`、`style`、注释后统计正文规模：

```bash
python3 tools/count_chars.py              # 汇总
python3 tools/count_chars.py -v           # 按文件明细
python3 tools/count_chars.py --pages 400  # 按 400 汉字/页估页数
python3 tools/count_chars.py --json       # JSON 输出
```

默认扫描 `docs/dl-book/*.html`（各章 + `index.html` + `glossary.html`）。

---

## 改书后建议顺序

1. 本地改 HTML
2. `python3 tools/check_structure.py`
3. `python3 tools/check_anchors.py`
4. （可选）`python3 tools/count_chars.py -v` 看篇幅变化
5. 推送到 `github` 远程以更新 GitHub Pages（见仓库根 `README.md`）
