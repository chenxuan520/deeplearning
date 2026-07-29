#!/usr/bin/env python3
"""Export the HTML ebook as a single light-theme PDF using Playwright."""

from __future__ import annotations

import argparse
import html
import os
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import unquote


BOOK_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = BOOK_DIR / "build" / "deeplearning-book.pdf"
ONLINE_BASE = "https://deeplearning.011203.xyz/"
ANCHOR_PROBE_BASE = "https://pdf-anchor.invalid/"
VOID_ELEMENTS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
PARTS = (
    ("第一部分 · 打地基", range(0, 4)),
    ("第二部分 · 学习是怎么发生的", range(4, 13)),
    ("第三部分 · 经典网络结构", range(13, 16)),
    ("第四部分 · 序列与 Transformer", range(16, 19)),
    ("第五部分 · 通往大模型", range(19, 23)),
    ("第六部分 · 代码实战", range(23, 26)),
    ("第七部分 · 番外", range(26, 30)),
)


class ChapterInnerParser(HTMLParser):
    """Extract the markup inside the first .chapter__inner element."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []
        self.capturing = False
        self.finished = False
        self.depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = dict(attrs).get("class", "") or ""
        if not self.capturing and not self.finished:
            if tag == "div" and "chapter__inner" in classes.split():
                self.capturing = True
                self.depth = 1
            return
        if not self.capturing:
            return
        self.parts.append(self.get_starttag_text() or f"<{tag}>")
        if tag not in VOID_ELEMENTS:
            self.depth += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.capturing:
            self.parts.append(self.get_starttag_text() or f"<{tag} />")

    def handle_endtag(self, tag: str) -> None:
        if not self.capturing:
            return
        self.depth -= 1
        if self.depth == 0:
            self.capturing = False
            self.finished = True
            return
        self.parts.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if self.capturing:
            self.parts.append(data)

    def handle_entityref(self, name: str) -> None:
        if self.capturing:
            self.parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self.capturing:
            self.parts.append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        if self.capturing:
            self.parts.append(f"<!--{data}-->")


class FirstHeadingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_heading = False
        self.done = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "h1" and not self.done:
            self.in_heading = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "h1" and self.in_heading:
            self.in_heading = False
            self.done = True

    def handle_data(self, data: str) -> None:
        if self.in_heading:
            self.parts.append(data)


def extract_chapter(path: Path) -> tuple[str, str]:
    source = path.read_text(encoding="utf-8")
    inner_parser = ChapterInnerParser()
    inner_parser.feed(source)
    if not inner_parser.finished:
        raise ValueError(f"Cannot find a complete .chapter__inner in {path}")

    content = "".join(inner_parser.parts)
    heading_parser = FirstHeadingParser()
    heading_parser.feed(content)
    title = " ".join("".join(heading_parser.parts).split())
    if not title:
        raise ValueError(f"Cannot find h1 title in {path}")
    return title, content


def toc_html(chapters: list[tuple[int, str]]) -> str:
    groups: list[str] = []
    titles = dict(chapters)
    for part, chapter_nums in PARTS:
        items = "\n".join(
            "          <li><a href=\"{2}chapter-{0:02d}.html\"><span>第 {0} 章</span>"
            "<strong>{1}</strong></a></li>".format(num, html.escape(titles[num]), ONLINE_BASE)
            for num in chapter_nums
        )
        groups.append(
            f"""      <section class="print-toc__group">
        <h2>{html.escape(part)}</h2>
        <ol>
{items}
        </ol>
      </section>"""
        )
    return "\n".join(groups)


def preprocessing_script() -> str:
    return r"""
  <script>
    (function () {
      "use strict";

      function slugify(text, index) {
        var base = String(text || "")
          .trim()
          .toLowerCase()
          .replace(/[：:]/g, " ")
          .replace(/\s+/g, "-")
          .replace(/[^\w\u4e00-\u9fff-]/g, "");
        return base || "section-" + index;
      }

      document.querySelectorAll("[data-lab]").forEach(function (lab) {
        var figure = lab.closest(".figure");
        (figure || lab).remove();
      });
      document.querySelectorAll("img").forEach(function (image) {
        image.loading = "eager";
      });
      document.querySelectorAll("details").forEach(function (details) {
        details.open = true;
      });

      document.querySelectorAll(".print-chapter").forEach(function (section) {
        var prefix = section.getAttribute("data-source");
        var usedIds = {};

        section.querySelectorAll("h2, h3").forEach(function (heading, index) {
          if (heading.closest(".quiz")) return;
          var text = heading.textContent.replace(/\s+/g, " ").trim();
          if (!text) return;
          var id = slugify(text, index);
          while (usedIds[id]) id = id + "-" + index;
          usedIds[id] = true;
          heading.id = prefix + "--" + id;
        });

        section.querySelectorAll("[id]").forEach(function (node) {
          if (/^H[23]$/.test(node.tagName)) return;
          node.id = prefix + "--" + node.id;
        });

        section.querySelectorAll("*").forEach(function (node) {
          Array.prototype.slice.call(node.attributes).forEach(function (attr) {
            var next = attr.value.replace(/url\(\s*#([^)\s]+)\s*\)/g, "url(#" + prefix + "--$1)");
            if (next !== attr.value) node.setAttribute(attr.name, next);
          });
          ["aria-labelledby", "aria-describedby"].forEach(function (name) {
            var value = node.getAttribute(name);
            if (value) {
              node.setAttribute(name, value.split(/\s+/).map(function (id) {
                return prefix + "--" + id;
              }).join(" "));
            }
          });
          if (node.tagName !== "A") {
            ["href", "xlink:href"].forEach(function (name) {
              var value = node.getAttribute(name);
              if (value && value.charAt(0) === "#") {
                node.setAttribute(name, "#" + prefix + "--" + value.slice(1));
              }
            });
          }
        });

        section.querySelectorAll("a[href]").forEach(function (link) {
          var href = link.getAttribute("href");
          var target;
          if (href.charAt(0) === "#") {
            link.setAttribute("href", "#" + prefix + "--" + href.slice(1));
          } else if ((target = href.match(/^(chapter-\d+|glossary|about)\.html(?:#(.+))?$/))) {
            if (target[1] === prefix) {
              link.setAttribute("href", "#" + target[1] + (target[2] ? "--" + target[2] : ""));
            } else {
              link.setAttribute("href", "https://deeplearning.011203.xyz/" + href);
            }
          } else if (/^index\.html(?:#.*)?$/.test(href)) {
            link.setAttribute("href", "https://deeplearning.011203.xyz/");
          } else if (/^[^:/?#]+\.html(?:#.*)?$/.test(href)) {
            link.setAttribute("href", "https://deeplearning.011203.xyz/" + href);
          }
          link.removeAttribute("target");
        });

        section.querySelectorAll("[id]").forEach(function (node) {
          if (node.namespaceURI !== "http://www.w3.org/1999/xhtml") return;
          var probe = document.createElement("a");
          probe.className = "print-anchor-probe";
          probe.href = "https://pdf-anchor.invalid/" + encodeURIComponent(node.id);
          probe.setAttribute("aria-hidden", "true");
          probe.textContent = ".";
          node.insertBefore(probe, node.firstChild);
        });
      });

      document.documentElement.setAttribute("data-print-ready", "true");
    })();
  </script>
"""


def document_html(title: str, body: str) -> str:
    base_uri = BOOK_DIR.as_uri().rstrip("/") + "/"
    font_style = ""
    font_file = os.environ.get("DL_BOOK_PDF_FONT")
    if font_file:
        font_path = Path(font_file).expanduser().resolve()
        if not font_path.is_file():
            raise ValueError(f"PDF font not found: {font_path}")
        font_style = f"""
  <style>
    @font-face {{
      font-family: "DL Book PDF";
      src: url("{font_path.as_uri()}") format("truetype");
      font-display: block;
    }}
    :root {{ --font-sans: "DL Book PDF", sans-serif; }}
  </style>"""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="author" content="chenxuan" />
  <title>{html.escape(title)}</title>
  <base href="{html.escape(base_uri, quote=True)}" />
  <link rel="stylesheet" href="assets/book.css" />
  <link rel="stylesheet" href="assets/diagrams.css" />
  <link rel="stylesheet" href="tools/pdf-print.css" />
{font_style}
</head>
<body class="print-book">
{body}
{preprocessing_script()}
</body>
</html>
"""


def chapter_section(key: str, content: str, appendix: bool = False) -> str:
    appendix_class = " print-chapter--appendix" if appendix else ""
    return f"""  <section class="chapter print-chapter{appendix_class}" id="{key}" data-source="{key}">
    <div class="chapter__inner">
{content}
    </div>
  </section>"""


def build_print_documents() -> list[tuple[str, str, str]]:
    chapters: list[tuple[int, str, str, str]] = []
    for num in range(30):
        key = f"chapter-{num:02d}"
        title, content = extract_chapter(BOOK_DIR / f"{key}.html")
        chapters.append((num, key, title, content))

    toc_chapters = [(num, title) for num, _, title, _ in chapters]
    front_matter = f"""  <section class="print-cover" id="book-cover">
    <div class="print-cover__inner">
      <p class="print-cover__eyebrow">用 C++ 从零写懂深度学习</p>
      <h1>从神经元到大模型</h1>
      <p class="print-cover__lead">从一个神经元出发，沿着前向传播、反向传播、经典网络与 Transformer，一路理解大模型的原理、训练和工程。</p>
      <p class="print-cover__author">chenxuan</p>
      <p class="print-cover__url">deeplearning.011203.xyz</p>
    </div>
  </section>

  <section class="print-toc" id="book-toc">
    <p class="print-toc__eyebrow">从神经元到大模型</p>
    <h1>目录</h1>
{toc_html(toc_chapters)}
    <section class="print-toc__group">
      <h2>附录</h2>
      <ol>
        <li><a href="{ONLINE_BASE}glossary.html"><span>附录</span><strong>术语表:名词速查</strong></a></li>
        <li><a href="{ONLINE_BASE}about.html"><span>关于</span><strong>关于本书</strong></a></li>
      </ol>
    </section>
  </section>"""

    documents = [("front-matter", "封面与目录", document_html("从神经元到大模型", front_matter))]
    for num, key, title, content in chapters:
        bookmark = f"第 {num} 章 · {title}"
        documents.append((key, bookmark, document_html(bookmark, chapter_section(key, content))))

    for key, filename, bookmark in (
        ("glossary", "glossary.html", "附录 · 术语表"),
        ("about", "about.html", "关于本书"),
    ):
        title, content = extract_chapter(BOOK_DIR / filename)
        documents.append((key, bookmark, document_html(title, chapter_section(key, content, appendix=True))))
    return documents


def find_chrome(explicit: str | None) -> Path | None:
    configured = explicit or os.environ.get("CHROME_BIN")
    if not configured:
        return None
    path = Path(configured).expanduser()
    if path.is_file():
        return path
    raise FileNotFoundError(f"Chrome/Chromium not found: {configured}")


def included_pdf_target(uri: str) -> tuple[str, str | None] | None:
    if uri in (ONLINE_BASE, ONLINE_BASE + "index.html"):
        return "front-matter", None
    if not uri.startswith(ONLINE_BASE):
        return None
    relative = uri[len(ONLINE_BASE) :]
    path, separator, fragment = relative.partition("#")
    if not path.endswith(".html"):
        return None
    key = path[:-5]
    if key in {f"chapter-{num:02d}" for num in range(30)} | {"glossary", "about"}:
        return key, unquote(fragment) if separator else None
    return None


def normalize_browser_fragment(fragment: str) -> str:
    return "".join(
        character
        for character in fragment.lower()
        if character in "-_"
        or "a" <= character <= "z"
        or "0" <= character <= "9"
        or "\u4e00" <= character <= "\u9fff"
    )


def page_number_stream_data(page_number: int, page_width: float) -> bytes:
    label = str(page_number)
    font_size = 8.5
    text_width = len(label) * font_size * 0.556
    x = (page_width - text_width) / 2
    return (
        "Q\nq\nBT\n/FPageNum 8.5 Tf\n"
        "0.4 0.45 0.55 rg\n"
        f"1 0 0 1 {x:.2f} 20 Tm\n"
        f"({label}) Tj\nET\nQ\n"
    ).encode("ascii")


def add_page_numbers(writer: Any) -> None:
    from pypdf.generic import (  # type: ignore[import-not-found]
        ArrayObject,
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
    )

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
            NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
        }
    )
    font_ref = writer._add_object(font)

    for page_index, page in enumerate(writer.pages):
        if page_index == 0:
            continue

        resources = page["/Resources"]
        fonts = resources.get("/Font")
        if fonts is None:
            fonts = DictionaryObject()
            resources[NameObject("/Font")] = fonts
        fonts[NameObject("/FPageNum")] = font_ref

        has_contents = "/Contents" in page
        stream = DecodedStreamObject()
        footer_data = page_number_stream_data(
            page_index + 1, float(page.mediabox.width)
        )
        stream.set_data(footer_data if has_contents else footer_data[2:])
        stream_ref = writer._add_object(stream)
        if not has_contents:
            page[NameObject("/Contents")] = stream_ref
        else:
            save_state = DecodedStreamObject()
            save_state.set_data(b"q\n")
            save_state_ref = writer._add_object(save_state)
            contents = page.raw_get("/Contents")
            if isinstance(contents, ArrayObject):
                page[NameObject("/Contents")] = ArrayObject(
                    [save_state_ref, *contents, stream_ref]
                )
            else:
                page[NameObject("/Contents")] = ArrayObject(
                    [save_state_ref, contents, stream_ref]
                )


def export_pdf(chrome: Path | None, output: Path) -> None:
    try:
        from playwright.sync_api import Error as PlaywrightError  # type: ignore[import-not-found]
        from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]
        from pypdf import PdfReader, PdfWriter  # type: ignore[import-not-found]
        from pypdf.generic import (  # type: ignore[import-not-found]
            ArrayObject,
            DictionaryObject,
            FloatObject,
            NameObject,
            NullObject,
        )
    except ImportError as error:
        raise RuntimeError(
            "Missing PDF dependencies; run: python3 -m pip install -r tools/requirements-pdf.txt"
        ) from error

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    documents = build_print_documents()
    with tempfile.TemporaryDirectory(prefix="dl-book-pdf-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        rendered: list[tuple[str, str, Path]] = []
        try:
            with sync_playwright() as playwright:
                launch_args = ["--allow-file-access-from-files", "--disable-dev-shm-usage"]
                if chrome:
                    browser = playwright.chromium.launch(
                        executable_path=str(chrome), headless=True, args=launch_args
                    )
                else:
                    browser = playwright.chromium.launch(headless=True, args=launch_args)
                page = browser.new_page()
                page.emulate_media(media="print")

                for index, (key, bookmark, document) in enumerate(documents):
                    html_path = temp_dir / f"{index:02d}-{key}.html"
                    pdf_path = temp_dir / f"{index:02d}-{key}.pdf"
                    html_path.write_text(document, encoding="utf-8")
                    page.goto(html_path.as_uri(), wait_until="load")
                    page.locator('html[data-print-ready="true"]').wait_for(timeout=10_000)
                    page.evaluate("document.fonts.ready")
                    if os.environ.get("DL_BOOK_PDF_FONT") and not page.evaluate(
                        'document.fonts.check("16px DL Book PDF")'
                    ):
                        raise RuntimeError(f"Configured PDF font failed to load in {key}")
                    if page.locator("[data-lab]").count():
                        raise RuntimeError(f"Interactive lab remained in print document: {key}")
                    broken_images = page.evaluate(
                        """async () => {
                          const images = Array.from(document.images);
                          images.forEach((image) => { image.loading = "eager"; });
                          await Promise.all(images.map((image) => new Promise((resolve) => {
                            if (image.complete) return resolve();
                            image.addEventListener("load", resolve, { once: true });
                            image.addEventListener("error", resolve, { once: true });
                          })));
                          return images
                            .filter((image) => image.naturalWidth === 0)
                            .map((image) => image.currentSrc || image.src);
                        }"""
                    )
                    if broken_images:
                        raise RuntimeError(
                            f"Static image failed to load in {key}: {', '.join(broken_images)}"
                        )
                    page.pdf(
                        path=str(pdf_path),
                        print_background=True,
                        prefer_css_page_size=True,
                    )
                    rendered.append((key, bookmark, pdf_path))
                browser.close()
        except PlaywrightError as error:
            raise RuntimeError(
                "Playwright could not run Chromium; run: python3 -m playwright install chromium"
            ) from error

        writer = PdfWriter()
        page_starts: dict[str, int] = {}
        for key, bookmark, pdf_path in rendered:
            first_page = len(writer.pages)
            page_starts[key] = first_page
            writer.append(PdfReader(pdf_path), import_outline=False)
            writer.add_outline_item(bookmark, first_page)

        anchor_destinations: dict[str, tuple[int, float, float]] = {}
        for page_index, page in enumerate(writer.pages):
            annotations = page.get("/Annots", [])
            kept_annotations = ArrayObject()
            for annotation_ref in annotations:
                annotation = annotation_ref.get_object()
                action = annotation.get("/A")
                uri = str(action.get("/URI", "")) if action and action.get("/S") == "/URI" else ""
                if uri.startswith(ANCHOR_PROBE_BASE):
                    rect = annotation.get("/Rect")
                    if not rect or len(rect) != 4:
                        raise RuntimeError(f"Anchor probe has no rectangle: {uri}")
                    anchor_id = unquote(uri[len(ANCHOR_PROBE_BASE) :])
                    anchor_destinations[anchor_id] = (
                        page_index,
                        float(rect[0]),
                        float(rect[3]),
                    )
                    continue
                kept_annotations.append(annotation_ref)
            if len(kept_annotations) != len(annotations):
                if kept_annotations:
                    page[NameObject("/Annots")] = kept_annotations
                else:
                    del page[NameObject("/Annots")]

        for page in writer.pages:
            for annotation_ref in page.get("/Annots", []):
                annotation = annotation_ref.get_object()
                action = annotation.get("/A")
                if not action or action.get("/S") != "/URI":
                    continue
                target = included_pdf_target(str(action.get("/URI", "")))
                if target is None:
                    continue
                target_key, fragment = target
                if fragment:
                    anchor_id = f"{target_key}--{fragment}"
                    if anchor_id not in anchor_destinations:
                        anchor_id = f"{target_key}--{normalize_browser_fragment(fragment)}"
                    if anchor_id not in anchor_destinations:
                        raise RuntimeError(f"PDF anchor destination not found: {anchor_id}")
                    target_page, left, top = anchor_destinations[anchor_id]
                    destination = ArrayObject(
                        [
                            writer.pages[target_page].indirect_reference,
                            NameObject("/XYZ"),
                            FloatObject(left),
                            FloatObject(top),
                            NullObject(),
                        ]
                    )
                else:
                    destination = ArrayObject(
                        [writer.pages[page_starts[target_key]].indirect_reference, NameObject("/Fit")]
                    )
                annotation[NameObject("/A")] = DictionaryObject(
                    {NameObject("/S"): NameObject("/GoTo"), NameObject("/D"): destination}
                )
        writer.add_metadata(
            {
                "/Title": "从神经元到大模型",
                "/Author": "chenxuan",
                "/Subject": "用 C++ 从零写懂深度学习",
            }
        )
        add_page_numbers(writer)
        writer.compress_identical_objects(
            remove_duplicates=True, remove_unreferenced=True
        )
        with output.open("wb") as pdf_file:
            writer.write(pdf_file)

    if not output.is_file() or output.stat().st_size < 100_000:
        raise RuntimeError(f"Playwright did not create a valid PDF at {output}")
    if output.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError(f"Output is not a PDF: {output}")

    result = PdfReader(output)
    outlines = [item for item in result.outline if not isinstance(item, list)]
    if len(outlines) != len(documents):
        raise RuntimeError(f"Expected {len(documents)} PDF bookmarks, found {len(outlines)}")
    for page_index, page in enumerate(result.pages):
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        if abs(width - 595) > 2 or abs(height - 842) > 2:
            raise RuntimeError(f"Non-A4 page detected: {width:.2f} x {height:.2f} pt")

        fonts = ((page.get("/Resources") or {}).get("/Font") or {})
        if page_index == 0:
            if "/FPageNum" in fonts:
                raise RuntimeError("Cover page unexpectedly contains a page number")
        else:
            if "/FPageNum" not in fonts:
                raise RuntimeError(f"Page {page_index + 1} has no page-number font")
            page_number_font = fonts["/FPageNum"].get_object()
            if (
                page_number_font.get("/Subtype") != "/Type1"
                or page_number_font.get("/BaseFont") != "/Helvetica"
            ):
                raise RuntimeError(f"Page {page_index + 1} has an invalid page-number font")
            contents = page.raw_get("/Contents")
            expected = page_number_stream_data(page_index + 1, width)
            if isinstance(contents, ArrayObject):
                if len(contents) < 3:
                    raise RuntimeError(f"Page {page_index + 1} has invalid content streams")
                if contents[0].get_object().get_data() != b"q\n":
                    raise RuntimeError(f"Page {page_index + 1} does not isolate its original content")
                footer_data = contents[-1].get_object().get_data()
            else:
                footer_data = contents.get_object().get_data()
                expected = expected[2:]
            if footer_data != expected:
                raise RuntimeError(f"Page {page_index + 1} has an invalid page-number footer")

        for annotation_ref in page.get("/Annots", []):
            action = annotation_ref.get_object().get("/A")
            if action and action.get("/S") == "/URI":
                uri = str(action.get("/URI", ""))
                if uri.startswith(ANCHOR_PROBE_BASE):
                    raise RuntimeError(f"Anchor probe remained in PDF: {uri}")
                if included_pdf_target(uri) is not None:
                    raise RuntimeError(f"Included page still uses an external PDF link: {uri}")

    size_mib = output.stat().st_size / (1024 * 1024)
    print(f"PDF generated: {output} ({size_mib:.1f} MiB)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"PDF output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--chrome", help="Chrome/Chromium executable path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        export_pdf(find_chrome(args.chrome), args.output)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
