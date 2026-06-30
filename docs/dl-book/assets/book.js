/*
 * book.js — 《深度学习入门》Web 书的导航与阅读体验
 * 职责: 注入顶部 header、阅读进度条、侧边目录抽屉、上一章/下一章导航,
 *       处理键盘翻页、记忆"最后阅读章节"、滚动淡入。
 * 纯静态零依赖。章节页用 <body data-chapter="N"> 标识自己是第 N 章;
 * 封面页用 <body data-cover>。
 */
(function () {
  "use strict";

  // 全书章节清单 (顺序即阅读顺序, 索引 == data-chapter 的值)
  var CHAPTERS = [
    { num: "0", title: "导论:这本书到底在讲什么", file: "chapter-00.html", part: "第一部分 · 打地基" },
    { num: "1", title: "一个神经元", file: "chapter-01.html", part: "第一部分 · 打地基" },
    { num: "2", title: "搭成网络:前向传播", file: "chapter-02.html", part: "第一部分 · 打地基" },
    { num: "3", title: "怎么衡量“错”:损失函数", file: "chapter-03.html", part: "第二部分 · 学习是怎么发生的" },
    { num: "4", title: "怎么变“好”:梯度下降", file: "chapter-04.html", part: "第二部分 · 学习是怎么发生的" },
    { num: "5", title: "反向传播", file: "chapter-05.html", part: "第二部分 · 学习是怎么发生的" },
    { num: "6", title: "让它真的训得动", file: "chapter-06.html", part: "第二部分 · 学习是怎么发生的" },
    { num: "7", title: "为什么需要注意力", file: "chapter-07.html", part: "第三部分 · 序列与 Transformer" },
    { num: "8", title: "注意力机制", file: "chapter-08.html", part: "第三部分 · 序列与 Transformer" },
    { num: "9", title: "Transformer 的完整结构", file: "chapter-09.html", part: "第三部分 · 序列与 Transformer" },
    { num: "10", title: "字符级语言模型", file: "chapter-10.html", part: "第四部分 · 通往大模型" },
    { num: "11", title: "通往大模型", file: "chapter-11.html", part: "第四部分 · 通往大模型" }
  ];

  var STORAGE_LAST = "dlbook:last";

  var store = {
    get: function (k) {
      try { return window.localStorage.getItem(k); } catch (e) { return null; }
    },
    set: function (k, v) {
      try { window.localStorage.setItem(k, v); } catch (e) { /* file:// 下可能受限, 忽略 */ }
    }
  };

  function el(tag, cls, html) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (html != null) node.innerHTML = html;
    return node;
  }

  function chapterIndex() {
    var v = document.body.getAttribute("data-chapter");
    if (v == null || v === "") return -1;
    var n = Number(v);
    return isNaN(n) ? -1 : n;
  }

  // ---------- 目录抽屉 ----------
  function buildToc(currentIdx) {
    var backdrop = el("div", "book-toc-backdrop");
    var aside = el("aside", "book-toc");
    aside.setAttribute("aria-label", "目录");

    var title = el("div", "book-toc__title", "目录 · Contents");
    aside.appendChild(title);

    var list = el("ol", "book-toc__list");
    var lastPart = null;
    CHAPTERS.forEach(function (ch, i) {
      if (ch.part !== lastPart) {
        lastPart = ch.part;
        var partLi = el("li");
        partLi.appendChild(el("div", "book-toc__title", ch.part));
        partLi.firstChild.style.marginTop = i === 0 ? "0" : "1rem";
        list.appendChild(partLi);
      }
      var li = el("li");
      var a = el("a", "book-toc__link" + (i === currentIdx ? " is-current" : ""));
      a.href = ch.file;
      a.appendChild(el("span", "book-toc__num", ch.num));
      a.appendChild(el("span", "book-toc__name", ch.title));
      li.appendChild(a);
      list.appendChild(li);
    });
    aside.appendChild(list);

    function close() {
      aside.classList.remove("is-open");
      backdrop.classList.remove("is-open");
    }
    function open() {
      aside.classList.add("is-open");
      backdrop.classList.add("is-open");
    }
    backdrop.addEventListener("click", close);
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") close();
    });

    document.body.appendChild(backdrop);
    document.body.appendChild(aside);
    return { open: open, close: close };
  }

  // ---------- 顶部 header + 进度条 ----------
  function buildHeader(currentIdx, toc) {
    var ch = CHAPTERS[currentIdx];

    var progress = el("div", "book-progress");
    progress.id = "bookProgress";

    var header = el("header", "book-header");

    var menuBtn = el("button", "book-header__menu", "&#9776;");
    menuBtn.type = "button";
    menuBtn.setAttribute("aria-label", "打开目录");
    menuBtn.addEventListener("click", toc.open);

    var home = el("a", "book-header__home", "深度学习入门");
    home.href = "index.html";

    var current = el("span", "book-header__current", "第 " + ch.num + " 章 · " + ch.title);
    var pct = el("span", "book-header__pct");
    pct.id = "bookPct";
    pct.textContent = "0%";

    header.appendChild(menuBtn);
    header.appendChild(home);
    header.appendChild(current);
    header.appendChild(pct);

    document.body.appendChild(progress);
    document.body.appendChild(header);
    return { progress: progress, pct: pct };
  }

  // ---------- 底部上一章 / 下一章 ----------
  function buildPrevNext(currentIdx) {
    var nav = el("nav", "chapter-nav");
    nav.setAttribute("aria-label", "章节导航");
    var prev = CHAPTERS[currentIdx - 1];
    var next = CHAPTERS[currentIdx + 1];

    if (prev) {
      var a1 = el("a", "chapter-nav__link chapter-nav__link--prev");
      a1.href = prev.file;
      a1.appendChild(el("span", "chapter-nav__dir", "← 上一章"));
      a1.appendChild(el("span", "chapter-nav__title", "第 " + prev.num + " 章 · " + prev.title));
      nav.appendChild(a1);
    } else {
      var p1 = el("a", "chapter-nav__link chapter-nav__link--prev");
      p1.href = "index.html";
      p1.appendChild(el("span", "chapter-nav__dir", "← 返回"));
      p1.appendChild(el("span", "chapter-nav__title", "全书目录"));
      nav.appendChild(p1);
    }

    if (next) {
      var a2 = el("a", "chapter-nav__link chapter-nav__link--next");
      a2.href = next.file;
      a2.appendChild(el("span", "chapter-nav__dir", "下一章 →"));
      a2.appendChild(el("span", "chapter-nav__title", "第 " + next.num + " 章 · " + next.title));
      nav.appendChild(a2);
    } else {
      var p2 = el("a", "chapter-nav__link chapter-nav__link--next");
      p2.href = "index.html";
      p2.appendChild(el("span", "chapter-nav__dir", "读完啦 →"));
      p2.appendChild(el("span", "chapter-nav__title", "回到目录 / 重读"));
      nav.appendChild(p2);
    }

    var host = document.querySelector(".chapter") || document.body;
    host.appendChild(nav);
  }

  // ---------- 阅读进度 ----------
  function setupProgress(refs) {
    var ticking = false;
    function update() {
      ticking = false;
      var doc = document.documentElement;
      var max = doc.scrollHeight - doc.clientHeight;
      var ratio = max > 0 ? window.scrollY / max : 0;
      if (ratio < 0) ratio = 0;
      if (ratio > 1) ratio = 1;
      refs.progress.style.width = (ratio * 100).toFixed(1) + "%";
      refs.pct.textContent = Math.round(ratio * 100) + "%";
    }
    function onScroll() {
      if (!ticking) {
        ticking = true;
        window.requestAnimationFrame(update);
      }
    }
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    update();
  }

  // ---------- 键盘左右翻页 ----------
  function setupKeyboard(currentIdx) {
    document.addEventListener("keydown", function (e) {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      var t = e.target;
      if (t && /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName)) return;
      if (e.key === "ArrowRight") {
        var next = CHAPTERS[currentIdx + 1];
        if (next) window.location.href = next.file;
      } else if (e.key === "ArrowLeft") {
        var prev = CHAPTERS[currentIdx - 1];
        if (prev) window.location.href = prev.file;
      }
    });
  }

  // ---------- 滚动淡入 ----------
  function setupReveal() {
    var nodes = [].slice.call(document.querySelectorAll(".reveal"));
    if (!nodes.length) return;
    if (!("IntersectionObserver" in window)) {
      nodes.forEach(function (n) { n.classList.add("is-visible"); });
      return;
    }
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          obs.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12 });
    nodes.forEach(function (n) { obs.observe(n); });
  }

  // ---------- 封面: 继续阅读 ----------
  function setupCover() {
    setupReveal();
    var last = store.get(STORAGE_LAST);
    var btn = document.querySelector("[data-continue]");
    if (!btn) return;
    var found = null;
    for (var i = 0; i < CHAPTERS.length; i++) {
      if (CHAPTERS[i].file === last) { found = CHAPTERS[i]; break; }
    }
    if (found) {
      btn.href = found.file;
      btn.textContent = "继续阅读 · 第 " + found.num + " 章";
      btn.removeAttribute("hidden");
    } else {
      btn.href = CHAPTERS[0].file;
      btn.textContent = "从头开始读 →";
      btn.removeAttribute("hidden");
    }
  }

  function init() {
    if (document.body.hasAttribute("data-cover")) {
      setupCover();
      return;
    }
    var idx = chapterIndex();
    if (idx < 0 || idx >= CHAPTERS.length) {
      setupReveal();
      return;
    }
    var toc = buildToc(idx);
    var refs = buildHeader(idx, toc);
    buildPrevNext(idx);
    setupProgress(refs);
    setupKeyboard(idx);
    setupReveal();
    store.set(STORAGE_LAST, CHAPTERS[idx].file);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
