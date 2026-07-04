/*
 * book.js — 《从神经元到大模型》Web 书的导航与阅读体验
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
    { num: "1", title: "数学预备:看懂公式的一点点数学", file: "chapter-01.html", part: "第一部分 · 打地基" },
    { num: "2", title: "一个神经元", file: "chapter-02.html", part: "第一部分 · 打地基", core: true },
    { num: "3", title: "搭成网络:前向传播", file: "chapter-03.html", part: "第一部分 · 打地基" },
    { num: "4", title: "怎么衡量“错”:损失函数", file: "chapter-04.html", part: "第二部分 · 学习是怎么发生的" },
    { num: "5", title: "怎么变“好”:梯度下降", file: "chapter-05.html", part: "第二部分 · 学习是怎么发生的" },
    { num: "6", title: "反向传播", file: "chapter-06.html", part: "第二部分 · 学习是怎么发生的", core: true },
    { num: "7", title: "激活函数全家福", file: "chapter-07.html", part: "第二部分 · 学习是怎么发生的" },
    { num: "8", title: "优化器:从 SGD 到 Adam", file: "chapter-08.html", part: "第二部分 · 学习是怎么发生的" },
    { num: "9", title: "让它真的训得动", file: "chapter-09.html", part: "第二部分 · 学习是怎么发生的" },
    { num: "10", title: "正则化与泛化", file: "chapter-10.html", part: "第二部分 · 学习是怎么发生的" },
    { num: "11", title: "评估与数据", file: "chapter-11.html", part: "第二部分 · 学习是怎么发生的" },
    { num: "12", title: "强化学习:试错里学策略", file: "chapter-12.html", part: "第二部分 · 学习是怎么发生的" },
    { num: "13", title: "CNN 卷积神经网络", file: "chapter-13.html", part: "第三部分 · 经典网络结构" },
    { num: "14", title: "RNN 与 LSTM", file: "chapter-14.html", part: "第三部分 · 经典网络结构" },
    { num: "15", title: "词嵌入与 word2vec", file: "chapter-15.html", part: "第三部分 · 经典网络结构" },
    { num: "16", title: "为什么需要注意力", file: "chapter-16.html", part: "第四部分 · 序列与 Transformer" },
    { num: "17", title: "注意力机制", file: "chapter-17.html", part: "第四部分 · 序列与 Transformer", core: true },
    { num: "18", title: "Transformer 的完整结构", file: "chapter-18.html", part: "第四部分 · 序列与 Transformer", core: true },
    { num: "19", title: "字符级语言模型", file: "chapter-19.html", part: "第五部分 · 通往大模型" },
    { num: "20", title: "通往大模型:原理与训练", file: "chapter-20.html", part: "第五部分 · 通往大模型", core: true },
    { num: "21", title: "大模型的工程与基础设施", file: "chapter-21.html", part: "第五部分 · 通往大模型" },
    { num: "22", title: "用好大模型:提示、RAG 与 Agent", file: "chapter-22.html", part: "第五部分 · 通往大模型" },
    { num: "23", title: "MNIST 实战:第一、二部分", file: "chapter-23.html", part: "第六部分 · 代码实战" },
    { num: "24", title: "mini-LM 实战:第四、五部分", file: "chapter-24.html", part: "第六部分 · 代码实战" },
    { num: "25", title: "井字棋 Q-learning 实战:强化学习", file: "chapter-25.html", part: "第六部分 · 代码实战" },
    { num: "26", title: "无监督与自监督学习", file: "chapter-26.html", part: "第七部分 · 番外" },
    { num: "27", title: "机器学习全景图", file: "chapter-27.html", part: "第七部分 · 番外" }
  ];

  var STORAGE_LAST = "dlbook:last";
  var STORAGE_SEARCH = "dlbook:search";
  var REPO_URL = "https://github.com/chenxuan520/deeplearning";

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

  function slugify(text, index) {
    var base = String(text || "")
      .trim()
      .toLowerCase()
      .replace(/[：:]/g, " ")
      .replace(/\s+/g, "-")
      .replace(/[^\w\u4e00-\u9fff-]/g, "");
    return base || "section-" + index;
  }

  function buildChapterRail(currentIdx) {
    var aside = el("aside", "book-rail book-rail--chapters");
    aside.setAttribute("aria-label", "章节导航");
    aside.appendChild(el("div", "book-rail__title", "章节"));

    var list = el("ol", "book-rail__list");
    var lastPart = null;
    CHAPTERS.forEach(function (ch, i) {
      if (ch.part !== lastPart) {
        lastPart = ch.part;
        var part = el("li", "book-rail__part", ch.part);
        list.appendChild(part);
      }
      var li = el("li");
      var a = el("a", "book-rail__link" + (i === currentIdx ? " is-current" : ""));
      a.href = ch.file;
      a.appendChild(el("span", "book-rail__num", ch.num));
      a.appendChild(el("span", "book-rail__name", ch.title));
      if (ch.core) {
        var star = el("span", "book-rail__core", "★");
        star.title = "核心章节";
        a.appendChild(star);
      }
      if (i === currentIdx) aside._currentLink = a;
      li.appendChild(a);
      list.appendChild(li);
    });
    var isGlossary = /glossary\.html$/.test(location.pathname);
    list.appendChild(el("li", "book-rail__part", "附录"));
    var gli = el("li");
    var ga = el("a", "book-rail__link" + (isGlossary ? " is-current" : ""));
    ga.href = "glossary.html";
    ga.appendChild(el("span", "book-rail__num", "★"));
    ga.appendChild(el("span", "book-rail__name", "术语表"));
    gli.appendChild(ga);
    list.appendChild(gli);
    aside.appendChild(list);
    return aside;
  }

  // 把侧栏里的"当前章"滚动到可视区中间, 省得每次切章都要手动找。
  function scrollRailToCurrent(scroller, link) {
    if (!scroller || !link) return;
    // 元素相对滚动容器的偏移 - 半个容器高 + 半个元素高 = 居中
    var target = link.offsetTop - scroller.clientHeight / 2 + link.offsetHeight / 2;
    scroller.scrollTop = Math.max(0, target);
  }

  // 给 inner 里的 h2/h3 赋 id 并返回条目列表。
  // 搜索索引与本章大纲都用它, 保证同一个标题算出的锚点 id 完全一致。
  function assignHeadingIds(inner) {
    var headings = [].slice.call(inner.querySelectorAll("h2, h3"));
    var usedIds = {};
    var entries = [];
    headings.forEach(function (heading, index) {
      if (heading.closest(".quiz")) return;
      var text = heading.textContent.replace(/\s+/g, " ").trim();
      if (!text) return;
      var id = slugify(text, index);
      while (usedIds[id]) id = id + "-" + index;
      usedIds[id] = true;
      heading.id = id;
      entries.push({ heading: heading, id: id, text: text });
    });
    return entries;
  }

  function buildOutlineRail(inner) {
    var aside = el("aside", "book-rail book-rail--outline");
    aside.setAttribute("aria-label", "本章目录");
    aside.appendChild(el("div", "book-rail__title", "本章"));

    var list = el("ol", "book-rail__list book-rail__list--outline");
    var links = [];

    assignHeadingIds(inner).forEach(function (entry) {
      var heading = entry.heading;
      var li = el("li", heading.tagName === "H3" ? "book-rail__outline-item book-rail__outline-item--sub" : "book-rail__outline-item");
      var a = el("a", "book-rail__outline-link");
      a.href = "#" + entry.id;
      a.textContent = entry.text;
      li.appendChild(a);
      list.appendChild(li);
      links.push({ link: a, heading: heading });
    });

    if (!links.length) {
      aside.appendChild(el("p", "book-rail__empty", "本章暂无小节标题"));
      return { aside: aside, links: [] };
    }

    aside.appendChild(list);
    return { aside: aside, links: links };
  }

  // 小标题跳转后停靠位置: 视口高度的这个比例处 (偏上一点, 标题上方留少量空白,
  // 而不是停在正正中间——读者视线不用往上找标题)
  var HEADING_ANCHOR_RATIO = 0.28;

  // 把某个元素滚到视口偏上位置 (smooth 可选)
  function scrollHeadingIntoView(target, smooth) {
    var rect = target.getBoundingClientRect();
    var y = rect.top + window.pageYOffset - (window.innerHeight * HEADING_ANCHOR_RATIO);
    y = Math.max(0, y);
    if (smooth) window.scrollTo({ top: y, behavior: "smooth" });
    else window.scrollTo(0, y);
  }

  function setupOutlineSpy(links) {
    if (!links.length) return;
    var current = null;
    var lockUntil = 0;
    function setCurrent(link) {
      if (current === link) return;
      if (current) current.classList.remove("is-current");
      current = link;
      if (current) current.classList.add("is-current");
    }
    // 点击右侧目录: 把该小标题滚到视口偏上位置, 并立即高亮它。
    // 这样即使小节内容很少, 也不会误定位/误高亮到它下面的标题。
    links.forEach(function (item) {
      item.link.addEventListener("click", function (e) {
        e.preventDefault();
        lockUntil = Date.now() + 900; // 平滑滚动期间, 先别让 spy 抢高亮
        setCurrent(item.link);
        scrollHeadingIntoView(item.heading, true);
        if (window.history && window.history.replaceState) {
          window.history.replaceState(null, "", "#" + item.heading.id);
        }
      });
    });
    if ("IntersectionObserver" in window) {
      // 判定线设在视口偏上处 (约 28% 高度), 与"点击后标题停靠的位置"对齐:
      // 谁跨过这条线就高亮谁, 避免定位点和高亮项对不上。
      var obs = new IntersectionObserver(function (entries) {
        if (Date.now() < lockUntil) return;
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          links.forEach(function (item) {
            if (item.heading === entry.target) setCurrent(item.link);
          });
        });
      }, { rootMargin: "-28% 0px -72% 0px", threshold: 0 });
      links.forEach(function (item) { obs.observe(item.heading); });
    }
    setCurrent(links[0].link);
  }

  function buildDesktopLayout(currentIdx) {
    var chapter = document.querySelector(".chapter");
    var inner = chapter && chapter.querySelector(".chapter__inner");
    if (!chapter || !inner) return null;

    var layout = el("div", "book-layout");
    var main = el("div", "book-layout__main");
    var chapterRail = buildChapterRail(currentIdx);
    var outline = buildOutlineRail(inner);

    layout.appendChild(chapterRail);
    layout.appendChild(main);
    layout.appendChild(outline.aside);
    chapter.insertBefore(layout, inner);
    main.appendChild(inner);
    setupOutlineSpy(outline.links);
    // 进 DOM 后再滚: 此时 .book-rail 才有真实高度
    requestAnimationFrame(function () {
      scrollRailToCurrent(chapterRail, chapterRail._currentLink);
    });
    return main;
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
      if (ch.core) {
        var tstar = el("span", "book-toc__core", "★");
        tstar.title = "核心章节";
        a.appendChild(tstar);
      }
      if (i === currentIdx) aside._currentLink = a;
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
      // 打开抽屉时把当前章滚到中间, 长目录不用手动翻
      scrollRailToCurrent(aside, aside._currentLink);
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
  function buildHeader(currentIdx, toc, search) {
    var ch = CHAPTERS[currentIdx];
    var currentLabel = ch
      ? "第 " + ch.num + " 章 · " + ch.title
      : ((document.querySelector(".chapter__inner h1") || {}).textContent || "从神经元到大模型");

    var progress = el("div", "book-progress");
    progress.id = "bookProgress";

    var header = el("header", "book-header");

    var menuBtn = el("button", "book-header__menu", "&#9776;");
    menuBtn.type = "button";
    menuBtn.setAttribute("aria-label", "打开目录");
    menuBtn.addEventListener("click", toc.open);

    var home = el("a", "book-header__home");
    home.href = REPO_URL;
    home.target = "_blank";
    home.rel = "noopener noreferrer";
    home.setAttribute("aria-label", "在 GitHub 上查看源码");
    home.innerHTML =
      '<span class="book-header__home-icon">' + GITHUB_ICON + "</span>" +
      '<span class="book-header__home-text">从神经元到大模型</span>';

    var current = el("span", "book-header__current", currentLabel);
    var pct = el("span", "book-header__pct");
    pct.id = "bookPct";
    pct.textContent = "0%";

    header.appendChild(menuBtn);
    header.appendChild(home);
    header.appendChild(current);
    if (search) header.appendChild(makeSearchTrigger(search, "book-header__search", "搜索全书", SEARCH_ICON));
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

    var host = document.querySelector(".book-layout__main") || document.querySelector(".chapter") || document.body;
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

  // ---------- 全书搜索 ----------
  function escapeHtml(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }
  function escapeRegExp(s) { return String(s).replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

  function isMacPlatform() {
    return /Mac|iPhone|iPad|iPod/.test(navigator.platform || navigator.userAgent || "");
  }

  function searchShortcutHint() {
    return isMacPlatform() ? "⌘K 或 /" : "Ctrl+K 或 /";
  }

  function persistSearchQuery(query) {
    var q = (query || "").trim();
    if (q) store.set(STORAGE_SEARCH, q);
    else {
      try { window.localStorage.removeItem(STORAGE_SEARCH); } catch (e) { /* ignore */ }
    }
  }

  function loadSearchQuery() {
    return store.get(STORAGE_SEARCH) || "";
  }

  var SEARCH_ICON =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7"></circle><line x1="16.5" y1="16.5" x2="21" y2="21"></line></svg>';

  var GITHUB_ICON =
    '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>';

  // 把一章的 .chapter__inner 拆成若干"以小标题分段"的可搜索片段
  function extractSections(inner, ch, chIdx) {
    assignHeadingIds(inner).forEach(function (entry) {
      entry.heading.setAttribute("data-search-hid", entry.id);
    });
    var sections = [];
    var current = { chIdx: chIdx, num: ch.num, chTitle: ch.title, file: ch.file, heading: ch.title, headingId: "", text: "" };
    function pushCurrent() {
      var t = current.text.replace(/\s+/g, " ").trim();
      if (t) { current.text = t; sections.push(current); }
    }
    function walk(node) {
      var kids = node.childNodes;
      for (var i = 0; i < kids.length; i++) {
        var c = kids[i];
        if (c.nodeType === 1) {
          var tag = (c.tagName || "").toUpperCase();
          if (tag === "H2" || tag === "H3") {
            pushCurrent();
            var text = (c.textContent || "").replace(/\s+/g, " ").trim();
            current = { chIdx: chIdx, num: ch.num, chTitle: ch.title, file: ch.file,
              heading: text, headingId: c.getAttribute("data-search-hid") || "", text: text + " " };
          } else if (tag === "SCRIPT" || tag === "STYLE" || tag === "SVG") {
            /* 跳过脚本与 SVG 图内文字 */
          } else {
            walk(c);
          }
        } else if (c.nodeType === 3) {
          current.text += c.nodeValue;
        }
      }
    }
    walk(inner);
    pushCurrent();
    return sections;
  }

  function createSearch() {
    var overlay = el("div", "book-search");
    overlay.innerHTML =
      '<div class="book-search__backdrop" data-search-close></div>' +
      '<div class="book-search__panel" role="dialog" aria-modal="true" aria-label="全书搜索">' +
      '  <div class="book-search__bar">' +
      '    <span class="book-search__icon">' + SEARCH_ICON + '</span>' +
      '    <input type="search" name="book-search" class="book-search__input" placeholder="搜索全书:标题、正文、代码…" autocomplete="off" spellcheck="false" aria-label="搜索全书" />' +
      '    <span class="book-search__kbd" data-search-kbd aria-hidden="true"></span>' +
      '    <button type="button" class="book-search__close" data-search-close aria-label="关闭搜索">Esc</button>' +
      '  </div>' +
      '  <div class="book-search__status" data-search-status></div>' +
      '  <div class="book-search__results" data-search-results></div>' +
      '</div>';
    document.body.appendChild(overlay);

    var input = overlay.querySelector(".book-search__input");
    var kbdEl = overlay.querySelector("[data-search-kbd]");
    var statusEl = overlay.querySelector("[data-search-status]");
    var resultsEl = overlay.querySelector("[data-search-results]");
    var INDEX = [];
    var indexReady = false, indexLoading = false, pending = null, debounce = null, activeIndex = -1;
    var shortcutLabel = searchShortcutHint();
    var READY_HINT = "已索引全书 · " + shortcutLabel + " 打开搜索 · ↑↓ 选择 · Enter 跳转 · Esc 关闭";

    if (kbdEl) kbdEl.textContent = shortcutLabel;
    input.setAttribute("placeholder", "搜索全书… (" + shortcutLabel + ")");

    function setStatus(t) { statusEl.textContent = t; }

    function ensureIndex() {
      if (indexReady || indexLoading) return;
      indexLoading = true;
      setStatus("正在准备全书索引…");
      Promise.all(CHAPTERS.map(function (ch, i) {
        return fetch(ch.file)
          .then(function (r) { return r.text(); })
          .then(function (html) {
            var doc = new DOMParser().parseFromString(html, "text/html");
            var inner = doc.querySelector(".chapter__inner");
            return inner ? extractSections(inner, ch, i) : [];
          })
          .catch(function () { return []; });
      })).then(function (all) {
        INDEX = [];
        all.forEach(function (list) { INDEX = INDEX.concat(list); });
        indexReady = true; indexLoading = false;
        if (!INDEX.length) setStatus("无法建立索引(可能是以 file:// 方式打开)。请改用本地静态服务器访问。");
        else setStatus(READY_HINT);
        if (pending != null) { var q = pending; pending = null; runSearch(q); }
      });
    }

    function makeSnippet(text, terms) {
      var lower = text.toLowerCase();
      var pos = -1;
      terms.forEach(function (t) { var p = lower.indexOf(t); if (p !== -1 && (pos === -1 || p < pos)) pos = p; });
      if (pos < 0) pos = 0;
      var start = Math.max(0, pos - 32);
      var end = Math.min(text.length, pos + 100);
      var raw = (start > 0 ? "…" : "") + text.slice(start, end) + (end < text.length ? "…" : "");
      var html = escapeHtml(raw);
      terms.forEach(function (t) {
        if (!t) return;
        html = html.replace(new RegExp("(" + escapeRegExp(escapeHtml(t)) + ")", "ig"), "<mark>$1</mark>");
      });
      return html;
    }

    function runSearch(query) {
      var q = (query || "").trim().toLowerCase();
      activeIndex = -1;
      persistSearchQuery(query);
      if (!q) { resultsEl.innerHTML = ""; if (indexReady) setStatus(READY_HINT); return; }
      if (!indexReady) { pending = query; return; }
      var terms = q.split(/\s+/).filter(Boolean);
      var results = [];
      INDEX.forEach(function (sec) {
        var headLower = sec.heading.toLowerCase();
        var hay = (sec.heading + " " + sec.text).toLowerCase();
        if (!terms.every(function (t) { return hay.indexOf(t) !== -1; })) return;
        var score = 0;
        terms.forEach(function (t) {
          if (headLower.indexOf(t) !== -1) score += 6;
          var idx = 0, count = 0;
          while ((idx = hay.indexOf(t, idx)) !== -1) { count++; idx += t.length; }
          score += count;
        });
        results.push({ sec: sec, score: score });
      });
      results.sort(function (a, b) { return b.score - a.score || a.sec.chIdx - b.sec.chIdx; });
      results = results.slice(0, 40);
      if (!results.length) {
        resultsEl.innerHTML = '<div class="book-search__empty">没有找到 “' + escapeHtml(query) + '” 相关内容</div>';
        setStatus("无匹配结果"); return;
      }
      setStatus("找到 " + results.length + " 条结果" + (results.length === 40 ? "(仅显示前 40 条)" : ""));
      resultsEl.innerHTML = results.map(function (r, i) {
        var sec = r.sec;
        var href = sec.file + (sec.headingId ? "#" + sec.headingId : "");
        return '<a class="book-search__result" href="' + href + '" data-ri="' + i + '">' +
          '<span class="book-search__result-chapter">第 ' + sec.num + ' 章 · ' + escapeHtml(sec.chTitle) + '</span>' +
          '<span class="book-search__result-heading">' + escapeHtml(sec.heading) + '</span>' +
          '<span class="book-search__result-snippet">' + makeSnippet(sec.text, terms) + '</span>' +
          '</a>';
      }).join("");
    }

    function setActive(i) {
      var items = resultsEl.querySelectorAll(".book-search__result");
      if (!items.length) return;
      if (i < 0) i = items.length - 1;
      if (i >= items.length) i = 0;
      activeIndex = i;
      items.forEach(function (n, idx) { n.classList.toggle("is-active", idx === i); });
      items[i].scrollIntoView({ block: "nearest" });
    }

    function open() {
      if (overlay.classList.contains("is-open")) {
        input.focus();
        return;
      }
      overlay.classList.add("is-open");
      document.body.style.overflow = "hidden";
      ensureIndex();
      var saved = loadSearchQuery();
      if (saved) {
        input.value = saved;
        pending = saved;
        if (indexReady) runSearch(saved);
      }
      setTimeout(function () {
        input.focus();
        if (saved) {
          var len = saved.length;
          input.setSelectionRange(len, len);
        } else {
          input.select();
        }
      }, 30);
    }
    function close() {
      overlay.classList.remove("is-open");
      document.body.style.overflow = "";
    }

    function pageBaseName() {
      var p = location.pathname || "";
      var i = p.lastIndexOf("/");
      return i >= 0 ? p.slice(i + 1) : p;
    }

    function followSearchResult(href) {
      if (!href) return;
      var hashIdx = href.indexOf("#");
      var file = hashIdx >= 0 ? href.slice(0, hashIdx) : href;
      var hash = hashIdx >= 0 ? href.slice(hashIdx + 1) : "";
      var here = pageBaseName();

      if (file && file !== here) {
        window.location.href = href;
        return;
      }

      close();
      if (!hash) {
        window.scrollTo({ top: 0, behavior: "smooth" });
        if (window.history && window.history.replaceState) {
          window.history.replaceState(null, "", here);
        }
        return;
      }

      var id = hash;
      try { id = decodeURIComponent(hash); } catch (e) { /* keep raw */ }
      var target = document.getElementById(id);
      if (target) {
        scrollHeadingIntoView(target, true);
        if (window.history && window.history.replaceState) {
          window.history.replaceState(null, "", "#" + hash);
        }
      } else {
        window.location.hash = hash;
      }
    }

    overlay.querySelectorAll("[data-search-close]").forEach(function (n) { n.addEventListener("click", close); });
    resultsEl.addEventListener("click", function (e) {
      var link = e.target.closest(".book-search__result");
      if (!link) return;
      e.preventDefault();
      followSearchResult(link.getAttribute("href"));
    });
    input.addEventListener("input", function () {
      if (debounce) clearTimeout(debounce);
      var v = input.value;
      debounce = setTimeout(function () { runSearch(v); }, 140);
    });
    input.addEventListener("keydown", function (e) {
      if (e.key === "ArrowDown") { e.preventDefault(); setActive(activeIndex + 1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); setActive(activeIndex - 1); }
      else if (e.key === "Enter") {
        var items = resultsEl.querySelectorAll(".book-search__result");
        var target = items[activeIndex >= 0 ? activeIndex : 0];
        if (target) { e.preventDefault(); followSearchResult(target.getAttribute("href")); }
      }
    });
    overlay.addEventListener("keydown", function (e) { if (e.key === "Escape") { e.preventDefault(); close(); } });

    return { open: open, close: close, preload: ensureIndex };
  }

  function makeSearchTrigger(search, cls, label, html) {
    var btn = el("button", cls, html);
    btn.type = "button";
    var hint = searchShortcutHint();
    btn.setAttribute("aria-label", label + " (" + hint + ")");
    btn.title = label + " · " + hint;
    btn.addEventListener("click", search.open);
    return btn;
  }

  function setupSearchShortcut(search) {
    document.addEventListener("keydown", function (e) {
      var t = e.target;
      var typing = t && /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName);
      if ((e.metaKey || e.ctrlKey) && (e.key === "k" || e.key === "K")) { e.preventDefault(); search.open(); return; }
      if (e.key === "/" && !typing && !e.metaKey && !e.ctrlKey && !e.altKey) { e.preventDefault(); search.open(); }
    });
  }

  // ---------- 术语表:点击弹出释义 ----------
  var GLOSSARY = { byId: {}, matchers: [], ready: false, loading: null };

  // 正文自动匹配只用 data-aliases; 这些太泛, 即便写了别名也不链
  var GLOSSARY_ALIAS_BLOCK = {
    model: true, token: true, tokens: true, training: true, train: true, data: true,
    hidden: true, attention: true, parameter: true, parameters: true, vector: true,
    matrix: true, tensor: true, tensors: true, quantization: true, softmax: true,
    embedding: true, encoder: true, decoder: true, loss: true, gradient: true,
    模型: true, 训练: true, 数据: true, 参数: true, 向量: true, 矩阵: true, 张量: true,
    量化: true, 分词: true, 点积: true, 归一化: true, 注意力: true, 嵌入: true,
    编码: true, 解码: true, 损失: true, 梯度: true, 前馈: true, 残差: true
  };

  function glossaryAliasOk(alias) {
    if (!alias) return false;
    var a = alias.trim();
    if (!a) return false;
    if (GLOSSARY_ALIAS_BLOCK[a.toLowerCase()]) return false;
    if (/^[a-zA-Z_]+$/.test(a) && a.length < 4) return false;
    if (/^[\u4e00-\u9fff]+$/.test(a) && a.length < 3) return false;
    return true;
  }

  function glossaryBasePath() {
    var path = location.pathname || "";
    if (/\/docs\/dl-book\//.test(path) || /\/dl-book\//.test(path)) return "glossary.html";
    if (/chapter-\d+\.html$/.test(path) || /glossary\.html$/.test(path) || /index\.html$/.test(path)) {
      return "glossary.html";
    }
    return "glossary.html";
  }

  function parseGlossaryItem(li) {
    if (!li.id) return null;
    var strong = li.querySelector("strong");
    var title = strong ? strong.textContent.replace(/\s+/g, " ").trim() : li.id;
    var link = li.querySelector("a[href]");
    var href = link ? link.getAttribute("href") : ("glossary.html#" + li.id);
    // 释义里常有 “预测 → 对答案 → …”, 不能按 → 截断; 先去掉章节链接再取 — 后正文
    var clone = li.cloneNode(true);
    [].slice.call(clone.querySelectorAll("a")).forEach(function (a) { a.remove(); });
    var raw = (clone.textContent || "").replace(/\s+/g, " ").trim();
    var def = "";
    var dash = raw.indexOf("—");
    if (dash !== -1) def = raw.slice(dash + 1).trim();
    var aliases = (li.getAttribute("data-aliases") || "").split(",").map(function (s) {
      return s.trim();
    }).filter(glossaryAliasOk);
    var seen = {};
    aliases = aliases.filter(function (a) {
      var k = a.toLowerCase();
      if (seen[k]) return false;
      seen[k] = true;
      return true;
    });
    return { id: li.id, title: title, def: def, href: href, aliases: aliases };
  }

  function aliasRegex(alias) {
    var escaped = escapeRegExp(alias);
    if (/^[A-Za-z_][\w.\-/]*$/.test(alias)) {
      return new RegExp("\\b" + escaped + "_?\\b", "gi");
    }
    return new RegExp(escaped, "gi");
  }

  function loadGlossary() {
    if (GLOSSARY.ready) return Promise.resolve(GLOSSARY);
    if (GLOSSARY.loading) return GLOSSARY.loading;
    GLOSSARY.loading = fetch(glossaryBasePath())
      .then(function (r) { return r.text(); })
      .then(function (html) {
        var doc = new DOMParser().parseFromString(html, "text/html");
        var items = [].slice.call(doc.querySelectorAll(".glossary li[id]"));
        GLOSSARY.byId = {};
        GLOSSARY.matchers = [];
        items.forEach(function (li) {
          var entry = parseGlossaryItem(li);
          if (!entry) return;
          GLOSSARY.byId[entry.id] = entry;
          entry.aliases.forEach(function (alias) {
            if (!glossaryAliasOk(alias)) return;
            GLOSSARY.matchers.push({ id: entry.id, alias: alias, regex: aliasRegex(alias) });
          });
        });
        GLOSSARY.matchers.sort(function (a, b) { return b.alias.length - a.alias.length; });
        GLOSSARY.ready = true;
        return GLOSSARY;
      })
      .catch(function () {
        GLOSSARY.ready = true;
        return GLOSSARY;
      });
    return GLOSSARY.loading;
  }

  function glossaryShouldSkip(node) {
    var p = node;
    while (p) {
      if (!p.tagName) { p = p.parentElement; continue; }
      var tag = p.tagName.toUpperCase();
      if (tag === "SCRIPT" || tag === "STYLE" || tag === "SVG" || tag === "TEXTAREA" ||
          tag === "INPUT" || tag === "CODE" || tag === "KBD" || tag === "SAMP" || tag === "PRE") return true;
      if (/^H[1-6]$/.test(tag)) return true;
      if (p.classList) {
        if (p.classList.contains("chapter__eyebrow") || p.classList.contains("glossary-popover") ||
            p.classList.contains("glossary") || p.classList.contains("code-walk__code") ||
            p.classList.contains("code-walk") || p.classList.contains("diagram") ||
            p.classList.contains("term--glossary")) return true;
      }
      if (p.classList && p.classList.contains("chapter__inner")) return false;
      p = p.parentElement;
    }
    return true;
  }

  function findGlossaryMatches(text) {
    var all = [];
    GLOSSARY.matchers.forEach(function (matcher) {
      var re = matcher.regex;
      re.lastIndex = 0;
      var m;
      while ((m = re.exec(text)) !== null) {
        all.push({ start: m.index, end: m.index + m[0].length, id: matcher.id, text: m[0] });
      }
    });
    all.sort(function (a, b) {
      if (a.start !== b.start) return a.start - b.start;
      return (b.end - b.start) - (a.end - a.start);
    });
    var picked = [];
    var lastEnd = 0;
    all.forEach(function (m) {
      if (m.start < lastEnd) return;
      picked.push(m);
      lastEnd = m.end;
    });
    return picked;
  }

  function wrapGlossaryTextNode(node) {
    var text = node.nodeValue;
    if (!text || !text.trim()) return;
    var matches = findGlossaryMatches(text);
    if (!matches.length) return;
    var frag = document.createDocumentFragment();
    var last = 0;
    matches.forEach(function (m) {
      if (m.start > last) frag.appendChild(document.createTextNode(text.slice(last, m.start)));
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "term--glossary";
      btn.textContent = text.slice(m.start, m.end);
      btn.setAttribute("data-glossary-id", m.id);
      btn.setAttribute("aria-label", "查看术语释义: " + text.slice(m.start, m.end));
      frag.appendChild(btn);
      last = m.end;
    });
    if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
    node.parentNode.replaceChild(frag, node);
  }

  function linkGlossaryInText(root) {
    if (!GLOSSARY.ready || !GLOSSARY.matchers.length) return;
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
    var nodes = [];
    while (walker.nextNode()) {
      if (!glossaryShouldSkip(walker.currentNode.parentElement)) nodes.push(walker.currentNode);
    }
    nodes.forEach(wrapGlossaryTextNode);
  }

  function findGlossaryForTermLabel(label) {
    var lower = String(label || "").toLowerCase();
    var best = null;
    var bestScore = 0;
    Object.keys(GLOSSARY.byId).forEach(function (id) {
      var e = GLOSSARY.byId[id];
      var score = 0;
      e.aliases.forEach(function (a) {
        var al = a.toLowerCase();
        if (lower.indexOf(al) !== -1) score = Math.max(score, al.length);
        if (al.indexOf(lower) !== -1) score = Math.max(score, lower.length);
      });
      if (e.title.toLowerCase().indexOf(lower) !== -1) score = Math.max(score, lower.length);
      if (score > bestScore) { bestScore = score; best = e; }
    });
    return bestScore >= 3 ? best : null;
  }

  function enhanceManualTerms(root) {
    [].slice.call(root.querySelectorAll("span.term")).forEach(function (span) {
      if (span.classList.contains("term--glossary")) return;
      if (glossaryShouldSkip(span)) return;
      var entry = findGlossaryForTermLabel(span.textContent);
      if (!entry) return;
      span.classList.add("term--glossary");
      span.setAttribute("data-glossary-id", entry.id);
      span.setAttribute("tabindex", "0");
      span.setAttribute("role", "button");
      span.setAttribute("aria-label", "查看术语释义: " + span.textContent.replace(/\s+/g, " ").trim());
    });
  }

  function createGlossaryPopover() {
    var backdrop = el("div", "glossary-backdrop");
    backdrop.hidden = true;
    var pop = el("div", "glossary-popover is-below");
    pop.hidden = true;
    pop.setAttribute("role", "dialog");
    pop.setAttribute("aria-modal", "false");
    pop.innerHTML =
      '<div class="glossary-popover__arrow"></div>' +
      '<div class="glossary-popover__inner">' +
      '  <p class="glossary-popover__title" data-glossary-title></p>' +
      '  <p class="glossary-popover__body" data-glossary-body></p>' +
      '  <div class="glossary-popover__foot">' +
      '    <a class="glossary-popover__link" data-glossary-link href="glossary.html">术语表</a>' +
      '    <span class="glossary-popover__hint">Esc 关闭</span>' +
      "  </div>" +
      "</div>";
    document.body.appendChild(backdrop);
    document.body.appendChild(pop);

    var activeTrigger = null;

    function close() {
      backdrop.hidden = true;
      pop.hidden = true;
      if (activeTrigger) activeTrigger.classList.remove("is-active");
      activeTrigger = null;
    }

    function positionNear(trigger) {
      var rect = trigger.getBoundingClientRect();
      var margin = 10;
      var popRect = pop.getBoundingClientRect();
      var top = rect.bottom + margin;
      var placeBelow = true;
      if (top + popRect.height > window.innerHeight - margin) {
        top = rect.top - popRect.height - margin;
        placeBelow = false;
      }
      if (top < margin) top = margin;
      var left = rect.left + rect.width / 2 - popRect.width / 2;
      left = Math.max(margin, Math.min(left, window.innerWidth - popRect.width - margin));
      pop.style.top = top + "px";
      pop.style.left = left + "px";
      pop.classList.toggle("is-below", placeBelow);
      pop.classList.toggle("is-above", !placeBelow);
      var arrowX = rect.left + rect.width / 2 - left - 5;
      arrowX = Math.max(12, Math.min(arrowX, popRect.width - 20));
      pop.style.setProperty("--arrow-x", arrowX + "px");
    }

    function open(trigger, entry) {
      if (activeTrigger === trigger) { close(); return; }
      if (activeTrigger) activeTrigger.classList.remove("is-active");
      activeTrigger = trigger;
      activeTrigger.classList.add("is-active");
      pop.querySelector("[data-glossary-title]").textContent = entry.title;
      pop.querySelector("[data-glossary-body]").textContent = entry.def || "详见术语表。";
      var link = pop.querySelector("[data-glossary-link]");
      link.href = entry.href || ("glossary.html#" + entry.id);
      link.textContent = entry.href && entry.href.indexOf("glossary") === 0 ? "术语表" : "详解章节 →";
      backdrop.hidden = false;
      pop.hidden = false;
      requestAnimationFrame(function () { positionNear(trigger); });
    }

    backdrop.addEventListener("click", close);
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && !pop.hidden) { e.preventDefault(); close(); }
    });
    window.addEventListener("resize", function () {
      if (!pop.hidden && activeTrigger) positionNear(activeTrigger);
    });
    window.addEventListener("scroll", function () {
      if (!pop.hidden && activeTrigger) positionNear(activeTrigger);
    }, true);

    return {
      open: open,
      close: close,
      handleClick: function (e) {
        var t = e.target.closest(".term--glossary,[data-glossary-id]");
        if (!t || !t.getAttribute("data-glossary-id")) return;
        e.preventDefault();
        e.stopPropagation();
        var entry = GLOSSARY.byId[t.getAttribute("data-glossary-id")];
        if (entry) open(t, entry);
      },
      handleKeydown: function (e) {
        if (e.key !== "Enter" && e.key !== " ") return;
        var t = e.target.closest(".term--glossary,[data-glossary-id]");
        if (!t) return;
        e.preventDefault();
        t.click();
      }
    };
  }

  var glossaryPopover = null;

  function setupGlossary(root) {
    if (!root || /glossary\.html$/.test(location.pathname)) return;
    enhanceManualTerms(root);
    linkGlossaryInText(root);
    if (!glossaryPopover) glossaryPopover = createGlossaryPopover();
    root.addEventListener("click", glossaryPopover.handleClick);
    root.addEventListener("keydown", glossaryPopover.handleKeydown);
  }

  // 跨页跳转到 #锚点: id 是 JS 运行期才赋的, 浏览器原生跳转会失败, 这里补一次带偏移的滚动
  function handleInitialHash() {
    if (!window.location.hash) return;
    var id;
    try { id = decodeURIComponent(window.location.hash.slice(1)); } catch (e) { id = window.location.hash.slice(1); }
    if (!id) return;
    var target = document.getElementById(id);
    if (!target) return;
    requestAnimationFrame(function () {
      requestAnimationFrame(function () { scrollHeadingIntoView(target, false); });
    });
  }

  // ---------- 封面: 继续阅读 ----------
  function setupCover(search) {
    setupReveal();
    var actions = document.querySelector(".cover__actions");
    if (actions && search) {
      var hint = searchShortcutHint();
      actions.appendChild(makeSearchTrigger(search, "button button--ghost", "搜索全书", "搜索全书 · " + hint));
    }
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
    var search = createSearch();
    setupSearchShortcut(search);
    search.preload();
    if (document.body.hasAttribute("data-cover")) {
      setupCover(search);
      return;
    }
    var idx = chapterIndex();
    var hasChapterBody = !!document.querySelector(".chapter .chapter__inner");
    if ((idx < 0 || idx >= CHAPTERS.length) && !hasChapterBody) {
      setupReveal();
      return;
    }
    var isChapter = idx >= 0 && idx < CHAPTERS.length;
    var toc = buildToc(idx);
    var refs = buildHeader(idx, toc, search);
    buildDesktopLayout(idx);
    if (isChapter) {
      buildPrevNext(idx);
      setupKeyboard(idx);
    }
    setupProgress(refs);
    setupReveal();
    handleInitialHash();
    loadGlossary().then(function () {
      var inner = document.querySelector(".chapter__inner");
      if (inner) setupGlossary(inner);
    });
    if (isChapter) store.set(STORAGE_LAST, CHAPTERS[idx].file);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
