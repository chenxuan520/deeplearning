/*
 * mythos.js — 专题页《从沙子到 Mythos》
 * 左侧"半圆转盘"时间轴: 节点沿左侧圆弧分布, 当前屏对应节点转到弧尖(3 点方向)、
 * 放大高亮, 邻近节点次亮, 其余淡出。滚动即转盘旋转。
 * 另含: 进屏淡入(is-in)、顶部时间码与进度条、键盘翻页、hash 定位。
 * 纯静态零依赖, 滚动容器是 .deck。
 */
(function () {
  "use strict";

  var deck = document.getElementById("deck");
  var track = document.getElementById("dial-track");
  var progress = document.querySelector(".deck-progress");
  var reelCode = document.getElementById("reel-code");
  if (!deck || !track) return;

  var slides = [].slice.call(deck.querySelectorAll(".slide"));
  var total = slides.length;

  // 转盘几何: 节点绕"屏幕左外侧"的圆心排布, 弧向右鼓。
  // 当前项停在 0°(弧尖=正右, 与指针对齐); 上方为负角、下方为正角。
  var RADIUS = 360;
  var STEP_DEG = 4.6;    // 相邻节点角度差(半径≈0.95vh, 保证竖直间距适中)
  var VISIBLE = 9;       // 单侧最多显示几个(超出淡到 0)

  function actClassOf(slide) {
    var m = (slide.className || "").match(/\bact-\d\b/);
    return m ? m[0] : "act-1";
  }

  function dialEraOf(slide) {
    var raw = slide.getAttribute("data-dial-era") || slide.getAttribute("data-era") || "";
    raw = raw.trim();
    if (slide.classList.contains("slide--act") || slide.classList.contains("slide--constellation")) return raw;
    if (/^(序|全景|抽象层|表示层|贯穿始终)$/.test(raw)) return "";
    return raw.split("·")[0].trim();
  }

  function dialLabelOf(slide) {
    var raw = slide.getAttribute("data-label") || slide.id;
    if (slide.classList.contains("slide--act")) {
      var parts = raw.split("·");
      return (parts[1] || raw).trim();
    }
    return raw;
  }

  slides.forEach(function (s, i) { if (!s.id) s.id = "slide-n-" + i; });

  // 读取转盘几何: 与 mythos.css 的装饰弧严格对齐。
  //   CSS 弧: 直径 190vh → 半径 = 0.95 × 视口高; 弧尖(右缘)落在 --dial-tip。
  //   这里读同样的值, 节点才会正好贴在那条弧上。
  var TIP_X = 130;
  function readGeom() {
    var probe = document.createElement("div");
    probe.style.position = "absolute";
    probe.style.left = "var(--dial-tip)";
    probe.style.visibility = "hidden";
    document.body.appendChild(probe);
    var tip = probe.getBoundingClientRect().left;
    probe.remove();
    TIP_X = isNaN(tip) ? 130 : tip;
    RADIUS = window.innerHeight * 0.95; // = CSS 190vh 圆的半径
  }

  // ---------- 生成节点 ----------
  var items = slides.map(function (slide, i) {
    var a = document.createElement("a");
    a.className = "dial-item " + actClassOf(slide);
    a.href = "#" + slide.id;

    var era = document.createElement("span");
    era.className = "dial-item__era";
    era.textContent = dialEraOf(slide);
    if (!era.textContent) a.classList.add("dial-item--no-era");
    var dot = document.createElement("span");
    dot.className = "dial-item__dot";
    var label = document.createElement("span");
    label.className = "dial-item__label";
    label.textContent = dialLabelOf(slide);

    a.appendChild(era);
    a.appendChild(dot);
    a.appendChild(label);
    a.addEventListener("click", function (ev) {
      ev.preventDefault();
      slide.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    track.appendChild(a);
    return a;
  });

  // ---------- 把节点摆到弧上 (相对当前项 active) ----------
  // 圆心在屏幕左外侧 (tipX - R, cy); 当前项角度 0° → 落在弧尖(最右, x=tipX)。
  // 角度向上为负、向下为正。节点用 translate(-50%,-50%) 居中, 文字不旋转。
  function layout(active) {
    var tipX = TIP_X;                // 弧尖横坐标(与 needle 对齐, 在窄带内)
    var cy = track.clientHeight / 2; // 弧尖纵坐标 = 竖直中线
    var cx = tipX - RADIUS;          // 圆心(屏幕左外)
    items.forEach(function (a, i) {
      var d = i - active;
      var deg = d * STEP_DEG;                 // 当前项 0°
      var rad = (deg * Math.PI) / 180;
      var x = cx + Math.cos(rad) * RADIUS;    // 0° → cx+R = tipX
      var y = cy + Math.sin(rad) * RADIUS;
      var away = Math.abs(d);
      var opacity = away === 0 ? 1 : Math.max(0, 1 - away / VISIBLE);
      a.style.left = x.toFixed(1) + "px";
      a.style.top = y.toFixed(1) + "px";
      a.style.opacity = opacity.toFixed(2);
      a.style.pointerEvents = opacity < 0.12 ? "none" : "auto";
      a.classList.toggle("is-active", d === 0);
      a.classList.toggle("is-near", away === 1);
    });
  }

  // ---------- 当前屏 + 时间码 ----------
  var currentIdx = -1;
  var programmaticTargetIdx = -1;
  var programmaticUnlockTimer = 0;
  // 滚动到某屏时, 把地址栏 #hash 换成该屏 id: 用 replaceState, 不触发跳转、不灌历史。
  // 第一屏(封面)清掉 hash, 保持首屏 URL 干净。
  function updateHash(i) {
    var id = slides[i] && slides[i].id;
    if (!id) return;
    var url = i === 0 ? location.pathname + location.search : "#" + id;
    try { history.replaceState(null, "", url); } catch (e) {}
  }
  var creditEgg = document.querySelector(".story-credit__egg");
  var constellationSlide = document.getElementById("slide-constellation");

  function setActive(i) {
    if (i < 0 || i >= items.length) return;
    if (i !== currentIdx) {
      currentIdx = i;
      if (reelCode) reelCode.textContent = String(i + 1).padStart(2, "0") + " / " + total;
      updateHash(i);
      // 当前幕的 act-N 写到 body: 固定背景底据此平滑过渡主题色(见 mythos.css body::before)。
      // act-6(群星/高峰)保留各自盒子内背景, 固定底不设 act-6, 不影响它们。
      document.body.setAttribute("data-act", actClassOf(slides[i]));
      if (creditEgg) {
        var onConstellation = slides[i] === constellationSlide;
        creditEgg.classList.toggle("is-live", onConstellation);
        creditEgg.setAttribute("aria-disabled", onConstellation ? "false" : "true");
        if (!onConstellation && typeof closeConstellationPopover === "function") closeConstellationPopover();
      }
    }
    layout(i);
  }
  function scrollToSlide(slide, behavior) {
    var idx = slides.indexOf(slide);
    if (idx < 0) return;
    setActive(idx);
    lockActiveUntilScrollStops(idx);
    deck.scrollTo({ top: slide.offsetTop, behavior: behavior || "smooth" });
  }
  function lockActiveUntilScrollStops(i) {
    programmaticTargetIdx = i;
    clearTimeout(programmaticUnlockTimer);
    programmaticUnlockTimer = setTimeout(function () {
      setActive(programmaticTargetIdx);
      programmaticTargetIdx = -1;
    }, 1800);
  }

  // ---------- 进屏淡入 + 高亮 ----------
  if ("IntersectionObserver" in window) {
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          e.target.classList.add("is-in");
          if (e.intersectionRatio >= 0.55 && programmaticTargetIdx < 0) setActive(slides.indexOf(e.target));
        }
      });
    }, { root: deck, threshold: [0.2, 0.55, 0.85] });
    slides.forEach(function (s) { obs.observe(s); });
  } else {
    slides.forEach(function (s) { s.classList.add("is-in"); });
    deck.addEventListener("scroll", function () {
      if (programmaticTargetIdx >= 0) return;
      setActive(Math.round(deck.scrollTop / deck.clientHeight));
    });
  }

  // ---------- 进度条 ----------
  function updateProgress() {
    var max = deck.scrollHeight - deck.clientHeight;
    if (progress) progress.style.width = (max > 0 ? (deck.scrollTop / max) * 100 : 0).toFixed(2) + "%";
    if (programmaticTargetIdx >= 0) {
      var targetTop = slides[programmaticTargetIdx].offsetTop;
      if (Math.abs(deck.scrollTop - targetTop) < 3) {
        setActive(programmaticTargetIdx);
        programmaticTargetIdx = -1;
        clearTimeout(programmaticUnlockTimer);
      }
    }
  }
  deck.addEventListener("scroll", updateProgress, { passive: true });

  // ---------- 键盘翻页 ----------
  function go(delta) {
    var n = Math.min(slides.length - 1, Math.max(0, currentIdx + delta));
    scrollToSlide(slides[n], "smooth");
  }
  function isEditableTarget(target) {
    if (!target) return false;
    if (target.isContentEditable) return true;
    return /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName || "");
  }
  document.addEventListener("keydown", function (e) {
    if (e.defaultPrevented || e.isComposing || e.altKey || e.ctrlKey || e.metaKey) return;
    if (isEditableTarget(e.target)) return;
    var k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    if (k === "ArrowDown" || k === "ArrowRight" || k === "PageDown" || k === "j" || k === "l") { e.preventDefault(); go(1); }
    else if (k === "ArrowUp" || k === "ArrowLeft" || k === "PageUp" || k === "k" || k === "h") { e.preventDefault(); go(-1); }
    else if (k === "Home") { e.preventDefault(); scrollToSlide(slides[0], "smooth"); }
    else if (k === "End") { e.preventDefault(); scrollToSlide(slides[slides.length - 1], "smooth"); }
  });

  // ---------- 左侧时间线滚动 ----------
  var dialTargetIdx = 0;
  var dialScrollTimer = 0;
  var dialScrollAcc = 0;
  var dialScrolling = false;
  var DIAL_WHEEL_STEP = 28;
  function scrollDialTo(idx) {
    idx = Math.min(slides.length - 1, Math.max(0, idx));
    dialTargetIdx = idx;
    dialScrolling = true;
    setActive(idx);
    lockActiveUntilScrollStops(idx);
    clearTimeout(dialScrollTimer);
    dialScrollTimer = setTimeout(function () {
      dialScrolling = false;
      scrollToSlide(slides[dialTargetIdx], "smooth");
    }, 180);
  }
  track.addEventListener("wheel", function (e) {
    if (window.matchMedia("(max-width: 920px)").matches) return;
    e.preventDefault();
    dialScrollAcc += e.deltaY || e.deltaX || 0;
    if (Math.abs(dialScrollAcc) < DIAL_WHEEL_STEP) return;
    var step = dialScrollAcc > 0 ? 1 : -1;
    dialScrollAcc = 0;
    scrollDialTo((dialScrolling ? dialTargetIdx : (currentIdx < 0 ? 0 : currentIdx)) + step);
  }, { passive: false });

  // ---------- 群星弹窗 + 作者彩蛋(仅群星幕) ----------
  var closeConstellationPopover = function () {};
  function buildConstellationPopover() {
    var stars = [].slice.call(document.querySelectorAll(".constellation-star"));
    if (!stars.length && !creditEgg) return;

    var pop = document.createElement("aside");
    pop.className = "constellation-popover";
    pop.setAttribute("aria-live", "polite");
    pop.innerHTML =
      '<button type="button" class="constellation-popover__close" aria-label="关闭">×</button>' +
      '<p class="constellation-popover__kind"></p>' +
      '<h3 class="constellation-popover__title"></h3>' +
      '<p class="constellation-popover__text" data-popover-work></p>' +
      '<p class="constellation-popover__text" data-popover-meaning></p>';
    document.body.appendChild(pop);

    var closeBtn = pop.querySelector(".constellation-popover__close");
    var kind = pop.querySelector(".constellation-popover__kind");
    var title = pop.querySelector(".constellation-popover__title");
    var work = pop.querySelector("[data-popover-work]");
    var meaning = pop.querySelector("[data-popover-meaning]");

    function close() {
      stars.forEach(function (s) { s.classList.remove("is-selected"); });
      pop.classList.remove("is-open");
    }
    closeConstellationPopover = close;
    function placeNear(anchor) {
      if (window.matchMedia("(max-width: 920px)").matches) {
        pop.style.left = "";
        pop.style.top = "";
        return;
      }
      var gap = 14;
      var pad = 14;
      var rect = anchor.getBoundingClientRect();
      pop.classList.add("is-open");
      var popRect = pop.getBoundingClientRect();
      var left = rect.right + gap;
      var top = rect.top + rect.height / 2 - popRect.height / 2;
      if (left + popRect.width > window.innerWidth - pad) left = rect.left - popRect.width - gap;
      if (left < pad) left = pad;
      top = Math.min(window.innerHeight - popRect.height - pad, Math.max(pad, top));
      pop.style.left = left + "px";
      pop.style.top = top + "px";
    }
    function openFrom(anchor, data) {
      stars.forEach(function (s) { s.classList.remove("is-selected"); });
      if (anchor.classList && anchor.classList.contains("constellation-star")) {
        anchor.classList.add("is-selected");
      }
      kind.textContent = data.kind || "";
      title.textContent = data.title || "";
      work.textContent = data.work || "";
      meaning.textContent = data.meaning || "";
      meaning.style.display = meaning.textContent ? "" : "none";
      placeNear(anchor);
    }
    function open(star) {
      openFrom(star, {
        kind: star.getAttribute("data-kind") || "",
        title: star.getAttribute("data-title") || star.textContent.trim(),
        work: star.getAttribute("data-work") || "",
        meaning: star.getAttribute("data-meaning") || ""
      });
    }

    stars.forEach(function (star) {
      star.addEventListener("click", function () { open(star); });
    });
    if (creditEgg) {
      creditEgg.addEventListener("click", function () {
        if (!creditEgg.classList.contains("is-live")) return;
        openFrom(creditEgg, {
          kind: "彩蛋 · 作者",
          title: "chenxuan",
          work: "在 2026 年制作《从沙子到 Mythos》这份致敬文档。",
          meaning: ""
        });
      });
    }
    closeBtn.addEventListener("click", close);
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") close();
    });
    document.addEventListener("click", function (e) {
      if (!pop.classList.contains("is-open")) return;
      if (pop.contains(e.target) || e.target.closest(".constellation-star") || e.target.closest(".story-credit__egg")) return;
      close();
    });
  }

  // ---------- 移动端跳转 ----------
  function buildMobileJump() {
    var targets = slides.filter(function (s, i) {
      return i === 0 ||
        s.id === "slide-overview" ||
        s.classList.contains("slide--act") ||
        s.classList.contains("slide--constellation") ||
        s.classList.contains("slide--finale");
    });
    if (!targets.length) return;

    var wrap = document.createElement("div");
    wrap.className = "mobile-jump";
    wrap.innerHTML =
      '<button type="button" class="mobile-jump__btn" aria-expanded="false" aria-controls="mobile-jump-panel">跳转</button>' +
      '<div class="mobile-jump__backdrop" data-mobile-jump-close></div>' +
      '<div class="mobile-jump__panel" id="mobile-jump-panel" role="dialog" aria-modal="true" aria-label="跳转到某一幕">' +
      '  <div class="mobile-jump__head">' +
      '    <strong>跳转到某一幕</strong>' +
      '    <button type="button" class="mobile-jump__close" data-mobile-jump-close aria-label="关闭">Esc</button>' +
      '  </div>' +
      '  <div class="mobile-jump__list"></div>' +
      '</div>';
    document.body.appendChild(wrap);

    var btn = wrap.querySelector(".mobile-jump__btn");
    var list = wrap.querySelector(".mobile-jump__list");
    targets.forEach(function (slide) {
      var item = document.createElement("button");
      item.type = "button";
      item.className = "mobile-jump__item " + actClassOf(slide);
      var era = slide.getAttribute("data-era") || "";
      var label = slide.getAttribute("data-label") || slide.id;
      if (slide.classList.contains("slide--act")) era = "";
      if (/^(序|全景)$/.test(era)) era = "";
      if (!era) item.classList.add("mobile-jump__item--no-era");
      item.innerHTML =
        '<span class="mobile-jump__era">' + era + '</span>' +
        '<span class="mobile-jump__label">' + label + '</span>';
      item.addEventListener("click", function () {
        close();
        scrollToSlide(slide, "smooth");
      });
      list.appendChild(item);
    });

    function open() {
      wrap.classList.add("is-open");
      btn.setAttribute("aria-expanded", "true");
    }
    function close() {
      wrap.classList.remove("is-open");
      btn.setAttribute("aria-expanded", "false");
    }
    btn.addEventListener("click", function () {
      wrap.classList.contains("is-open") ? close() : open();
    });
    wrap.querySelectorAll("[data-mobile-jump-close]").forEach(function (n) {
      n.addEventListener("click", close);
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && wrap.classList.contains("is-open")) close();
    });
  }

  // ---------- 初始化 & 重排 ----------
  // 带 #hash 进来时: 定位到对应屏并高亮; 否则从封面开始。
  var initialHash = location.hash ? location.hash.slice(1) : "";
  function boot() {
    readGeom();
    var startIdx = 0;
    if (initialHash) {
      var t = document.getElementById(initialHash);
      var ti = t ? slides.indexOf(t) : -1;
      if (ti >= 0) startIdx = ti;
    }
    if (startIdx > 0) deck.scrollTo({ top: slides[startIdx].offsetTop, behavior: "auto" });
    setActive(startIdx);
    updateProgress();
  }
  window.addEventListener("resize", function () { readGeom(); layout(currentIdx < 0 ? 0 : currentIdx); });
  buildConstellationPopover();
  buildMobileJump();
  requestAnimationFrame(boot);
})();
