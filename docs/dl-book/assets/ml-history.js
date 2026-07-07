/*
 * ml-history.js — 专题页《百年长夜与黎明》(机器学习群星史)
 * 横向一屏一幕 (scroll-snap x)。底部胶片时间线随当前幕居中聚焦;
 * 顶部时间码 + 进度条; 幕内元素进屏淡入; 键盘/滚轮/按钮翻幕。
 * 图片外链失败时回退到 .is-broken 文字占位。纯静态零依赖。
 */
(function () {
  "use strict";

  var scroller = document.querySelector("[data-ml-history-scroll]");
  var timeline = document.querySelector("[data-ml-history-timeline]");
  var tlView = timeline ? timeline.parentElement : null;
  var era = document.querySelector("[data-ml-history-era]");
  var code = document.querySelector("[data-ml-history-code]");
  var progress = document.querySelector("[data-ml-history-progress]");
  var prevBtn = document.querySelector("[data-ml-history-prev]");
  var nextBtn = document.querySelector("[data-ml-history-next]");
  if (!scroller) return;

  var slides = Array.prototype.slice.call(scroller.querySelectorAll("[data-year]"));
  var total = slides.length;
  var points = [];
  var currentIdx = -1;
  var lockTimer = 0;
  var locked = false;
  var timelineDragging = false;
  var timelinePointer = 0;
  var timelineStartLeft = 0;
  var timelineMoved = false;
  var suppressTimelineClick = false;
  var timelinePointerId = null;
  var timelineProgrammatic = false;
  var timelineProgrammaticTimer = 0;
  var timelineSnapTimer = 0;

  // ---------- 底部路线热度时间线刻度 ----------
  // 节点仍按 slide index 一格一格移动; SVG 同时画符号/连接两条热度曲线。
  // 纵向高度统一表示热度: 谁更高, 谁在那个时间点更热。
  var STEP = 92;   // 相邻刻度水平间距(px)
  var SPAN = 6;    // 单侧最多强调几个邻近刻度
  var railSvg = null;
  var symbolPath = null;
  var connectionPath = null;

  if (timeline) {
    slides.forEach(function (slide, i) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "mh-timepoint mh-timepoint--" + (slide.dataset.line || "bridge");
      btn.dataset.line = slide.dataset.line || "bridge";
      btn.setAttribute(
        "aria-label",
        (slide.dataset.year || "") + " " + (slide.dataset.title || "")
      );
      btn.innerHTML =
        '<span class="mh-timepoint__dot"></span>' +
        '<span class="mh-timepoint__year">' + (slide.dataset.year || i + 1) + "</span>";
      btn.addEventListener("click", function () { goTo(i); });
      timeline.appendChild(btn);
      points.push(btn);
    });
    // 轨道总宽 = 两端各留半个 viewport 的引导量 + 刻度间距累计
    var PAD = (tlView ? tlView.clientWidth : window.innerWidth) / 2;
    timeline.dataset.pad = PAD;
    timeline.style.width = (PAD * 2 + STEP * (total - 1) + 16) + "px";
    railSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    railSvg.setAttribute("class", "mh-rails");
    symbolPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
    symbolPath.setAttribute("class", "mh-rail mh-rail--symbol");
    connectionPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
    connectionPath.setAttribute("class", "mh-rail mh-rail--connection");
    railSvg.appendChild(symbolPath);
    railSvg.appendChild(connectionPath);
    timeline.appendChild(railSvg);
  }

  function heatOf(slide, line) {
    var attr = line === "symbol" ? "symbolHeat" : "connectionHeat";
    var fallback = slide.dataset.heat || "2";
    return Math.max(0, Math.min(5, parseFloat(slide.dataset[attr] || fallback)));
  }

  function heatY(heat, h) {
    var top = 12;
    var bottom = h - 22;
    return bottom - (heat / 5) * (bottom - top);
  }

  function pointX(i) {
    var PAD = parseFloat(timeline ? timeline.dataset.pad || "0" : "0");
    return PAD + i * STEP;
  }

  function pointY(slide, h) {
    var line = slide.dataset.line || "bridge";
    if (line === "symbol") return heatY(heatOf(slide, "symbol"), h);
    if (line === "connection") return heatY(heatOf(slide, "connection"), h);
    if (line === "winter") return heatY(Math.max(heatOf(slide, "symbol"), heatOf(slide, "connection")), h);
    return heatY(Math.max(heatOf(slide, "symbol"), heatOf(slide, "connection")), h);
  }

  function pathFromPoints(pts) {
    if (!pts.length) return "";
    var dstr = "M " + pts[0][0] + " " + pts[0][1];
    for (var k = 1; k < pts.length; k++) {
      var x0 = pts[k - 1][0], y0 = pts[k - 1][1], x1 = pts[k][0], y1 = pts[k][1];
      var cx = (x0 + x1) / 2;
      dstr += " C " + cx + " " + y0 + " " + cx + " " + y1 + " " + x1 + " " + y1;
    }
    return dstr;
  }

  // 依据"当前项"重排双轨: 横向仍按年份顺序, 纵向显示路线与热度。
  function layoutArc(active) {
    if (!timeline) return;
    var h = timeline.clientHeight || 90;
    var PAD = parseFloat(timeline.dataset.pad || "0");
    var symbolPts = [];
    var connectionPts = [];
    for (var i = 0; i < total; i++) {
      var x = pointX(i);
      var d = Math.abs(i - active);
      var focus = d >= SPAN ? 0 : Math.cos((d / SPAN) * (Math.PI / 2));
      var y = pointY(slides[i], h);
      points[i].style.left = x + "px";
      points[i].style.top = y + "px";
      points[i].style.opacity = String(0.34 + focus * 0.66);
      symbolPts.push([x, heatY(heatOf(slides[i], "symbol"), h)]);
      connectionPts.push([x, heatY(heatOf(slides[i], "connection"), h)]);
    }
    if (railSvg) {
      var w = PAD * 2 + STEP * (total - 1) + 16;
      railSvg.setAttribute("width", w);
      railSvg.setAttribute("height", h);
      if (symbolPath) symbolPath.setAttribute("d", pathFromPoints(symbolPts));
      if (connectionPath) connectionPath.setAttribute("d", pathFromPoints(connectionPts));
    }
  }

  // ---------- 幕定位 ----------
  function pointCenterX(i) {
    if (!points[i]) return 0;
    var left = parseFloat(points[i].style.left || "0");
    return isNaN(left) ? pointX(i) : left;
  }

  function centerTimeline(i, behavior) {
    if (tlView && points[i]) {
      var target = pointCenterX(i) - tlView.clientWidth / 2;
      timelineProgrammatic = true;
      clearTimeout(timelineProgrammaticTimer);
      tlView.scrollTo({ left: target, behavior: behavior || "smooth" });
      timelineProgrammaticTimer = setTimeout(function () {
        timelineProgrammatic = false;
      }, behavior === "auto" ? 40 : 520);
    }
  }

  function goTo(i) {
    i = Math.max(0, Math.min(total - 1, i));
    locked = true;
    setActive(i, false);
    centerTimeline(i, "auto");
    scroller.scrollTo({ left: slides[i].offsetLeft, behavior: "smooth" });
    clearTimeout(lockTimer);
    lockTimer = setTimeout(function () { locked = false; }, 700);
  }

  function setActive(i, syncTimeline) {
    if (i === currentIdx) return;
    currentIdx = i;
    var slide = slides[i];

    slides.forEach(function (s, k) { s.classList.toggle("is-active", k === i); });
    points.forEach(function (p, k) { p.classList.toggle("is-active", k === i); });

    if (document.body.dataset.theme !== (slide.dataset.theme || "intro")) {
      document.body.dataset.theme = slide.dataset.theme || "intro";
    }
    if (era) era.textContent = slide.dataset.era || "";
    if (code) code.textContent = pad(i + 1) + " / " + pad(total);

    // 重排弧形 + 把当前刻度滑到弧顶(视口中央)
    layoutArc(i);
    if (syncTimeline !== false) centerTimeline(i, "auto");
  }

  function pad(n) { return (n < 10 ? "0" : "") + n; }

  function nearestTimelineIndex() {
    if (!tlView || !points.length) return currentIdx < 0 ? 0 : currentIdx;
    var center = tlView.scrollLeft + tlView.clientWidth / 2;
    var idx = 0;
    var best = Infinity;
    for (var i = 0; i < points.length; i++) {
      var d = Math.abs(pointCenterX(i) - center);
      if (d < best) { best = d; idx = i; }
    }
    return idx;
  }

  if (tlView) {
    tlView.addEventListener("pointerdown", function (e) {
      if (e.button !== 0) return;
      timelineDragging = false;
      timelineMoved = false;
      timelinePointerId = e.pointerId;
      timelinePointer = e.clientX;
      timelineStartLeft = tlView.scrollLeft;
    });
    tlView.addEventListener("pointermove", function (e) {
      if (timelinePointerId !== e.pointerId) return;
      var dx = e.clientX - timelinePointer;
      if (!timelineDragging && Math.abs(dx) > 5) {
        timelineDragging = true;
        timelineMoved = true;
        tlView.classList.add("is-dragging");
        try { tlView.setPointerCapture(e.pointerId); } catch (_) {}
      }
      if (!timelineDragging) return;
      tlView.scrollLeft = timelineStartLeft - dx;
    });
    function endTimelineDrag(e) {
      if (timelinePointerId !== e.pointerId) return;
      timelinePointerId = null;
      if (!timelineDragging) return;
      timelineDragging = false;
      tlView.classList.remove("is-dragging");
      try { tlView.releasePointerCapture(e.pointerId); } catch (_) {}
      if (timelineMoved) {
        suppressTimelineClick = true;
        setTimeout(function () { suppressTimelineClick = false; }, 80);
      }
    }
    tlView.addEventListener("pointerup", endTimelineDrag);
    tlView.addEventListener("pointercancel", endTimelineDrag);
    tlView.addEventListener("scroll", function () {
      if (timelineProgrammatic) return;
      clearTimeout(timelineSnapTimer);
      timelineSnapTimer = setTimeout(function () {
        goTo(nearestTimelineIndex());
      }, 160);
    }, { passive: true });
    tlView.addEventListener("click", function (e) {
      if (!suppressTimelineClick) return;
      e.preventDefault();
      e.stopPropagation();
    }, true);
  }

  // ---------- 滚动 → 判定当前幕 + 进度条 ----------
  function onScroll() {
    var max = scroller.scrollWidth - scroller.clientWidth;
    if (progress) {
      var ratio = max > 0 ? scroller.scrollLeft / max : 0;
      progress.style.width = (ratio * 100).toFixed(2) + "%";
    }
    if (locked) return;
    var mid = scroller.scrollLeft + scroller.clientWidth / 2;
    var idx = 0;
    var best = Infinity;
    for (var k = 0; k < total; k++) {
      var center = slides[k].offsetLeft + slides[k].offsetWidth / 2;
      var d = Math.abs(center - mid);
      if (d < best) { best = d; idx = k; }
    }
    setActive(idx);
  }
  scroller.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", function () {
    if (currentIdx >= 0) {
      var c = currentIdx; currentIdx = -1; setActive(c);
    }
  });

  // ---------- 竖向滚轮 → 横向翻页 ----------
  scroller.addEventListener(
    "wheel",
    function (e) {
      if (Math.abs(e.deltaY) <= Math.abs(e.deltaX)) return; // 本来就是横向手势, 放行
      e.preventDefault();
      scroller.scrollLeft += e.deltaY;
    },
    { passive: false }
  );

  // ---------- 键盘 ----------
  function editable(t) {
    return t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName || ""));
  }
  document.addEventListener("keydown", function (e) {
    if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || editable(e.target)) return;
    var k = e.key;
    if (k === "ArrowRight" || k === "PageDown" || k === "l" || k === "L") { e.preventDefault(); goTo(currentIdx + 1); }
    else if (k === "ArrowLeft" || k === "PageUp" || k === "h" || k === "H") { e.preventDefault(); goTo(currentIdx - 1); }
    else if (k === "Home") { e.preventDefault(); goTo(0); }
    else if (k === "End") { e.preventDefault(); goTo(total - 1); }
  });

  if (prevBtn) prevBtn.addEventListener("click", function () { goTo(currentIdx - 1); });
  if (nextBtn) nextBtn.addEventListener("click", function () { goTo(currentIdx + 1); });

  // ---------- 图片回退 ----------
  Array.prototype.forEach.call(document.images, function (img) {
    if (!img.loading) img.loading = "lazy";
    if (!img.decoding) img.decoding = "async";
    img.addEventListener("error", function () {
      var host = img.closest(".milestone__media, .tree-node");
      if (host) host.classList.add("is-broken");
    });
    if (img.complete && img.naturalWidth === 0) {
      var host = img.closest(".milestone__media, .tree-node");
      if (host) host.classList.add("is-broken");
    }
  });

  // ---------- 初始化 ----------
  setActive(0);
  onScroll();
})();
