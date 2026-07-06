/*
 * ml-history.js — 专题页《机器学习群星史》
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

  // ---------- 底部弧形时间线刻度 ----------
  // 节点沿一段"上凸的弧"排布: 弧顶(当前项)最高, 两侧渐低渐淡。
  var STEP = 92;   // 相邻刻度水平间距(px)
  var LIFT = 26;   // 弧顶相对两端抬高的幅度(px)
  var SPAN = 6;    // 单侧参与弧形抬升的刻度数(超出压平)
  var arcSvg = null, arcPath = null;

  if (timeline) {
    slides.forEach(function (slide, i) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "mh-timepoint";
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
    arcSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    arcSvg.setAttribute("class", "mh-arc");
    arcPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
    arcPath.setAttribute("class", "mh-arc__line");
    arcSvg.appendChild(arcPath);
    timeline.appendChild(arcSvg);
  }

  // 依据"当前项"重排弧: 当前项落在弧顶, 相邻按 cos 曲线下沉
  function layoutArc(active) {
    if (!timeline) return;
    var h = timeline.clientHeight || 90;
    var PAD = parseFloat(timeline.dataset.pad || "0");
    var baseY = h - 20;          // 弧两端基线
    var pts = [];
    for (var i = 0; i < total; i++) {
      var x = PAD + i * STEP + 8;
      var d = Math.abs(i - active);
      var lift = d >= SPAN ? 0 : Math.cos((d / SPAN) * (Math.PI / 2));
      var y = baseY - lift * LIFT;
      points[i].style.left = x + "px";
      points[i].style.top = y + "px";
      pts.push([x, y]);
    }
    // 平滑弧线路径
    if (arcPath && pts.length) {
      var dstr = "M " + pts[0][0] + " " + pts[0][1];
      for (var k = 1; k < pts.length; k++) {
        var x0 = pts[k - 1][0], y0 = pts[k - 1][1], x1 = pts[k][0], y1 = pts[k][1];
        var cx = (x0 + x1) / 2;
        dstr += " C " + cx + " " + y0 + " " + cx + " " + y1 + " " + x1 + " " + y1;
      }
      arcPath.setAttribute("d", dstr);
      arcSvg.setAttribute("width", (PAD * 2 + STEP * (total - 1) + 16));
      arcSvg.setAttribute("height", h);
    }
  }

  // ---------- 幕定位 ----------
  function goTo(i) {
    i = Math.max(0, Math.min(total - 1, i));
    locked = true;
    setActive(i);
    scroller.scrollTo({ left: slides[i].offsetLeft, behavior: "smooth" });
    clearTimeout(lockTimer);
    lockTimer = setTimeout(function () { locked = false; }, 700);
  }

  function setActive(i) {
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
    if (tlView && points[i]) {
      var target = points[i].offsetLeft - tlView.clientWidth / 2;
      tlView.scrollTo({ left: target, behavior: "smooth" });
    }
  }

  function pad(n) { return (n < 10 ? "0" : "") + n; }

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
