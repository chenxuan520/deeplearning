/*
 * labs.js — 书里的可交互"实验台"。
 * 用法: 在章节里放一个占位容器, 例如 <div data-lab="neuron"></div>,
 * 本脚本会自动往里塞入完整结构并接好交互逻辑。
 * 支持的 data-lab: neuron | propagation | attention | multihead | real-attention
 * 迁移自 docs/attention-guide/script.js, 改为容器作用域 (各组件互不干扰)。
 */
(function () {
  "use strict";

  /* ===================== 共享数据 ===================== */
  var attentionTokens = ["The", "curious", "robot", "studied", "attention"];
  var tokenVectors = {
    The: [0.1, 0.2, 0.1],
    curious: [0.9, 0.3, 0.2],
    robot: [0.8, 0.9, 0.4],
    studied: [0.3, 0.8, 0.9],
    attention: [0.4, 0.7, 1.0]
  };
  var multiHeadPatterns = [
    {
      name: "Head 0 / 语义主体",
      summary: "这个头更像在找“句子里谁是当前动作的主要参与者”。它会更偏向 robot、attention 这样语义更重的词。",
      weight: [
        [0.42, 0.18, 0.14, 0.14, 0.12],
        [0.12, 0.33, 0.29, 0.15, 0.11],
        [0.08, 0.13, 0.44, 0.21, 0.14],
        [0.07, 0.09, 0.31, 0.35, 0.18],
        [0.04, 0.08, 0.25, 0.18, 0.45]
      ]
    },
    {
      name: "Head 1 / 动作关系",
      summary: "这个头更像在找“谁和谁发生了动作关系”。它会让 studied 更明显地看向 robot 和 attention。",
      weight: [
        [0.35, 0.2, 0.18, 0.14, 0.13],
        [0.1, 0.25, 0.23, 0.22, 0.2],
        [0.07, 0.11, 0.29, 0.32, 0.21],
        [0.05, 0.06, 0.34, 0.19, 0.36],
        [0.04, 0.06, 0.18, 0.28, 0.44]
      ]
    },
    {
      name: "Head 2 / 局部位置",
      summary: "这个头更像在学局部邻近关系。它通常会更关注自己和相邻 token,类似一种软性的局部窗口。",
      weight: [
        [0.65, 0.2, 0.1, 0.03, 0.02],
        [0.22, 0.4, 0.25, 0.08, 0.05],
        [0.08, 0.24, 0.36, 0.23, 0.09],
        [0.04, 0.08, 0.28, 0.38, 0.22],
        [0.03, 0.05, 0.12, 0.28, 0.52]
      ]
    }
  ];

  function dot(a, b) {
    return a.reduce(function (sum, value, idx) { return sum + value * b[idx]; }, 0);
  }
  function softmax(values) {
    var maxValue = Math.max.apply(null, values);
    var exps = values.map(function (v) { return Math.exp(v - maxValue); });
    var sum = exps.reduce(function (a, b) { return a + b; }, 0);
    return exps.map(function (v) { return v / sum; });
  }

  function loadDemoJsonWithFallback(relativePath) {
    var bases = [".", "https://chenxuan520.github.io/deeplearning"];
    var lastError = null;
    function tryIndex(index) {
      if (index >= bases.length) {
        return Promise.reject(lastError || new Error("demo asset unavailable"));
      }
      var base = bases[index];
      var url = (base === "." ? "" : base) + "/" + relativePath;
      return fetch(url + "?cb=" + Date.now(), { cache: "no-store" })
        .then(function (r) {
          if (!r.ok) {
            throw new Error("HTTP " + r.status);
          }
          var type = r.headers.get("content-type") || "";
          if (type.indexOf("application/json") === -1) {
            throw new Error("non-json response");
          }
          return r.json();
        })
        .catch(function (err) {
          lastError = err;
          return tryIndex(index + 1);
        });
    }
    return tryIndex(0);
  }

  /* ===================== 1. 神经元实验台 ===================== */
  var NEURON_TPL =
    '<div class="lab">' +
    '  <div class="lab__controls">' +
    '    <label>输入 x1<input type="range" min="-2" max="2" step="0.1" value="1.0" data-neuron="x1" /><span class="lab__value" data-neuron-read="x1"></span></label>' +
    '    <label>输入 x2<input type="range" min="-2" max="2" step="0.1" value="0.5" data-neuron="x2" /><span class="lab__value" data-neuron-read="x2"></span></label>' +
    '    <label>权重 w1<input type="range" min="-3" max="3" step="0.1" value="1.2" data-neuron="w1" /><span class="lab__value" data-neuron-read="w1"></span></label>' +
    '    <label>权重 w2<input type="range" min="-3" max="3" step="0.1" value="-0.8" data-neuron="w2" /><span class="lab__value" data-neuron-read="w2"></span></label>' +
    '    <label>偏置 b<input type="range" min="-3" max="3" step="0.1" value="0.4" data-neuron="b" /><span class="lab__value" data-neuron-read="b"></span></label>' +
    '    <label>激活函数' +
    '      <select data-neuron="activation">' +
    '        <option value="sigmoid">Sigmoid</option>' +
    '        <option value="relu">ReLU</option>' +
    '        <option value="tanh">Tanh</option>' +
    '        <option value="leaky_relu">LeakyReLU (slope 0.01)</option>' +
    '        <option value="gelu">GELU (Transformer 常用)</option>' +
    '      </select>' +
    '    </label>' +
    '  </div>' +
    '  <div class="lab__viz">' +
    '    <div class="formula-card">' +
    '      <h3>当前计算</h3>' +
    '      <p class="formula" data-neuron-formula></p>' +
    '      <div class="meter"><div class="meter__bar" data-neuron-bar></div></div>' +
    '      <p class="explain" data-neuron-explain></p>' +
    '    </div>' +
    '  </div>' +
    '</div>';

  function initNeuron(root) {
    root.innerHTML = NEURON_TPL;
    var state = { x1: 1, x2: 0.5, w1: 1.2, w2: -0.8, b: 0.4, activation: "sigmoid" };

    function activate(value, name) {
      if (name === "relu") return Math.max(0, value);
      if (name === "tanh") return Math.tanh(value);
      if (name === "leaky_relu") return value > 0 ? value : 0.01 * value;
      if (name === "gelu") {
        var sign = value < 0 ? -1 : 1;
        var ax = Math.abs(value / Math.SQRT2);
        var t = 1 / (1 + 0.3275911 * ax);
        var erfApprox =
          sign *
          (1 -
            ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
              0.254829592) * t * Math.exp(-ax * ax));
        return 0.5 * value * (1 + erfApprox);
      }
      return 1 / (1 + Math.exp(-value));
    }

    function render() {
      Object.keys(state).forEach(function (key) {
        var out = root.querySelector('[data-neuron-read="' + key + '"]');
        if (out) out.textContent = typeof state[key] === "number" ? state[key].toFixed(2) : state[key];
      });
      var sum = state.x1 * state.w1 + state.x2 * state.w2 + state.b;
      var out = activate(sum, state.activation);
      var formula =
        state.x1.toFixed(2) + " × " + state.w1.toFixed(2) + " + " +
        state.x2.toFixed(2) + " × " + state.w2.toFixed(2) + " + " +
        state.b.toFixed(2) + " = " + sum.toFixed(3) + " → " + state.activation + " → " + out.toFixed(3);
      root.querySelector("[data-neuron-formula]").textContent = formula;
      root.querySelector("[data-neuron-bar]").style.width =
        Math.max(6, Math.min(100, ((out + 1) / 2) * 100)) + "%";
      root.querySelector("[data-neuron-explain]").textContent =
        "先把输入按权重放大或缩小,再加上偏置;非线性激活函数让网络不只是做一层线性变换。现在输出 " +
        out.toFixed(3) + ",这就是这个神经元对当前输入的“判断结果”。";
    }

    root.querySelectorAll("[data-neuron]").forEach(function (control) {
      control.addEventListener("input", function () {
        var key = control.dataset.neuron;
        state[key] = control.tagName === "SELECT" ? control.value : Number(control.value);
        render();
      });
    });
    render();
  }

  /* ===================== 2. 前向 / 反向传播动画 ===================== */
  var PROP_TPL =
    '<div class="propagation">' +
    '  <div class="propagation__controls">' +
    '    <button class="button button--primary" type="button" data-prop-action="set-forward">切到前向传播</button>' +
    '    <button class="button button--ghost" type="button" data-prop-action="set-backward">切到反向传播</button>' +
    '    <button class="button button--ghost" type="button" data-prop-action="prev">上一步</button>' +
    '    <button class="button button--ghost" type="button" data-prop-action="next">下一步</button>' +
    '    <button class="button button--ghost" type="button" data-prop-action="reset">重置</button>' +
    '    <label class="propagation__lr">Learning Rate' +
    '      <input type="range" min="0.02" max="1.00" step="0.01" value="0.20" data-prop-control="lr" />' +
    '      <span class="lab__value" data-prop-read="lr"></span>' +
    '    </label>' +
    '    <div class="propagation__status" data-prop-status>当前模式:前向传播 / Step 0</div>' +
    '  </div>' +
    '  <div class="propagation__stage">' +
    '    <div class="propagation__left">' +
    '      <div class="network" data-network-stage>' +
    '        <svg class="network__svg" aria-hidden="true">' +
    '          <defs>' +
    '            <marker id="prop-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">' +
    '              <path d="M0,0 L8,4 L0,8 Z" fill="rgba(225,233,255,0.9)"></path>' +
    '            </marker>' +
    '            <marker id="prop-arrow-forward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">' +
    '              <path d="M0,0 L8,4 L0,8 Z" fill="#8ef0d1"></path>' +
    '            </marker>' +
    '            <marker id="prop-arrow-backward" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">' +
    '              <path d="M0,0 L8,4 L0,8 Z" fill="#ff9a8f"></path>' +
    '            </marker>' +
    '          </defs>' +
    '          <g data-network-edges></g>' +
    '        </svg>' +
    '        <div class="network__layers">' +
    '          <div class="network__layer">' +
    '            <div class="network__title">输入层</div>' +
    '            <div class="network__node-wrap"><div class="network__node" data-node="input-0">x1</div><div class="network__meta">input0 = 1.00</div></div>' +
    '            <div class="network__node-wrap"><div class="network__node" data-node="input-1">x2</div><div class="network__meta">input1 = 0.50</div></div>' +
    '          </div>' +
    '          <div class="network__layer">' +
    '            <div class="network__title">隐藏层</div>' +
    '            <div class="network__node-wrap"><div class="network__node" data-node="hidden-0">h1</div><div class="network__meta">b10 = 0.10</div></div>' +
    '            <div class="network__node-wrap"><div class="network__node" data-node="hidden-1">h2</div><div class="network__meta">b11 = -0.05</div></div>' +
    '            <div class="network__node-wrap"><div class="network__node" data-node="hidden-2">h3</div><div class="network__meta">b12 = 0.08</div></div>' +
    '          </div>' +
    '          <div class="network__layer">' +
    '            <div class="network__title">输出层</div>' +
    '            <div class="network__node-wrap"><div class="network__node" data-node="output-0">y</div><div class="network__meta">b20 = 0.20</div></div>' +
    '          </div>' +
    '        </div>' +
    '        <div class="network__pulse" data-network-pulse></div>' +
    '      </div>' +
    '      <div class="propagation-formula">' +
    '        <div class="formula-card">' +
    '          <h3>当前公式</h3>' +
    '          <p class="formula" data-prop-formula-main>z = Wx + b</p>' +
    '          <p class="formula formula--sub" data-prop-formula-sub>a = σ(z)</p>' +
    '          <p class="explain" data-prop-formula-explain>前向传播时,先做线性变换,再过激活函数;反向传播时切到梯度链式法则。</p>' +
    '        </div>' +
    '        <div class="formula-card">' +
    '          <h3>链式法则拆解</h3>' +
    '          <p class="formula" data-prop-chain-main>dL/dW = dL/dy · dy/dz · dz/dW</p>' +
    '          <p class="formula formula--sub" data-prop-chain-sub>还没进入反向传播时,这里显示“当前没有梯度拆解”。</p>' +
    '          <p class="explain" data-prop-chain-explain>链式法则把“总误差对某个参数的影响”拆成一段段局部影响再连乘起来。</p>' +
    '        </div>' +
    '      </div>' +
    '      <div class="propagation-flow">' +
    '        <div class="flow-box"><h4>数值流动</h4><p class="flow-box__line" data-prop-values>x = [1.00, 0.50] → h = [0.42, 0.73, 0.18] → y = [0.81]</p></div>' +
    '        <div class="flow-box"><h4>梯度流动</h4><p class="flow-box__line" data-prop-grads>dL/dy = [-0.19] → dL/dh = [-0.08, 0.03, 0.11] → dL/dx = [0.04, -0.02]</p></div>' +
    '      </div>' +
    '    </div>' +
    '    <div class="propagation__notes">' +
    '      <div class="formula-card">' +
    '        <h3>当前阶段</h3>' +
    '        <p class="formula" data-prop-title>还没有播放动画</p>' +
    '        <p class="explain" data-prop-explain>点击上面的按钮,观察“数值”如何向前流动,以及“误差信号”如何向后流动。</p>' +
    '      </div>' +
    '      <div class="propagation-change">' +
    '        <h3>参数如何变化</h3>' +
    '        <p class="explain" data-prop-lr-note>学习率越大,每一步改参数的幅度越大;太小学得慢,太大又可能来回震荡。</p>' +
    '        <div class="change-list" data-prop-params></div>' +
    '      </div>' +
    '    </div>' +
    '  </div>' +
    '</div>';

  function initPropagation(root) {
    root.innerHTML = PROP_TPL;
    var $ = function (s) { return root.querySelector(s); };
    var $$ = function (s) { return root.querySelectorAll(s); };
    var SVG_NS = "http://www.w3.org/2000/svg";
    var NETWORK_EDGES = [
      { id: "w100", from: "input-0", to: "hidden-0", label: "w100" },
      { id: "w101", from: "input-0", to: "hidden-1", label: "w101" },
      { id: "w111", from: "input-1", to: "hidden-1", label: "w111" },
      { id: "w112", from: "input-1", to: "hidden-2", label: "w112" },
      { id: "w200", from: "hidden-0", to: "output-0", label: "w200" },
      { id: "w210", from: "hidden-1", to: "output-0", label: "w210" },
      { id: "w220", from: "hidden-2", to: "output-0", label: "w220" }
    ];
    var layoutTimer = null;

    function buildNetworkEdges() {
      var edgesGroup = $("[data-network-edges]");
      if (!edgesGroup) return;
      edgesGroup.innerHTML = "";
      NETWORK_EDGES.forEach(function (def) {
        var g = document.createElementNS(SVG_NS, "g");
        g.setAttribute("class", "network__edge");
        g.setAttribute("data-edge", def.id);
        var line = document.createElementNS(SVG_NS, "line");
        line.setAttribute("class", "network__edge-line");
        line.setAttribute("marker-end", "url(#prop-arrow)");
        var label = document.createElementNS(SVG_NS, "text");
        label.setAttribute("class", "network__edge-label");
        label.textContent = def.label;
        g.appendChild(line);
        g.appendChild(label);
        edgesGroup.appendChild(g);
      });
    }

    function layoutNetworkEdges() {
      var stage = $("[data-network-stage]");
      var svg = stage && stage.querySelector(".network__svg");
      var edgesGroup = $("[data-network-edges]");
      if (!stage || !svg || !edgesGroup) return;
      var width = stage.clientWidth;
      var height = stage.clientHeight;
      if (width < 1 || height < 1) return;
      svg.setAttribute("viewBox", "0 0 " + width + " " + height);
      svg.setAttribute("width", String(width));
      svg.setAttribute("height", String(height));

      function center(nodeId) {
        var node = stage.querySelector('[data-node="' + nodeId + '"]');
        if (!node) return null;
        var stageRect = stage.getBoundingClientRect();
        var nodeRect = node.getBoundingClientRect();
        return {
          x: nodeRect.left - stageRect.left + nodeRect.width / 2,
          y: nodeRect.top - stageRect.top + nodeRect.height / 2,
          r: nodeRect.width / 2
        };
      }

      NETWORK_EDGES.forEach(function (def) {
        var group = edgesGroup.querySelector('[data-edge="' + def.id + '"]');
        var from = center(def.from);
        var to = center(def.to);
        if (!group || !from || !to) return;
        var dx = to.x - from.x;
        var dy = to.y - from.y;
        var len = Math.hypot(dx, dy) || 1;
        var ux = dx / len;
        var uy = dy / len;
        var x1 = from.x + ux * from.r;
        var y1 = from.y + uy * from.r;
        var x2 = to.x - ux * to.r;
        var y2 = to.y - uy * to.r;
        var mx = (x1 + x2) / 2 + uy * 14;
        var my = (y1 + y2) / 2 - ux * 14;
        var line = group.querySelector(".network__edge-line");
        var label = group.querySelector(".network__edge-label");
        if (line) {
          line.setAttribute("x1", String(x1));
          line.setAttribute("y1", String(y1));
          line.setAttribute("x2", String(x2));
          line.setAttribute("y2", String(y2));
        }
        if (label) {
          label.setAttribute("x", String(mx));
          label.setAttribute("y", String(my));
          label.textContent = def.label;
        }
      });
    }

    function scheduleLayout() {
      if (layoutTimer) clearTimeout(layoutTimer);
      layoutTimer = setTimeout(function () {
        layoutTimer = null;
        requestAnimationFrame(layoutNetworkEdges);
      }, 16);
    }

    buildNetworkEdges();
    scheduleLayout();
    window.addEventListener("resize", scheduleLayout);

    var state = { learningRate: 0.2 };
    var paramDefs = {
      hidden0: { label: "w(hidden,0)", old: 0.8, grad: 0.2 },
      hidden1: { label: "w(hidden,1)", old: -0.2, grad: -0.1 },
      out0: { label: "w(out,0)", old: 1.1, grad: 0.2 },
      bout: { label: "b(out)", old: 0.2, grad: -0.2 }
    };
    function formatUpdate(oldValue, grad, lr) {
      var nextValue = oldValue - lr * grad;
      return oldValue.toFixed(2) + " → " + nextValue.toFixed(2) + " (grad " + (grad >= 0 ? "+" : "") + grad.toFixed(2) + ")";
    }
    function link(text, targets) {
      return '<span class="formula-link" data-link-targets="' + targets.join(",") + '">' + text + "</span>";
    }

    var steps = {
      forward: [
        {
          node: ["input-0", "input-1"], secondaryNode: ["hidden-0", "hidden-1", "hidden-2"],
          edge: ["w100", "w101", "w111", "w112"], title: "前向传播:输入层接收原始特征",
          explain: "这一步传的不是误差,而是原始输入值。图片像素、表格特征或 token embedding 都先进入输入层。",
          pulse: { x: 12, y: 55 },
          formulaMain: link("x", ["input-0", "input-1"]) + " = [x1, x2]",
          formulaSub: "输入层本身通常不做复杂运算,它负责把原始特征送进网络。",
          formulaExplain: "把输入层想成“数据入口”。真正的参数化变换通常从下一层开始。",
          chainMain: "当前还没有进入链式法则", chainSub: "前向传播关注数值怎么往前算,不是梯度怎么往回传。",
          chainExplain: "所以这里先看输入值和参数如何组合成下一层的净输入。",
          values: "x = [1.00, 0.50] → h = [?, ?, ?] → y = [?]",
          grads: "梯度还没有开始传播,当前阶段重点是把输入送进网络。",
          params: function () { return [["w(hidden,0)", "0.80 (等待更新)"], ["w(out,0)", "1.10 (等待更新)"]]; }
        },
        {
          node: ["hidden-0", "hidden-1", "hidden-2"], secondaryNode: ["input-0", "input-1", "output-0"],
          edge: ["w100", "w101", "w111", "w112", "w200", "w210", "w220"], title: "前向传播:隐藏层做加权求和和激活",
          explain: "隐藏层把上一层输出乘权重、加偏置,再过激活函数,逐步形成更抽象的特征表示。",
          pulse: { x: 48, y: 48 },
          formulaMain: link("z", ["hidden-0", "hidden-1", "hidden-2"]) + " = " + link("W", ["w100", "w101", "w111", "w112"]) + " " + link("x", ["input-0", "input-1"]) + " + " + link("b", ["hidden-0", "hidden-1", "hidden-2"]),
          formulaSub: link("a", ["hidden-0", "hidden-1", "hidden-2"]) + " = σ(" + link("z", ["hidden-0", "hidden-1", "hidden-2"]) + ")",
          formulaExplain: "最经典的一层神经元公式:先线性变换,再经过激活函数,得到新的中间表示。",
          chainMain: "z_j = Σ_i " + link("w_ji", ["w100", "w101", "w111", "w112"]) + " " + link("x_i", ["input-0", "input-1"]) + " + " + link("b_j", ["hidden-0", "hidden-1", "hidden-2"]),
          chainSub: link("a_j", ["hidden-0", "hidden-1", "hidden-2"]) + " = σ(" + link("z_j", ["hidden-0", "hidden-1", "hidden-2"]) + ")",
          chainExplain: "这一层每个节点都在重复同一个模式:收输入、乘权重、加偏置、过激活。",
          values: "x = [1.00, 0.50] → h = [0.42, 0.73, 0.18] → y = [?]",
          grads: "这里仍然是数值流,不是误差信号。隐藏层的输出会作为下一层输入。",
          params: function () { return [["z(hidden,0)", "1.00×0.80 + 0.50×-0.20 + 0.10 → 0.80"], ["a(hidden,0)", "sigmoid(0.80) → 0.69"]]; }
        },
        {
          node: ["output-0"], secondaryNode: ["hidden-0", "hidden-1", "hidden-2"],
          edge: ["w200", "w210", "w220"], title: "前向传播:输出层给出预测",
          explain: "最后一层把隐藏特征压缩成预测。分类里可能是各类别得分,语言模型里则是下一个 token 的 logits。",
          pulse: { x: 84, y: 52 },
          formulaMain: link("y", ["output-0"]) + " = " + link("W_out", ["w200", "w210", "w220"]) + " " + link("h", ["hidden-0", "hidden-1", "hidden-2"]) + " + " + link("b_out", ["output-0"]),
          formulaSub: link("loss", ["output-0"]) + " = L(" + link("y", ["output-0"]) + ", target)",
          formulaExplain: "输出层先得到预测,再和真实目标比较,才能知道模型错了多少。",
          chainMain: "L = ½ (target − " + link("y", ["output-0"]) + ")²", chainSub: "先有预测,再谈误差",
          chainExplain: "你只有先得到输出,才能把预测和目标作比较,形成 loss。",
          values: "x = [1.00, 0.50] → h = [0.42, 0.73, 0.18] → y = [0.81]",
          grads: "前向阶段结束,模型得到一个输出。接下来用 loss 判断它偏了多少。",
          params: function () { return [["y_pred", "[0.81]"], ["loss 前状态", "预测已得到,等待与目标比较"]]; }
        }
      ],
      backward: [
        {
          node: ["output-0"], secondaryNode: ["hidden-0", "hidden-1", "hidden-2"],
          edge: ["w200", "w210", "w220"], title: "反向传播:先从输出误差开始",
          explain: "反向传播不是把原始输入往回传,而是把“模型错了多少”变成梯度,从输出层往回走。",
          pulse: { x: 84, y: 52 },
          formulaMain: link("dL/dy", ["output-0"]) + " = ∂L / ∂y", formulaSub: "输出层先知道自己错了多少",
          formulaExplain: "反向传播的起点不是输入,而是损失函数。只有先知道损失,才知道往哪个方向改。",
          chainMain: link("dL/dw200", ["w200"]) + " = " + link("dL/dy", ["output-0"]) + " · " + link("dy/dnet", ["output-0"]) + " · " + link("dnet/dw200", ["w200", "hidden-0"]),
          chainSub: "= δ_out · " + link("h0", ["hidden-0"]),
          chainExplain: "链式法则第一次展开:总误差对输出层权重的影响被拆成三段局部影响相乘。",
          values: "预测 y = [0.81],真实目标 t = [1.00]",
          grads: "dL/dy = y − t = −0.19 → 输出偏低,相关权重要朝增大输出的方向更新。",
          params: function (lr) { return [[paramDefs.out0.label, formatUpdate(paramDefs.out0.old, paramDefs.out0.grad, lr)], [paramDefs.bout.label, formatUpdate(paramDefs.bout.old, paramDefs.bout.grad, lr)]]; }
        },
        {
          node: ["hidden-0", "hidden-1", "hidden-2"], secondaryNode: ["input-0", "input-1", "output-0"],
          edge: ["w100", "w101", "w111", "w112", "w200", "w210", "w220"], title: "反向传播:隐藏层接收梯度信号",
          explain: "每个隐藏节点会收到来自后面层的梯度,知道自己对最终错误贡献了多少,再继续往前传。",
          pulse: { x: 48, y: 48 },
          formulaMain: link("δ_l", ["hidden-0", "hidden-1", "hidden-2"]) + " = (" + link("Wᵀ", ["w200", "w210", "w220"]) + " " + link("δ_(l+1)", ["output-0"]) + ") ⊙ σ'(" + link("z_l", ["hidden-0", "hidden-1", "hidden-2"]) + ")",
          formulaSub: "链式法则把后层误差信号传回当前层",
          formulaExplain: "反向传播最核心的精神:后一层的误差,经过当前层的局部导数修正后,继续往前流。",
          chainMain: link("dL/dw100", ["w100"]) + " = " + link("dL/dout", ["output-0"]) + " · " + link("dout/dnet10", ["hidden-0"]) + " · " + link("dnet10/dw100", ["w100", "input-0"]),
          chainSub: "= " + link("δ_10", ["hidden-0"]) + " · " + link("x0", ["input-0"]),
          chainExplain: "对隐藏层权重,梯度不再只看输出误差,而要先把输出误差一路传回当前隐藏节点。",
          values: "隐藏表示 h = [0.42, 0.73, 0.18] 不变,但它们现在各自收到了梯度。",
          grads: "dL/dh = [-0.08, 0.03, 0.11] → 不同隐藏节点对最终误差的责任大小不一样。",
          params: function (lr) { return [[paramDefs.hidden0.label, formatUpdate(paramDefs.hidden0.old, paramDefs.hidden0.grad, lr)], [paramDefs.hidden1.label, formatUpdate(paramDefs.hidden1.old, paramDefs.hidden1.grad, lr)]]; }
        },
        {
          node: ["input-0", "input-1"], secondaryNode: ["hidden-0", "hidden-1", "hidden-2"],
          edge: ["w100", "w101", "w111", "w112"], title: "反向传播:更早层据此调整参数",
          explain: "真正更新的是层间权重和偏置。越靠前的层,需要等梯度一路传回来,才能知道自己该怎么改。",
          pulse: { x: 12, y: 55 },
          formulaMain: link("W_new", ["w100", "w101", "w111", "w112", "w200", "w210", "w220"]) + " = " + link("W_old", ["w100", "w101", "w111", "w112", "w200", "w210", "w220"]) + " − " + link("η", []) + " · " + link("dL/dW", ["w100", "w101", "w111", "w112", "w200", "w210", "w220"]),
          formulaSub: link("b_new", ["hidden-0", "hidden-1", "hidden-2", "output-0"]) + " = " + link("b_old", ["hidden-0", "hidden-1", "hidden-2", "output-0"]) + " − " + link("η", []) + " · " + link("dL/db", ["hidden-0", "hidden-1", "hidden-2", "output-0"]),
          formulaExplain: "最后落到参数更新规则:η 就是 learning rate,它直接决定一次更新走多远。",
          chainMain: "η 越大,W_new 变化越大", chainSub: "η 越小,训练更稳但更慢",
          chainExplain: "这也是为什么学习率是训练里最关键的超参数之一:它直接决定你每一步迈多大。",
          values: "输入本身通常不更新,但与输入相连的早期权重会更新。",
          grads: "dL/dx = [0.04, -0.02] → 不是为了改输入,而是继续把责任链条往更早层推。",
          params: function (lr) { return [["w(input→hidden)", "学习率 = " + lr.toFixed(2) + ",每次更新幅度 = lr × gradient"], ["更新规则", "new_w = old_w − learning_rate × gradient"]]; }
        }
      ]
    };

    var refs = {
      title: $("[data-prop-title]"), explain: $("[data-prop-explain]"), values: $("[data-prop-values]"),
      grads: $("[data-prop-grads]"), params: $("[data-prop-params]"),
      fMain: $("[data-prop-formula-main]"), fSub: $("[data-prop-formula-sub]"), fExp: $("[data-prop-formula-explain]"),
      cMain: $("[data-prop-chain-main]"), cSub: $("[data-prop-chain-sub]"), cExp: $("[data-prop-chain-explain]"),
      lrRead: $('[data-prop-read="lr"]'), lrNote: $("[data-prop-lr-note]"), status: $("[data-prop-status]"),
      pulse: $("[data-network-pulse]"),
      modeForwardBtn: $('[data-prop-action="set-forward"]'), modeBackwardBtn: $('[data-prop-action="set-backward"]')
    };
    var timer = null, edgeTimers = [], mode = "forward", stepIndex = -1;

    function clearFormulaLinks() {
      $$(".formula-link").forEach(function (n) { n.classList.remove("is-hovered"); });
      $$("[data-node]").forEach(function (n) { n.classList.remove("is-linked"); });
      $$("[data-edge]").forEach(function (n) { n.classList.remove("is-linked"); });
    }
    function bindFormulaLinks() {
      clearFormulaLinks();
      $$(".formula-link").forEach(function (linkNode) {
        var apply = function (active) {
          var targets = (linkNode.dataset.linkTargets || "").split(",").map(function (s) { return s.trim(); }).filter(Boolean);
          linkNode.classList.toggle("is-hovered", active);
          targets.forEach(function (target) {
            var node = root.querySelector('[data-node="' + target + '"]');
            if (node) node.classList.toggle("is-linked", active);
            var edge = root.querySelector('[data-edge="' + target + '"]');
            if (edge) edge.classList.toggle("is-linked", active);
          });
        };
        linkNode.onmouseenter = function () { apply(true); };
        linkNode.onmouseleave = function () { apply(false); };
      });
    }
    function clearEdgePulses() {
      edgeTimers.forEach(function (t) { clearTimeout(t); });
      edgeTimers = [];
      $$("[data-edge]").forEach(function (e) { e.classList.remove("is-pulse"); });
    }
    function pulseEdges(edgeNames) {
      clearEdgePulses();
      edgeNames.forEach(function (name, index) {
        var startTimer = setTimeout(function () {
          var edge = root.querySelector('[data-edge="' + name + '"]');
          if (!edge) return;
          edge.classList.remove("is-pulse");
          void edge.offsetWidth;
          edge.classList.add("is-pulse");
          var endTimer = setTimeout(function () { edge.classList.remove("is-pulse"); }, 720);
          edgeTimers.push(endTimer);
        }, index * 180);
        edgeTimers.push(startTimer);
      });
    }
    function renderStatic() {
      refs.lrRead.textContent = state.learningRate.toFixed(2);
      refs.lrNote.textContent = "当前 learning rate = " + state.learningRate.toFixed(2) + "。它决定每一步参数改多少:越大改得越猛,越小改得越保守。";
    }
    function updateModeButtons() {
      if (refs.modeForwardBtn) {
        refs.modeForwardBtn.classList.toggle("button--primary", mode === "forward");
        refs.modeForwardBtn.classList.toggle("button--ghost", mode !== "forward");
      }
      if (refs.modeBackwardBtn) {
        refs.modeBackwardBtn.classList.toggle("button--primary", mode === "backward");
        refs.modeBackwardBtn.classList.toggle("button--ghost", mode !== "backward");
      }
    }
    function clearActive() {
      $$("[data-node]").forEach(function (n) { n.classList.remove("is-active-forward", "is-active-backward", "is-secondary-forward", "is-secondary-backward"); });
      $$("[data-edge]").forEach(function (e) { e.classList.remove("is-active-forward", "is-active-backward"); });
    }
    function reset() {
      if (timer) { clearTimeout(timer); timer = null; }
      clearEdgePulses();
      clearActive();
      refs.pulse.classList.remove("is-visible", "is-backward");
      refs.title.textContent = "还没有播放动画";
      refs.explain.textContent = "点击上面的按钮,观察“数值”如何向前流动,以及“误差信号”如何向后流动。";
      refs.fMain.innerHTML = "z = W x + b"; refs.fSub.innerHTML = "a = σ(z)";
      refs.fExp.textContent = "前向传播时,先做线性变换,再过激活函数;反向传播时切到梯度链式法则。";
      refs.cMain.innerHTML = "dL/dW = dL/dy · dy/dz · dz/dW";
      refs.cSub.innerHTML = "还没进入反向传播时,这里显示“当前没有梯度拆解”。";
      refs.cExp.textContent = "链式法则把“总误差对某个参数的影响”拆成一段段局部影响再连乘起来。";
      refs.values.textContent = "x = [1.00, 0.50] → h = [0.42, 0.73, 0.18] → y = [0.81]";
      refs.grads.textContent = "dL/dy = [-0.19] → dL/dh = [-0.08, 0.03, 0.11] → dL/dx = [0.04, -0.02]";
      refs.params.innerHTML =
        '<div class="change-item"><strong>w(hidden,0)</strong><span>0.80 → 0.76</span></div>' +
        '<div class="change-item"><strong>w(out,0)</strong><span>1.10 → 1.06</span></div>';
      refs.status.textContent = "当前模式:" + (mode === "forward" ? "前向传播" : "反向传播") + " / Step " + Math.max(0, stepIndex + 1);
      renderStatic();
      updateModeButtons();
      bindFormulaLinks();
    }
    function show() {
      var list = steps[mode];
      refs.status.textContent = "当前模式:" + (mode === "forward" ? "前向传播" : "反向传播") + " / Step " + Math.max(0, stepIndex + 1);
      if (stepIndex < 0 || stepIndex >= list.length) { clearEdgePulses(); clearActive(); refs.pulse.classList.remove("is-visible", "is-backward"); bindFormulaLinks(); return; }
      var step = list[stepIndex];
      var activeClass = mode === "forward" ? "is-active-forward" : "is-active-backward";
      clearEdgePulses(); clearActive();
      step.node.forEach(function (name) { var n = root.querySelector('[data-node="' + name + '"]'); if (n) n.classList.add(activeClass); });
      step.secondaryNode.forEach(function (name) { var n = root.querySelector('[data-node="' + name + '"]'); if (n) n.classList.add(mode === "forward" ? "is-secondary-forward" : "is-secondary-backward"); });
      step.edge.forEach(function (name) { var e = root.querySelector('[data-edge="' + name + '"]'); if (e) e.classList.add(activeClass); });
      pulseEdges(step.edge);
      refs.pulse.classList.add("is-visible");
      refs.pulse.classList.toggle("is-backward", mode === "backward");
      refs.pulse.style.transform = "translate(" + step.pulse.x + "%, " + step.pulse.y + "%)";
      scheduleLayout();
      refs.title.textContent = step.title; refs.explain.textContent = step.explain;
      refs.fMain.innerHTML = step.formulaMain; refs.fSub.innerHTML = step.formulaSub; refs.fExp.textContent = step.formulaExplain;
      refs.cMain.innerHTML = step.chainMain; refs.cSub.innerHTML = step.chainSub; refs.cExp.textContent = step.chainExplain;
      refs.values.textContent = step.values; refs.grads.textContent = step.grads;
      refs.params.innerHTML = step.params(state.learningRate).map(function (pair) { return '<div class="change-item"><strong>' + pair[0] + "</strong><span>" + pair[1] + "</span></div>"; }).join("");
      bindFormulaLinks();
    }

    $$("[data-prop-action]").forEach(function (button) {
      button.addEventListener("click", function () {
        var action = button.dataset.propAction;
        if (action === "reset") { stepIndex = -1; reset(); return; }
        if (action === "set-forward") { mode = "forward"; stepIndex = -1; reset(); return; }
        if (action === "set-backward") { mode = "backward"; stepIndex = -1; reset(); return; }
        var list = steps[mode];
        if (action === "next") { stepIndex = Math.min(list.length - 1, stepIndex + 1); show(); return; }
        if (action === "prev") { stepIndex = Math.max(-1, stepIndex - 1); if (stepIndex === -1) reset(); else show(); }
      });
    });
    $('[data-prop-control="lr"]').addEventListener("input", function (e) { state.learningRate = Number(e.target.value); renderStatic(); });
    reset();
  }

  /* ===================== 3. Attention 实验台 ===================== */
  var ATT_TPL =
    '<div class="attention-lab">' +
    '  <div class="attention-lab__toolbar">' +
    '    <div class="token-picker" data-token-picker></div>' +
    '    <label>Temperature<input type="range" min="0.4" max="2.0" step="0.1" value="1.0" data-attention="temperature" /><span class="lab__value" data-attention-read="temperature"></span></label>' +
    '    <label class="checkbox"><input type="checkbox" data-attention="causal" />打开 causal mask(只能看自己和前文)</label>' +
    '  </div>' +
    '  <div class="attention-lab__body">' +
    '    <div class="sentence-strip" data-sentence-strip></div>' +
    '    <div class="attention-formula formula-card">' +
    '      <h3>Attention 公式映射</h3>' +
    '      <p class="formula" data-attention-formula-main>score = Q · K / √d</p>' +
    '      <p class="formula formula--sub" data-attention-formula-sub>weight = softmax(score), output = Σ(weight × V)</p>' +
    '      <p class="explain">hover 公式里的关键项,观察下面 token / score / softmax 条 / 加权结果如何联动高亮。</p>' +
    '    </div>' +
    '    <div class="attention-grid">' +
    '      <div class="attention-panel"><h3>打分矩阵(Score)</h3><div class="matrix" data-score-matrix></div></div>' +
    '      <div class="attention-panel"><h3>归一化后权重(Softmax)</h3><div class="bars" data-softmax-bars></div></div>' +
    '      <div class="attention-panel"><h3>加权结果(Weighted Sum)</h3><div class="vector-box" data-attention-result></div><p class="explain" data-attention-explain></p></div>' +
    '    </div>' +
    '  </div>' +
    '</div>';

  function initAttention(root) {
    root.innerHTML = ATT_TPL;
    var $ = function (s) { return root.querySelector(s); };
    var $$ = function (s) { return root.querySelectorAll(s); };
    var state = { queryIndex: 4, temperature: 1, causal: false };

    var tokenPicker = $("[data-token-picker]"), sentenceStrip = $("[data-sentence-strip]");
    var scoreMatrix = $("[data-score-matrix]"), softmaxBars = $("[data-softmax-bars]");
    var attResult = $("[data-attention-result]"), attExplain = $("[data-attention-explain]");
    var fMain = $("[data-attention-formula-main]"), fSub = $("[data-attention-formula-sub]");

    function linkA(text, targets) { return '<span class="formula-link" data-att-link-targets="' + targets.join(",") + '">' + text + "</span>"; }
    function clearLinks() {
      $$(".formula-link[data-att-link-targets]").forEach(function (n) { n.classList.remove("is-hovered"); });
      $$(".token-chip, .matrix__cell, .bars__item, .vector-box").forEach(function (n) { n.classList.remove("is-linked"); });
    }
    function bindLinks() {
      clearLinks();
      $$(".formula-link[data-att-link-targets]").forEach(function (node) {
        var apply = function (active) {
          var targets = (node.dataset.attLinkTargets || "").split(",").map(function (s) { return s.trim(); }).filter(Boolean);
          node.classList.toggle("is-hovered", active);
          targets.forEach(function (target) { var elx = root.querySelector('[data-att-link="' + target + '"]'); if (elx) elx.classList.toggle("is-linked", active); });
        };
        node.onmouseenter = function () { apply(true); };
        node.onmouseleave = function () { apply(false); };
      });
    }
    function render() {
      tokenPicker.innerHTML = ""; sentenceStrip.innerHTML = ""; scoreMatrix.innerHTML = ""; softmaxBars.innerHTML = "";
      attentionTokens.forEach(function (token, index) {
        var button = document.createElement("button");
        button.type = "button"; button.textContent = index + ": " + token;
        button.className = index === state.queryIndex ? "is-active" : "";
        button.addEventListener("click", function () { state.queryIndex = index; render(); });
        tokenPicker.appendChild(button);
        var chip = document.createElement("div");
        chip.className = "token-chip " + (index === state.queryIndex ? "is-focus" : "");
        chip.textContent = token; chip.dataset.attLink = "token-" + index;
        sentenceStrip.appendChild(chip);
      });
      var queryToken = attentionTokens[state.queryIndex];
      var queryVector = tokenVectors[queryToken];
      var scores = attentionTokens.map(function (token, keyIndex) {
        if (state.causal && keyIndex > state.queryIndex) return -1e9;
        return dot(queryVector, tokenVectors[token]) / state.temperature;
      });
      var weights = softmax(scores);
      var resultVector = [0, 0, 0];
      attentionTokens.forEach(function (token, index) {
        var scoreCell = document.createElement("div");
        scoreCell.className = "matrix__cell"; scoreCell.dataset.attLink = "score-" + index;
        scoreCell.innerHTML = "<strong>" + token + "</strong><span>" + (scores[index] < -1e8 ? "masked" : scores[index].toFixed(3)) + "</span>";
        scoreMatrix.appendChild(scoreCell);
        var bar = document.createElement("div");
        bar.className = "bars__item"; bar.dataset.attLink = "weight-" + index;
        bar.innerHTML = "<strong>" + token + "</strong><div>" + weights[index].toFixed(3) + '</div><div class="bars__track"><div class="bars__fill" style="width:' + weights[index] * 100 + '%"></div></div>';
        softmaxBars.appendChild(bar);
        tokenVectors[token].forEach(function (value, dim) { resultVector[dim] += value * weights[index]; });
      });
      attResult.textContent = "[" + resultVector.map(function (v) { return v.toFixed(3); }).join(", ") + "]";
      attResult.dataset.attLink = "result";
      attExplain.textContent = "当前 query 是 “" + queryToken + "”。它先和所有 key 做点积得到 score,再经过 softmax 变成权重;权重大说明这个 token 对当前 query 更重要。最后对所有 value 做加权求和,得到新的上下文表示。";
      fMain.innerHTML = linkA("Q", ["token-" + state.queryIndex]) + " · " + linkA("K", attentionTokens.map(function (_, i) { return "token-" + i; })) + " / √d → " + linkA("score", attentionTokens.map(function (_, i) { return "score-" + i; }));
      fSub.innerHTML = linkA("weight = softmax(score)", attentionTokens.map(function (_, i) { return "weight-" + i; })) + ", " + linkA("output = Σ(weight × V)", attentionTokens.map(function (_, i) { return "weight-" + i; }).concat(attentionTokens.map(function (_, i) { return "token-" + i; })).concat(["result"]));
      bindLinks();
      $('[data-attention-read="temperature"]').textContent = state.temperature.toFixed(1);
    }
    $('[data-attention="temperature"]').addEventListener("input", function (e) { state.temperature = Number(e.target.value); render(); });
    $('[data-attention="causal"]').addEventListener("change", function (e) { state.causal = e.target.checked; render(); });
    render();
  }

  /* ===================== 4. 多头热力图 ===================== */
  var MULTI_TPL =
    '<div class="multihead">' +
    '  <div class="multihead__intro"><h3>为什么要多头(Multi-Head)</h3>' +
    '    <p>单头 attention 只有一种“关注视角”。多头的想法是:同一个 token,在不同子空间里学不同关系——一个头偏向“谁是主语”,一个头偏向“动作和宾语”,还有一个头更关注“位置或局部结构”。</p></div>' +
    '  <div class="multihead-flow" aria-label="多头注意力把同一输入分到多个 head">' +
    '    <svg viewBox="0 0 680 230" role="img">' +
    '      <defs>' +
    '        <marker id="mh-arrow" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#c4b7ff" /></marker>' +
    '        <marker id="mh-main-arrow" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#8499bd" /></marker>' +
    '      </defs>' +
    '      <rect x="36" y="92" width="150" height="44" rx="10" />' +
    '      <text x="111" y="111" text-anchor="middle">同一输入 X</text>' +
    '      <text x="111" y="127" text-anchor="middle" class="multihead-flow__sub">完整 token 表示</text>' +
    '      <line class="multihead-flow__main" x1="186" y1="114" x2="250" y2="114" marker-end="url(#mh-main-arrow)" />' +
    '      <line class="multihead-flow__split" x1="250" y1="54" x2="250" y2="174" />' +
    '      <line class="multihead-flow__head" x1="250" y1="54" x2="330" y2="54" marker-end="url(#mh-arrow)" />' +
    '      <line class="multihead-flow__head" x1="250" y1="114" x2="330" y2="114" marker-end="url(#mh-arrow)" />' +
    '      <line class="multihead-flow__head" x1="250" y1="174" x2="330" y2="174" marker-end="url(#mh-arrow)" />' +
    '      <g class="multihead-flow__heads">' +
    '        <rect x="330" y="32" width="120" height="44" rx="10" />' +
    '        <text x="390" y="50" text-anchor="middle">Head 1</text>' +
    '        <text x="390" y="66" text-anchor="middle">Q1/K1/V1</text>' +
    '        <rect x="330" y="92" width="120" height="44" rx="10" />' +
    '        <text x="390" y="110" text-anchor="middle">Head 2</text>' +
    '        <text x="390" y="126" text-anchor="middle">Q2/K2/V2</text>' +
    '        <rect x="330" y="152" width="120" height="44" rx="10" />' +
    '        <text x="390" y="170" text-anchor="middle">Head 3</text>' +
    '        <text x="390" y="186" text-anchor="middle">Q3/K3/V3</text>' +
    '      </g>' +
    '      <line class="multihead-flow__main" x1="450" y1="54" x2="530" y2="114" marker-end="url(#mh-main-arrow)" />' +
    '      <line class="multihead-flow__main" x1="450" y1="114" x2="530" y2="114" marker-end="url(#mh-main-arrow)" />' +
    '      <line class="multihead-flow__main" x1="450" y1="174" x2="530" y2="114" marker-end="url(#mh-main-arrow)" />' +
    '      <rect x="530" y="92" width="120" height="44" rx="10" />' +
    '      <text x="590" y="110" text-anchor="middle">Concat + W<tspan baseline-shift="sub">O</tspan></text>' +
    '      <text x="590" y="126" text-anchor="middle" class="multihead-flow__sub">拼接再融合</text>' +
    '    </svg>' +
    '  </div>' +
    '  <div class="multihead__grid">' +
    '    <div class="multihead__panel"><h3>头选择</h3><div class="token-picker" data-head-picker></div><p class="explain" data-head-summary></p></div>' +
    '    <div class="multihead__panel"><h3>每个头的注意力热度</h3><div class="head-heatmaps" data-head-heatmaps></div></div>' +
    '  </div>' +
    '</div>';

  function initMultihead(root) {
    root.innerHTML = MULTI_TPL;
    var headPicker = root.querySelector("[data-head-picker]");
    var headHeatmaps = root.querySelector("[data-head-heatmaps]");
    var headSummary = root.querySelector("[data-head-summary]");
    var state = { headIndex: 0 };
    function render() {
      headPicker.innerHTML = ""; headHeatmaps.innerHTML = "";
      multiHeadPatterns.forEach(function (head, index) {
        var button = document.createElement("button");
        button.type = "button"; button.textContent = head.name;
        button.className = index === state.headIndex ? "is-active" : "";
        button.addEventListener("click", function () { state.headIndex = index; render(); });
        headPicker.appendChild(button);
      });
      multiHeadPatterns.forEach(function (head) {
        var panel = document.createElement("article");
        panel.className = "heatmap"; panel.innerHTML = "<h4>" + head.name + "</h4>";
        var grid = document.createElement("div"); grid.className = "heatmap__grid";
        head.weight.forEach(function (row, rowIndex) {
          var rowNode = document.createElement("div"); rowNode.className = "heatmap__row";
          var label = document.createElement("div"); label.className = "heatmap__label"; label.textContent = attentionTokens[rowIndex];
          rowNode.appendChild(label);
          row.forEach(function (value) {
            var cell = document.createElement("div"); cell.className = "heatmap__cell";
            var alpha = 0.14 + value * 0.86;
            cell.style.background = "rgba(106, 195, 255, " + alpha + ")";
            cell.style.color = value > 0.55 ? "#04111c" : "#f5fbff";
            cell.textContent = value.toFixed(2);
            rowNode.appendChild(cell);
          });
          grid.appendChild(rowNode);
        });
        panel.appendChild(grid); headHeatmaps.appendChild(panel);
      });
      var focusHead = multiHeadPatterns[state.headIndex];
      headSummary.textContent = focusHead.name + ": " + focusHead.summary;
    }
    render();
  }

  /* ===================== 5. 真实 attention 权重回放 ===================== */
  var REAL_TPL =
    '<div class="real-attention">' +
    '  <div class="real-attention__toolbar">' +
    '    <label>选择 attention JSON<input type="file" accept=".json,application/json" data-real-attention-file /></label>' +
    '    <button class="button button--ghost" type="button" data-real-attention-load>读取内置样例</button>' +
    '    <button class="button button--ghost" type="button" data-real-attention-prev>上一步</button>' +
    '    <button class="button button--ghost" type="button" data-real-attention-next>下一步</button>' +
    '    <div class="propagation__status" data-real-attention-step>当前还没有生成步骤</div>' +
    '  </div>' +
    '  <p class="explain" data-real-attention-status>还没有加载真实权重。你可以选择一个导出的 JSON 文件,或尝试读取内置样例。</p>' +
    '  <div class="real-attention__content" data-real-attention-content></div>' +
    '</div>';

  function initRealAttention(root) {
    root.innerHTML = REAL_TPL;
    var $ = function (s) { return root.querySelector(s); };
    var fileInput = $("[data-real-attention-file]");
    var loadBtn = $("[data-real-attention-load]");
    var prevBtn = $("[data-real-attention-prev]");
    var nextBtn = $("[data-real-attention-next]");
    var stepLabel = $("[data-real-attention-step]");
    var statusLabel = $("[data-real-attention-status]");
    var content = $("[data-real-attention-content]");
    var state = { data: null, stepIndex: 0 };

    function render(data) {
      content.innerHTML = "";
      if (!data || !Array.isArray(data.tokens) || !Array.isArray(data.layers)) { statusLabel.textContent = "attention JSON 结构不正确。"; return; }
      var stepData = Array.isArray(data.steps) && data.steps.length > 0 ? data.steps[state.stepIndex] : null;
      var tokens = stepData ? stepData.context_tokens : data.tokens;
      var layers = stepData ? stepData.layers : data.layers;
      statusLabel.textContent = "已加载真实样例:backbone = " + (data.backbone || "unknown") + ",token 数 = " + tokens.length;
      if (stepData) stepLabel.textContent = "当前步骤:" + (state.stepIndex + 1) + " / " + data.steps.length + ",预测下一个 token = " + stepData.predicted_token;
      else stepLabel.textContent = "当前展示最终整段 attention 权重";
      layers.forEach(function (layer) {
        var layerNode = document.createElement("section"); layerNode.className = "real-attention__layer";
        layerNode.innerHTML = "<h3>Layer " + layer.layer_index + "</h3>";
        var heatmaps = document.createElement("div"); heatmaps.className = "head-heatmaps";
        layer.heads.forEach(function (head, headIndex) {
          var panel = document.createElement("article"); panel.className = "heatmap"; panel.innerHTML = "<h4>Head " + headIndex + "</h4>";
          var grid = document.createElement("div"); grid.className = "heatmap__grid";
          head.forEach(function (row, rowIndex) {
            var rowNode = document.createElement("div"); rowNode.className = "heatmap__row";
            var label = document.createElement("div"); label.className = "heatmap__label"; label.textContent = tokens[rowIndex];
            rowNode.appendChild(label);
            row.forEach(function (value) {
              var cell = document.createElement("div"); cell.className = "heatmap__cell";
              var alpha = 0.14 + Number(value) * 0.86;
              cell.style.background = "rgba(106, 195, 255, " + alpha + ")";
              cell.style.color = Number(value) > 0.55 ? "#04111c" : "#f5fbff";
              cell.textContent = Number(value).toFixed(2);
              rowNode.appendChild(cell);
            });
            grid.appendChild(rowNode);
          });
          panel.appendChild(grid); heatmaps.appendChild(panel);
        });
        layerNode.appendChild(heatmaps); content.appendChild(layerNode);
      });
    }
    function apply(data) { state.data = data; state.stepIndex = 0; render(data); }

    fileInput.addEventListener("change", function (event) {
      var file = event.target.files && event.target.files[0];
      if (!file) return;
      file.text().then(function (text) { apply(JSON.parse(text)); }).catch(function (error) { statusLabel.textContent = "读取文件失败:" + error.message; });
    });
    loadBtn.addEventListener("click", function () {
      fetch("assets/attention-sample.json").then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); }).then(apply).catch(function (error) {
        statusLabel.textContent = "读取内置样例失败:" + error.message + "。如果你是直接 file:// 打开页面,请改用本地静态服务器,或手动选择 JSON 文件。";
      });
    });
    prevBtn.addEventListener("click", function () { if (!state.data || !Array.isArray(state.data.steps) || !state.data.steps.length) return; state.stepIndex = Math.max(0, state.stepIndex - 1); render(state.data); });
    nextBtn.addEventListener("click", function () { if (!state.data || !Array.isArray(state.data.steps) || !state.data.steps.length) return; state.stepIndex = Math.min(state.data.steps.length - 1, state.stepIndex + 1); render(state.data); });
  }

  /* ===================== 激活函数曲线实验台 ===================== */
  var ACTC_TPL =
    '<div class="lab">' +
    '  <div class="lab__controls">' +
    '    <label>激活函数' +
    '      <select data-actc="fn">' +
    '        <option value="sigmoid">Sigmoid</option>' +
    '        <option value="tanh">Tanh</option>' +
    '        <option value="relu">ReLU</option>' +
    '        <option value="leaky_relu">LeakyReLU</option>' +
    '        <option value="gelu">GELU</option>' +
    '      </select>' +
    '    </label>' +
    '    <label>取值点 x<input type="range" min="-4" max="4" step="0.1" value="1.0" data-actc="x" /><span class="lab__value" data-actc-read="x"></span></label>' +
    '  </div>' +
    '  <div class="lab__viz">' +
    '    <div class="formula-card">' +
    '      <div data-actc-plot></div>' +
    '      <p class="formula" data-actc-formula></p>' +
    '      <p class="explain" data-actc-explain></p>' +
    '    </div>' +
    '  </div>' +
    '</div>';

  function actFn(x, name) {
    if (name === "relu") return Math.max(0, x);
    if (name === "tanh") return Math.tanh(x);
    if (name === "leaky_relu") return x > 0 ? x : 0.01 * x;
    if (name === "gelu") {
      var s = x < 0 ? -1 : 1, ax = Math.abs(x / Math.SQRT2), t = 1 / (1 + 0.3275911 * ax);
      var erf = s * (1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax));
      return 0.5 * x * (1 + erf);
    }
    return 1 / (1 + Math.exp(-x));
  }

  function initActivationCurve(root) {
    root.innerHTML = ACTC_TPL;
    var state = { fn: "sigmoid", x: 1.0 };
    var W = 520, H = 250, padL = 34, padR = 14, padT = 14, padB = 24;
    var xMin = -4, xMax = 4, yMin = -1.5, yMax = 4.2;
    function PX(x) { return padL + (x - xMin) / (xMax - xMin) * (W - padL - padR); }
    function PY(y) { return padT + (yMax - y) / (yMax - yMin) * (H - padT - padB); }
    function clampY(y) { return Math.max(yMin, Math.min(yMax, y)); }
    function render() {
      root.querySelector('[data-actc-read="x"]').textContent = state.x.toFixed(2);
      var pts = [];
      for (var i = 0; i <= 120; i++) {
        var xx = xMin + (xMax - xMin) * i / 120;
        pts.push(PX(xx).toFixed(1) + "," + PY(clampY(actFn(xx, state.fn))).toFixed(1));
      }
      var fx = actFn(state.x, state.fn);
      var h = 1e-3, d = (actFn(state.x + h, state.fn) - actFn(state.x - h, state.fn)) / (2 * h);
      var tx1 = state.x - 0.9, tx2 = state.x + 0.9;
      var ty1 = fx + d * (tx1 - state.x), ty2 = fx + d * (tx2 - state.x);
      var svg =
        '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="激活函数曲线" style="width:100%;height:auto;display:block">' +
        '<line x1="' + PX(xMin) + '" y1="' + PY(0) + '" x2="' + PX(xMax) + '" y2="' + PY(0) + '" stroke="#46587a" stroke-width="1"/>' +
        '<line x1="' + PX(0) + '" y1="' + padT + '" x2="' + PX(0) + '" y2="' + (H - padB) + '" stroke="#46587a" stroke-width="1"/>' +
        '<polyline points="' + pts.join(" ") + '" fill="none" stroke="#6ac3ff" stroke-width="2.5"/>' +
        '<line x1="' + PX(tx1) + '" y1="' + PY(clampY(ty1)) + '" x2="' + PX(tx2) + '" y2="' + PY(clampY(ty2)) + '" stroke="#8ef0d1" stroke-width="2" stroke-dasharray="4 3"/>' +
        '<circle cx="' + PX(state.x) + '" cy="' + PY(clampY(fx)) + '" r="5" fill="#ffcf72"/>' +
        "</svg>";
      root.querySelector("[data-actc-plot]").innerHTML = svg;
      root.querySelector("[data-actc-formula]").textContent =
        "f(" + state.x.toFixed(2) + ") = " + fx.toFixed(3) + "    f′(" + state.x.toFixed(2) + ") = " + d.toFixed(3) + " (切线斜率)";
      root.querySelector("[data-actc-explain]").textContent = Math.abs(d) < 0.05
        ? "此处曲线几乎是平的、导数≈0 → 反向传播的梯度会在这里“断流”(梯度消失)。拖到中间试试。"
        : "此处曲线有明显斜率、导数≠0 → 梯度能顺畅地传回去。拖到两端(尤其 Sigmoid/Tanh)看它怎么变平。";
    }
    root.querySelectorAll("[data-actc]").forEach(function (c) {
      c.addEventListener("input", function () {
        var k = c.getAttribute("data-actc");
        state[k] = k === "x" ? parseFloat(c.value) : c.value;
        render();
      });
    });
    render();
  }

  /* ===================== 梯度下降实验台 ===================== */
  var GD_TPL =
    '<div class="lab">' +
    '  <div class="lab__controls">' +
    '    <label>学习率 η<input type="range" min="0.2" max="12" step="0.2" value="1.0" data-gd="lr" /><span class="lab__value" data-gd-read="lr"></span></label>' +
    '    <div class="lab__btns">' +
    '      <button type="button" class="button button--primary" data-gd-act="step">走一步</button>' +
    '      <button type="button" class="button button--ghost" data-gd-act="run">自动跑</button>' +
    '      <button type="button" class="button button--ghost" data-gd-act="reset">重置</button>' +
    '    </div>' +
    '  </div>' +
    '  <div class="lab__viz">' +
    '    <div class="formula-card">' +
    '      <div data-gd-plot></div>' +
    '      <p class="explain" data-gd-explain></p>' +
    '    </div>' +
    '  </div>' +
    '</div>';

  function initGradientDescent(root) {
    root.innerHTML = GD_TPL;
    var START = -4.4;
    var state = { lr: 1.0, w: START, hist: [START], diverged: false };
    function f(w) { return 0.1 * w * w; }
    function df(w) { return 0.2 * w; }
    var W = 520, H = 250, padL = 30, padR = 14, padT = 14, padB = 24;
    var xMin = -5, xMax = 5, yMin = 0, yMax = 2.7;
    function PX(x) { return padL + (x - xMin) / (xMax - xMin) * (W - padL - padR); }
    function PY(y) { return padT + (yMax - y) / (yMax - yMin) * (H - padT - padB); }
    function cx(x) { return Math.max(xMin, Math.min(xMax, x)); }
    function render() {
      root.querySelector('[data-gd-read="lr"]').textContent = state.lr.toFixed(1);
      var pts = [];
      for (var i = 0; i <= 120; i++) {
        var xx = xMin + (xMax - xMin) * i / 120;
        pts.push(PX(xx).toFixed(1) + "," + PY(Math.min(yMax, f(xx))).toFixed(1));
      }
      var dots = "", path = "";
      state.hist.forEach(function (w, i) {
        var X = PX(cx(w)), Y = PY(Math.min(yMax, f(cx(w))));
        path += (i === 0 ? "M" : "L") + X.toFixed(1) + "," + Y.toFixed(1) + " ";
        dots += '<circle cx="' + X.toFixed(1) + '" cy="' + Y.toFixed(1) + '" r="3" fill="#8ef0d1" opacity="0.7"/>';
      });
      var cw = state.hist[state.hist.length - 1];
      var svg =
        '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="梯度下降" style="width:100%;height:auto;display:block">' +
        '<polyline points="' + pts.join(" ") + '" fill="none" stroke="#6ac3ff" stroke-width="2.5"/>' +
        '<path d="' + path + '" fill="none" stroke="#8ef0d1" stroke-width="1.4" stroke-dasharray="3 3"/>' +
        dots +
        '<circle cx="' + PX(cx(cw)).toFixed(1) + '" cy="' + PY(Math.min(yMax, f(cx(cw)))).toFixed(1) + '" r="6" fill="#ffcf72"/>' +
        "</svg>";
      root.querySelector("[data-gd-plot]").innerHTML = svg;
      var msg;
      if (state.diverged) msg = "学习率太大,球冲出了谷底、损失反而越来越大 → 发散了!调小 η 再重置试试。";
      else if (Math.abs(df(cw)) < 0.05) msg = "梯度≈0,已经滑到谷底附近,基本收敛。step = " + (state.hist.length - 1) + "。";
      else msg = "还在下坡:step = " + (state.hist.length - 1) + ",w = " + cw.toFixed(2) + ",loss = " + f(cw).toFixed(3) + "。";
      root.querySelector("[data-gd-explain]").textContent = msg;
    }
    function step() {
      if (state.diverged) return;
      var g = df(state.w);
      state.w = state.w - state.lr * g;
      state.hist.push(state.w);
      if (Math.abs(state.w) > 30) state.diverged = true;
      render();
    }
    function reset() { state.w = START; state.hist = [START]; state.diverged = false; render(); }
    root.querySelector('[data-gd="lr"]').addEventListener("input", function () {
      state.lr = parseFloat(this.value); render();
    });
    root.querySelectorAll("[data-gd-act]").forEach(function (b) {
      b.addEventListener("click", function () {
        var act = b.getAttribute("data-gd-act");
        if (act === "step") step();
        else if (act === "reset") reset();
        else if (act === "run") {
          var n = 0;
          (function loop() {
            if (n++ >= 20 || state.diverged) return;
            step();
            setTimeout(loop, 140);
          })();
        }
      });
    });
    render();
  }

  /* ===================== 语料清洗(第 20 章) ===================== */
  var CLEAN_TPL =
    '<div class="lab">' +
    '  <div class="lab__controls">' +
    '    <p class="explain">下面是 8 条「刚爬下来」的原始文本。逐个打开过滤器,看脏数据<strong>按什么方法、什么判据</strong>被剥掉:</p>' +
    '    <label class="checkbox"><input type="checkbox" data-clean="nav" />抽正文 · 去网页模板/导航</label>' +
    '    <label class="checkbox"><input type="checkbox" data-clean="lowq" />质量过滤 · 规则+分类器</label>' +
    '    <label class="checkbox"><input type="checkbox" data-clean="dedup" />精确去重 · 哈希</label>' +
    '    <label class="checkbox"><input type="checkbox" data-clean="neardup" />近似去重 · MinHash/Jaccard</label>' +
    '    <label class="checkbox"><input type="checkbox" data-clean="pii" />隐私过滤 · 正则+NER</label>' +
    '  </div>' +
    '  <div class="lab__viz">' +
    '    <div class="formula-card">' +
    '      <h3>清洗结果</h3>' +
    '      <p class="explain" data-clean-stat></p>' +
    '      <div class="clean-list" data-clean-out></div>' +
    '    </div>' +
    '  </div>' +
    '</div>';

  function initCorpusClean(root) {
    root.innerHTML = CLEAN_TPL;
    // method: 触发它的过滤器; how: 这条“凭什么被判掉”的白箱说明
    var RAW = [
      { kind: "keep", text: "反向传播通过链式法则,把输出层的误差一层层传回每一个参数。" },
      { kind: "nav", method: "nav", text: "首页 > 教程 > 深度学习    登录 | 注册 | 关于我们 | 联系我们",
        how: "正文抽取:文本块太短、链接/符号占比过高,判为模板碎块" },
      { kind: "lowq", method: "lowq", text: "【劲爆】三天学会 AI!点击领取内部绝密资料,加微信限时免费领!!!",
        how: "质量过滤:命中脏词表(领取/限时/免费)+ 感叹号密度过高,分类器判低质" },
      { kind: "keep", text: "梯度下降沿损失下降最快的方向更新权重,学习率决定每一步走多远。" },
      { kind: "dedup", method: "dedup", text: "反向传播通过链式法则,把输出层的误差一层层传回每一个参数。",
        how: "精确去重:哈希值与第 1 条完全相同(逐字一致)" },
      { kind: "lowq", method: "lowq", text: "这个文章是非常好的关于神经网络,你应该去阅读它因为对学习很有帮助的。",
        how: "质量过滤:机器翻译腔,语言模型困惑度偏高,信息密度低" },
      { kind: "neardup", method: "neardup", text: "反向传播利用链式法则,将输出层误差逐层回传到每个参数。(某站转载)",
        how: "近似去重:与第 1 条 n-gram Jaccard ≈ 0.82 > 0.8 阈值,判为改写转载" },
      { kind: "pii", method: "pii", text: "作者微信 lei_1990,手机 138-0013-8000,家住北京市海淀区中关村。",
        how: "隐私过滤:正则命中手机号,NER 识别到人名/地址" }
    ];
    var LABELS = {
      keep: "正文", nav: "网页模板", lowq: "广告/低质",
      dedup: "完全重复", neardup: "近似重复", pii: "隐私信息"
    };
    var state = { nav: false, lowq: false, dedup: false, neardup: false, pii: false };
    var out = root.querySelector("[data-clean-out]");
    var stat = root.querySelector("[data-clean-stat]");

    function render() {
      var kept = 0, html = "";
      RAW.forEach(function (row) {
        var dropped = !!row.method && state[row.method];
        if (!dropped) kept++;
        html +=
          '<div class="clean-row ' + (dropped ? "clean-row--drop" : "clean-row--keep") + '">' +
          '<span class="clean-row__tag">' + LABELS[row.kind] + "</span>" +
          '<span class="clean-row__text">' + row.text + "</span>" +
          (dropped ? '<span class="clean-row__how">✕ ' + row.how + "</span>" : "") +
          "</div>";
      });
      out.innerHTML = html;
      var pct = Math.round((kept / RAW.length) * 100);
      stat.innerHTML =
        "原始 " + RAW.length + " 条 → 保留 <strong>" + kept + "</strong> 条(保留率 " + pct + "%)。" +
        (kept === 2
          ? "五道过滤器全开后只剩两条真正的正文——每条被丢的下面都写清了「按什么方法、什么判据」丢的。"
          : "勾选左侧过滤器,被丢的行会显示它<strong>凭什么被判掉</strong>(哈希 / Jaccard / 规则 / 正则)。");
    }

    root.querySelectorAll("[data-clean]").forEach(function (control) {
      control.addEventListener("change", function () {
        state[control.dataset.clean] = control.checked;
        render();
      });
    });
    render();
  }

  /* ===================== 书页 Demo: MNIST ===================== */
  var MNIST_TPL =
    '<div class="lab lab--demo">' +
    '  <div class="lab__controls lab__controls--demo">' +
    '    <div class="demo-toolbar">' +
    '      <button type="button" class="button button--primary" data-mnist-act="predict">识别</button>' +
    '      <button type="button" class="button button--ghost" data-mnist-act="clear">清除重写</button>' +
    '    </div>' +
    '    <p class="explain" data-mnist-status>正在加载模型…</p>' +
    '  </div>' +
    '  <div class="lab__viz lab__viz--demo-grid">' +
    '    <div class="formula-card">' +
    '      <h3>手写输入</h3>' +
    '      <div class="mnist-demo__canvas-wrap">' +
    '        <canvas class="mnist-demo__canvas" width="280" height="280" data-mnist-canvas></canvas>' +
    '        <span class="mnist-demo__hint" data-mnist-hint>在这里写一个 0-9 的数字</span>' +
    '      </div>' +
    '      <div class="mnist-demo__preview-row">' +
    '        <canvas class="mnist-demo__preview" width="28" height="28" data-mnist-preview></canvas>' +
    '        <p class="explain" data-mnist-meta>右侧会显示当前画板下采样到 28x28 后的模型输入。</p>' +
    '      </div>' +
    '    </div>' +
    '    <div class="formula-card">' +
    '      <h3>预测结果</h3>' +
    '      <p class="formula" data-mnist-prediction>—</p>' +
    '      <div class="mnist-demo__bars" data-mnist-bars></div>' +
    '    </div>' +
    '  </div>' +
    '</div>';

  function initMnistDemo(root) {
    root.innerHTML = MNIST_TPL;
    var modelPath = "assets/demos/mnist/model.json";
    var SIZE = 280;
    var state = {
      model: null,
      drawing: false,
      hasInk: false,
      lastPoint: null
    };
    var canvas = root.querySelector("[data-mnist-canvas]");
    var ctx = canvas.getContext("2d");
    var preview = root.querySelector("[data-mnist-preview]");
    var previewCtx = preview.getContext("2d");
    var hint = root.querySelector("[data-mnist-hint]");

    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = 18;
    ctx.strokeStyle = "#f8fbff";
    ctx.fillStyle = ctx.strokeStyle;

    function activateValue(x) {
      var name = state.model && state.model.config && state.model.config.activation;
      if (name === "relu") return x > 0 ? x : 0;
      if (name === "tanh") return Math.tanh(x);
      if (name === "leaky_relu") return x > 0 ? x : 0.01 * x;
      if (name === "gelu") {
        var sign = x < 0 ? -1 : 1;
        var ax = Math.abs(x / Math.SQRT2);
        var t = 1 / (1 + 0.3275911 * ax);
        var erfApprox =
          sign *
          (1 -
            ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
              0.254829592) * t * Math.exp(-ax * ax));
        return 0.5 * x * (1 + erfApprox);
      }
      return 1 / (1 + Math.exp(-x));
    }

    function softmax(values) {
      var maxValue = Math.max.apply(null, values);
      var exps = values.map(function (v) { return Math.exp(v - maxValue); });
      var sum = exps.reduce(function (a, b) { return a + b; }, 0);
      return exps.map(function (v) { return v / sum; });
    }

    function forward(pixels) {
      var layers = [];
      layers.push(pixels);
      for (var layer = 1; layer < state.model.weights.length; layer++) {
        var weight = state.model.weights[layer];
        var bias = state.model.biases[layer];
        var input = layers[layer - 1];
        var output = new Array(weight.length).fill(0);
        var isLastLayer = layer + 1 === state.model.weights.length;
        for (var row = 0; row < weight.length; row++) {
          var sum = bias[row];
          for (var col = 0; col < weight[row].length; col++) {
            sum += weight[row][col] * input[col];
          }
          if (isLastLayer) {
            output[row] = state.model.config && state.model.config.softmax === "none"
              ? activateValue(sum)
              : sum;
          } else {
            output[row] = activateValue(sum);
          }
        }
        layers.push(output);
      }
      var logits = layers[layers.length - 1];
      return state.model.config && state.model.config.softmax === "none" ? logits : softmax(logits);
    }

    function normalizeOutput(values) {
      if (state.model && state.model.config && state.model.config.softmax === "none") {
        var clipped = values.map(function (value) { return Math.max(0, value); });
        var total = clipped.reduce(function (acc, value) { return acc + value; }, 0);
        if (total > 0) {
          return clipped.map(function (value) { return value / total; });
        }
      }
      var sum = values.reduce(function (acc, value) { return acc + value; }, 0);
      var minValue = Math.min.apply(null, values);
      if (minValue >= 0 && sum > 0.99 && sum < 1.01) {
        return values;
      }
      return softmax(values);
    }

    function eventPoint(event) {
      var rect = canvas.getBoundingClientRect();
      var point = event.touches ? event.touches[0] : event;
      return {
        x: (point.clientX - rect.left) * (SIZE / rect.width),
        y: (point.clientY - rect.top) * (SIZE / rect.height)
      };
    }

    function startDrawing(event) {
      event.preventDefault();
      if (!state.model) return;
      state.drawing = true;
      state.hasInk = true;
      state.lastPoint = eventPoint(event);
      hint.classList.add("is-hidden");
      ctx.beginPath();
      ctx.arc(state.lastPoint.x, state.lastPoint.y, ctx.lineWidth / 2, 0, Math.PI * 2);
      ctx.fill();
    }

    function moveDrawing(event) {
      if (!state.drawing) return;
      event.preventDefault();
      var point = eventPoint(event);
      ctx.beginPath();
      ctx.moveTo(state.lastPoint.x, state.lastPoint.y);
      ctx.lineTo(point.x, point.y);
      ctx.stroke();
      state.lastPoint = point;
    }

    function endDrawing(event) {
      if (!state.drawing) return;
      event.preventDefault();
      state.drawing = false;
      predictCurrentInput();
    }

    function clearAll() {
      ctx.clearRect(0, 0, SIZE, SIZE);
      previewCtx.clearRect(0, 0, 28, 28);
      state.hasInk = false;
      state.lastPoint = null;
      hint.classList.remove("is-hidden");
      renderBars(new Array(10).fill(0), -1);
      root.querySelector("[data-mnist-prediction]").textContent = "等待输入";
      root.querySelector("[data-mnist-meta]").textContent =
        "右侧会显示当前画板下采样到 28x28 后的模型输入。";
      if (state.model) {
        root.querySelector("[data-mnist-status]").textContent =
          "模型已加载。直接在画板上写数字，松手后会在浏览器里完成一次前向推理。";
      }
    }

    function extractPixels() {
      var img = ctx.getImageData(0, 0, SIZE, SIZE).data;
      var minX = SIZE;
      var minY = SIZE;
      var maxX = -1;
      var maxY = -1;
      for (var y = 0; y < SIZE; y++) {
        for (var x = 0; x < SIZE; x++) {
          if (img[(y * SIZE + x) * 4 + 3] > 32) {
            if (x < minX) minX = x;
            if (y < minY) minY = y;
            if (x > maxX) maxX = x;
            if (y > maxY) maxY = y;
          }
        }
      }
      var pixels = new Array(28 * 28).fill(0);
      if (maxX < 0) return pixels;

      var bw = maxX - minX + 1;
      var bh = maxY - minY + 1;
      var scale = 20 / Math.max(bw, bh);
      var dw = Math.max(1, Math.round(bw * scale));
      var dh = Math.max(1, Math.round(bh * scale));
      var tmp = document.createElement("canvas");
      tmp.width = 28;
      tmp.height = 28;
      var tmpCtx = tmp.getContext("2d");
      tmpCtx.imageSmoothingEnabled = true;
      var dx = Math.floor((28 - dw) / 2);
      var dy = Math.floor((28 - dh) / 2);
      tmpCtx.drawImage(canvas, minX, minY, bw, bh, dx, dy, dw, dh);

      var data = tmpCtx.getImageData(0, 0, 28, 28).data;
      var dense = new Array(28 * 28).fill(0);
      var sx = 0;
      var sy = 0;
      var mass = 0;
      for (var py = 0; py < 28; py++) {
        for (var px = 0; px < 28; px++) {
          var value = data[(py * 28 + px) * 4 + 3] / 255;
          if (value <= 0.08) continue;
          dense[py * 28 + px] = value;
          sx += px * value;
          sy += py * value;
          mass += value;
        }
      }
      if (mass <= 0) return pixels;

      var shiftX = Math.round(14 - sx / mass);
      var shiftY = Math.round(14 - sy / mass);
      for (var row = 0; row < 28; row++) {
        for (var col = 0; col < 28; col++) {
          var source = dense[row * 28 + col];
          if (source <= 0) continue;
          var ny = row + shiftY;
          var nx = col + shiftX;
          if (ny >= 0 && ny < 28 && nx >= 0 && nx < 28) {
            pixels[ny * 28 + nx] = Math.max(pixels[ny * 28 + nx], source);
          }
        }
      }
      return pixels;
    }

    function renderPreview(pixels) {
      var im = previewCtx.createImageData(28, 28);
      for (var y = 0; y < 28; y++) {
        for (var x = 0; x < 28; x++) {
          var value = pixels[y * 28 + x];
          var shade = Math.round(value * 255);
          var idx = (y * 28 + x) * 4;
          im.data[idx] = shade;
          im.data[idx + 1] = shade;
          im.data[idx + 2] = shade;
          im.data[idx + 3] = 255;
        }
      }
      previewCtx.putImageData(im, 0, 0);
    }

    function renderBars(probs, prediction) {
      var bars = root.querySelector("[data-mnist-bars]");
      bars.innerHTML = probs.map(function (p, idx) {
        var active = idx === prediction ? " mnist-demo__bar--top" : "";
        return '<div class="mnist-demo__bar-row"><span class="mnist-demo__bar-label">' + idx +
          '</span><div class="mnist-demo__bar-track"><div class="mnist-demo__bar' + active +
          '" style="width:' + (p * 100).toFixed(2) + '%"></div></div><span class="mnist-demo__bar-value">' +
          (p * 100).toFixed(1) + '%</span></div>';
      }).join("");
    }

    function predictCurrentInput() {
      if (!state.model) {
        return;
      }
      if (!state.hasInk) {
        root.querySelector("[data-mnist-status]").textContent = "请先在画板上写一个数字。";
        return;
      }
      var pixels = extractPixels();
      renderPreview(pixels);
      var outputs = forward(pixels);
      var probs = normalizeOutput(outputs);
      var prediction = 0;
      for (var i = 1; i < outputs.length; i++) {
        if (outputs[i] > outputs[prediction]) prediction = i;
      }
      renderBars(probs, prediction);
      root.querySelector("[data-mnist-status]").textContent =
        "已用当前手写输入完成一次前向推理。继续补笔或清除重写都可以。";
      root.querySelector("[data-mnist-meta]").textContent =
        "当前笔迹已裁剪、缩放并居中到 28x28，和 C++ MNIST demo 的输入维度一致。";
      root.querySelector("[data-mnist-prediction]").textContent =
        (state.model.config && state.model.config.softmax === "none"
          ? "预测 " + prediction + " · 相对分数 " + (probs[prediction] * 100).toFixed(1) + "%"
          : "预测 " + prediction + " · 置信度 " + (probs[prediction] * 100).toFixed(1) + "%");
    }

    function loadModel() {
      loadDemoJsonWithFallback(modelPath)
        .then(function (data) {
          state.model = data;
          root.querySelector("[data-mnist-status]").textContent =
            "模型已加载。直接在画板上写数字，松手后会在浏览器里完成一次前向推理。";
          clearAll();
        })
        .catch(function (err) {
          root.querySelector("[data-mnist-status]").textContent =
            "演示资源未就绪: " + err.message + "。Pages workflow 发布成功后这里会自动可用。";
        });
    }

    canvas.addEventListener("mousedown", startDrawing);
    canvas.addEventListener("mousemove", moveDrawing);
    window.addEventListener("mouseup", endDrawing);
    canvas.addEventListener("touchstart", startDrawing, { passive: false });
    canvas.addEventListener("touchmove", moveDrawing, { passive: false });
    canvas.addEventListener("touchend", endDrawing, { passive: false });
    canvas.addEventListener("touchcancel", endDrawing, { passive: false });

    root.addEventListener("click", function (event) {
      var actionButton = event.target.closest("[data-mnist-act]");
      if (!actionButton) return;
      var action = actionButton.getAttribute("data-mnist-act");
      if (action === "clear") {
        clearAll();
      } else if (action === "predict") {
        predictCurrentInput();
      }
    });

    renderBars(new Array(10).fill(0), -1);
    loadModel();
  }

  /* ===================== 书页 Demo: TicTacToe ===================== */
  var TICTACTOE_TPL =
    '<div class="lab lab--demo">' +
    '  <div class="lab__controls lab__controls--demo">' +
    '    <div class="demo-toolbar">' +
    '      <button type="button" class="button button--primary" data-ttt-act="reset">重新开局</button>' +
    '      <button type="button" class="button button--ghost" data-ttt-act="show-values">切换 Q 值显示</button>' +
    '    </div>' +
    '    <p class="explain" data-ttt-status>正在加载 Q 表…</p>' +
    '  </div>' +
    '  <div class="lab__viz lab__viz--demo-grid">' +
    '    <div class="formula-card">' +
    '      <h3>对局棋盘</h3>' +
    '      <div class="ttt-demo__board" data-ttt-board></div>' +
    '      <p class="explain" data-ttt-meta></p>' +
    '    </div>' +
    '    <div class="formula-card">' +
    '      <h3>当前状态 Q 值</h3>' +
    '      <div class="ttt-demo__values" data-ttt-values></div>' +
    '    </div>' +
    '  </div>' +
    '</div>';

  function initTictactoeDemo(root) {
    root.innerHTML = TICTACTOE_TPL;
    var dataPath = "assets/demos/tictactoe/q_table.json";
    var WIN_LINES = [
      [0, 1, 2], [3, 4, 5], [6, 7, 8],
      [0, 3, 6], [1, 4, 7], [2, 5, 8],
      [0, 4, 8], [2, 4, 6]
    ];
    var state = {
      data: null,
      board: new Array(9).fill(0),
      showValues: true,
      message: "",
      agentViewBoard: null
    };

    function legalActions(board) {
      var out = [];
      for (var i = 0; i < 9; i++) if (board[i] === 0) out.push(i);
      return out;
    }

    function encodeBoard(board) {
      var key = 0;
      var base = 1;
      for (var i = 0; i < 9; i++) {
        key += board[i] * base;
        base *= 3;
      }
      return String(key);
    }

    function result(board) {
      for (var i = 0; i < WIN_LINES.length; i++) {
        var line = WIN_LINES[i];
        var a = board[line[0]];
        if (a !== 0 && a === board[line[1]] && a === board[line[2]]) return a;
      }
      return legalActions(board).length ? 0 : 3;
    }

    function rowForBoard(board) {
      var key = encodeBoard(board);
      return (state.data.qTable && state.data.qTable[key]) || [];
    }

    function bestAction(board) {
      var row = rowForBoard(board);
      var actions = legalActions(board);
      if (!actions.length) return -1;
      var best = actions[0];
      var bestValue = row[best] || 0;
      for (var i = 1; i < actions.length; i++) {
        var action = actions[i];
        var value = row[action] || 0;
        if (value > bestValue) {
          bestValue = value;
          best = action;
        }
      }
      return best;
    }

    function renderValues() {
      var el = root.querySelector("[data-ttt-values]");
      var boardForValues = state.agentViewBoard || state.board;
      var row = rowForBoard(boardForValues);
      var actions = legalActions(boardForValues);
      if (!actions.length) {
        el.innerHTML = '<p class="muted">终局状态没有可选动作。</p>';
        return;
      }
      el.innerHTML = actions.map(function (action) {
        var value = row[action] || 0;
        return '<div class="ttt-demo__value-row"><span>落子 ' + action +
          '</span><span>' + value.toFixed(4) + '</span></div>';
      }).join("");
    }

    function renderBoard() {
      var boardEl = root.querySelector("[data-ttt-board]");
      var overlayBoard = state.agentViewBoard || state.board;
      var overlayRow = rowForBoard(overlayBoard);
      boardEl.innerHTML = state.board.map(function (cell, idx) {
        var text = cell === 1 ? "X" : cell === 2 ? "O" : "";
        var disabled = cell !== 0 || result(state.board) !== 0 ? " disabled" : "";
        return '<button type="button" class="ttt-demo__cell" data-ttt-cell="' + idx + '"' + disabled + '>' +
          '<span>' + text + '</span>' +
          (state.showValues && overlayBoard[idx] === 0 && state.data && state.data.qTable ? '<small>' +
            ((overlayRow[idx] || 0).toFixed(2)) + '</small>' : '') +
          '</button>';
      }).join("");
      renderValues();
      var res = result(state.board);
      var meta = "";
      if (res === 1) meta = "X 获胜。";
      else if (res === 2) meta = "你执 O，这局输了。";
      else if (res === 3) meta = "这局和棋。";
      else if (state.board.some(function (cell) { return cell === 2; })) {
        meta = "棋盘显示的是当前局面；格子旁和右侧列表展示的是 X 上一步决策时评估过的动作值。";
      } else {
        meta = "你执 O，点击空格落子；X 会按训练好的 Q 表选择动作。";
      }
      root.querySelector("[data-ttt-meta]").textContent = meta;
    }

    function reset() {
      state.board = new Array(9).fill(0);
      state.agentViewBoard = state.board.slice();
      if (state.data) {
        var action = bestAction(state.board);
        if (action >= 0) state.board[action] = 1;
      }
      renderBoard();
    }

    function onPlayerMove(action) {
      if (state.board[action] !== 0 || result(state.board) !== 0) return;
      state.board[action] = 2;
      if (result(state.board) !== 0) {
        state.agentViewBoard = state.board.slice();
        renderBoard();
        return;
      }
      state.agentViewBoard = state.board.slice();
      var reply = bestAction(state.board);
      if (reply >= 0) state.board[reply] = 1;
      renderBoard();
    }

    root.addEventListener("click", function (event) {
      var cell = event.target.closest("[data-ttt-cell]");
      if (cell) {
        onPlayerMove(Number(cell.getAttribute("data-ttt-cell")));
        return;
      }
      var actionButton = event.target.closest("[data-ttt-act]");
      if (!actionButton) return;
      var action = actionButton.getAttribute("data-ttt-act");
      if (action === "reset") {
        reset();
      } else if (action === "show-values") {
        state.showValues = !state.showValues;
        renderBoard();
      }
    });

    loadDemoJsonWithFallback(dataPath)
      .then(function (data) {
        state.data = data;
        root.querySelector("[data-ttt-status]").textContent =
          "Q 表已加载。这里展示的是训练 30000 局后的真实评估结果和当前局面各动作的 Q 值。";
        reset();
      })
      .catch(function (err) {
        root.querySelector("[data-ttt-status]").textContent =
          "演示资源未就绪: " + err.message + "。Pages workflow 发布成功后这里会自动可用。";
      });
  }

  /* ===================== 自动挂载 ===================== */
  var INITS = {
    neuron: initNeuron,
    propagation: initPropagation,
    attention: initAttention,
    multihead: initMultihead,
    "real-attention": initRealAttention,
    "activation-curve": initActivationCurve,
    "gradient-descent": initGradientDescent,
    "corpus-clean": initCorpusClean,
    "mnist-demo": initMnistDemo,
    "tictactoe-demo": initTictactoeDemo
  };

  function mountAll() {
    document.querySelectorAll("[data-lab]").forEach(function (root) {
      var type = root.getAttribute("data-lab");
      if (INITS[type] && !root.dataset.labMounted) {
        root.dataset.labMounted = "1";
        INITS[type](root);
        // a11y: 给动态生成的表单控件补一个 name, 消除“缺少 id/name”告警
        root.querySelectorAll("input, select, textarea").forEach(function (el, i) {
          if (!el.name && !el.id) el.name = type + "-field-" + i;
        });
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mountAll);
  } else {
    mountAll();
  }
})();
