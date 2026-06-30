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
    '        <div class="network__edges">' +
    '          <div class="network__edge network__edge--i0h0" data-edge="w100"><span>w100</span></div>' +
    '          <div class="network__edge network__edge--i0h1" data-edge="w101"><span>w101</span></div>' +
    '          <div class="network__edge network__edge--i1h1" data-edge="w111"><span>w111</span></div>' +
    '          <div class="network__edge network__edge--i1h2" data-edge="w112"><span>w112</span></div>' +
    '          <div class="network__edge network__edge--h0o" data-edge="w200"><span>w200</span></div>' +
    '          <div class="network__edge network__edge--h1o" data-edge="w210"><span>w210</span></div>' +
    '          <div class="network__edge network__edge--h2o" data-edge="w220"><span>w220</span></div>' +
    '        </div>' +
    '        <div class="network__layer">' +
    '          <div class="network__title">输入层</div>' +
    '          <div class="network__node-wrap"><div class="network__node" data-node="input-0">x1</div><div class="network__meta">input0 = 1.00</div></div>' +
    '          <div class="network__node-wrap"><div class="network__node" data-node="input-1">x2</div><div class="network__meta">input1 = 0.50</div></div>' +
    '        </div>' +
    '        <div class="network__layer">' +
    '          <div class="network__title">隐藏层</div>' +
    '          <div class="network__node-wrap"><div class="network__node" data-node="hidden-0">h1</div><div class="network__meta">b10 = 0.10</div></div>' +
    '          <div class="network__node-wrap"><div class="network__node" data-node="hidden-1">h2</div><div class="network__meta">b11 = -0.05</div></div>' +
    '          <div class="network__node-wrap"><div class="network__node" data-node="hidden-2">h3</div><div class="network__meta">b12 = 0.08</div></div>' +
    '        </div>' +
    '        <div class="network__layer">' +
    '          <div class="network__title">输出层</div>' +
    '          <div class="network__node-wrap"><div class="network__node" data-node="output-0">y</div><div class="network__meta">b20 = 0.20</div></div>' +
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
      pulse: $("[data-network-pulse]")
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

  /* ===================== 自动挂载 ===================== */
  var INITS = {
    neuron: initNeuron,
    propagation: initPropagation,
    attention: initAttention,
    multihead: initMultihead,
    "real-attention": initRealAttention
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
