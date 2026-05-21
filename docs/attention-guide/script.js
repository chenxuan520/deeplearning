const sections = [...document.querySelectorAll(".section, .hero__inner")]
const tocLinks = [...document.querySelectorAll(".toc a")]

const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible")
      }
    })
  },
  { threshold: 0.12 }
)

document.querySelectorAll(".reveal").forEach((node) => revealObserver.observe(node))

const sectionObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return
      const id = entry.target.getAttribute("id")
      tocLinks.forEach((link) => link.classList.toggle("is-active", link.getAttribute("href") === `#${id}`))
    })
  },
  { rootMargin: "-45% 0px -45% 0px", threshold: 0 }
)

sections.forEach((node) => {
  if (node.id) sectionObserver.observe(node)
})

const neuronState = {
  x1: 1,
  x2: 0.5,
  w1: 1.2,
  w2: -0.8,
  b: 0.4,
  activation: "sigmoid",
}

function activate(value, name) {
  if (name === "relu") return Math.max(0, value)
  if (name === "tanh") return Math.tanh(value)
  if (name === "leaky_relu") return value > 0 ? value : 0.01 * value
  if (name === "gelu") {
    // 精确 GELU = x * 0.5 * (1 + erf(x / sqrt(2)))
    // JS 没自带 erf, 这里用一个常用近似 (Abramowitz & Stegun 7.1.26)
    const sign = value < 0 ? -1 : 1
    const ax = Math.abs(value / Math.SQRT2)
    const t = 1 / (1 + 0.3275911 * ax)
    const erfApprox =
      sign *
      (1 -
        ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
          0.254829592) *
          t *
          Math.exp(-ax * ax))
    return 0.5 * value * (1 + erfApprox)
  }
  return 1 / (1 + Math.exp(-value))
}

function renderNeuronLab() {
  Object.entries(neuronState).forEach(([key, value]) => {
    const output = document.querySelector(`[data-neuron-read="${key}"]`)
    if (output) output.textContent = typeof value === "number" ? value.toFixed(2) : value
  })

  const sum = neuronState.x1 * neuronState.w1 + neuronState.x2 * neuronState.w2 + neuronState.b
  const out = activate(sum, neuronState.activation)
  const formula = `${neuronState.x1.toFixed(2)} x ${neuronState.w1.toFixed(2)} + ${neuronState.x2.toFixed(2)} x ${neuronState.w2.toFixed(2)} + ${neuronState.b.toFixed(2)} = ${sum.toFixed(3)} -> ${neuronState.activation} -> ${out.toFixed(3)}`

  document.querySelector("[data-neuron-formula]").textContent = formula
  document.querySelector("[data-neuron-bar]").style.width = `${Math.max(6, Math.min(100, ((out + 1) / 2) * 100))}%`
  document.querySelector("[data-neuron-explain]").textContent =
    `先把输入按权重放大或缩小，再加上偏置。非线性激活函数的意义，是让网络不只是做一层线性变换。现在输出 ${out.toFixed(3)}，这就是一个神经元对当前输入的“判断结果”。`
}

document.querySelectorAll("[data-neuron]").forEach((control) => {
  control.addEventListener("input", () => {
    const key = control.dataset.neuron
    neuronState[key] = control.tagName === "SELECT" ? control.value : Number(control.value)
    renderNeuronLab()
  })
})

renderNeuronLab()

const propagationState = {
  learningRate: 0.2,
}

const propagationParamDefs = {
  hidden0: { label: "w(hidden,0)", old: 0.8, grad: 0.2 },
  hidden1: { label: "w(hidden,1)", old: -0.2, grad: -0.1 },
  out0: { label: "w(out,0)", old: 1.1, grad: 0.2 },
  bout: { label: "b(out)", old: 0.2, grad: -0.2 },
}

function formatUpdate(oldValue, grad, learningRate) {
  const nextValue = oldValue - learningRate * grad
  return `${oldValue.toFixed(2)} -> ${nextValue.toFixed(2)} (grad ${grad >= 0 ? "+" : ""}${grad.toFixed(2)})`
}

function link(text, targets) {
  return `<span class="formula-link" data-link-targets="${targets.join(",")}">${text}</span>`
}

const propagationSteps = {
  forward: [
    {
      node: ["input-0", "input-1"],
      secondaryNode: ["hidden-0", "hidden-1", "hidden-2"],
      edge: ["w100", "w101", "w111", "w112"],
      title: "前向传播：输入层接收原始特征",
      explain: "这一步传的不是误差，而是原始输入值。比如图片像素、表格特征或 token embedding，都会先进入输入层。",
      pulse: { x: 12, y: 55 },
      formulaMain: `${link("x", ["input-0", "input-1"])} = [x1, x2]`,
      formulaSub: `输入层本身通常不做复杂运算，它负责把原始特征送进网络。`,
      formulaExplain: "你可以把输入层想成“数据入口”。真正的参数化变换，通常从下一层开始。",
      chainMain: "当前还没有进入链式法则",
      chainSub: "前向传播关注的是数值怎么往前算，不是梯度怎么往回传。",
      chainExplain: "所以这里先看输入值和参数如何组合成下一层的净输入。",
      values: "x = [1.00, 0.50] -> h = [?, ?, ?] -> y = [?]",
      grads: "梯度还没有开始传播，当前阶段重点是把输入送进网络。",
      params: () => [
        ["w(hidden,0)", "0.80 (等待更新)"],
        ["w(out,0)", "1.10 (等待更新)"],
      ],
    },
    {
      node: ["hidden-0", "hidden-1", "hidden-2"],
      secondaryNode: ["input-0", "input-1", "output-0"],
      edge: ["w100", "w101", "w111", "w112", "w200", "w210", "w220"],
      title: "前向传播：隐藏层做加权求和和激活",
      explain: "隐藏层把上一层输出乘权重、加偏置，再过激活函数。网络在这里逐步形成更抽象的特征表示。",
      pulse: { x: 48, y: 48 },
      formulaMain: `${link("z", ["hidden-0", "hidden-1", "hidden-2"])} = ${link("W", ["w100", "w101", "w111", "w112"])} ${link("x", ["input-0", "input-1"])} + ${link("b", ["hidden-0", "hidden-1", "hidden-2"])} `,
      formulaSub: `${link("a", ["hidden-0", "hidden-1", "hidden-2"])} = sigma(${link("z", ["hidden-0", "hidden-1", "hidden-2"])})`,
      formulaExplain: "这就是最经典的一层神经元公式：先线性变换，再经过激活函数，得到新的中间表示。",
      chainMain: `z_j = sum_i ${link("w_ji", ["w100", "w101", "w111", "w112"])} ${link("x_i", ["input-0", "input-1"])} + ${link("b_j", ["hidden-0", "hidden-1", "hidden-2"])}`,
      chainSub: `${link("a_j", ["hidden-0", "hidden-1", "hidden-2"])} = sigma(${link("z_j", ["hidden-0", "hidden-1", "hidden-2"])})`,
      chainExplain: "这一层的每个节点都在重复同一个模式：收输入、乘权重、加偏置、过激活。",
      values: "x = [1.00, 0.50] -> h = [0.42, 0.73, 0.18] -> y = [?]",
      grads: "这里仍然是数值流，不是误差信号。隐藏层的输出会作为下一层输入。",
      params: () => [
        ["z(hidden,0)", "1.00 x 0.80 + 0.50 x -0.20 + 0.10 -> 0.80"],
        ["a(hidden,0)", "sigmoid(0.80) -> 0.69"],
      ],
    },
    {
      node: ["output-0"],
      secondaryNode: ["hidden-0", "hidden-1", "hidden-2"],
      edge: ["w200", "w210", "w220"],
      title: "前向传播：输出层给出预测",
      explain: "最后一层把隐藏特征压缩成预测结果。分类任务里可能是各类别得分，语言模型里则是下一个 token 的 logits。",
      pulse: { x: 84, y: 52 },
      formulaMain: `${link("y", ["output-0"])} = ${link("W_out", ["w200", "w210", "w220"])} ${link("h", ["hidden-0", "hidden-1", "hidden-2"])} + ${link("b_out", ["output-0"])}`,
      formulaSub: `${link("loss", ["output-0"])} = L(${link("y", ["output-0"])}, target)`,
      formulaExplain: "输出层先得到预测，再和真实目标比较，才能知道模型到底错了多少。",
      chainMain: `L = 1/2 (target - ${link("y", ["output-0"])})^2`,
      chainSub: "先有预测，再谈误差",
      chainExplain: "你只有先得到输出，才能把预测和目标作比较，形成 loss。",
      values: "x = [1.00, 0.50] -> h = [0.42, 0.73, 0.18] -> y = [0.81]",
      grads: "前向阶段结束后，模型终于得到一个输出。接下来就可以用 loss 判断它到底偏了多少。",
      params: () => [
        ["y_pred", "[0.81]"],
        ["loss 前状态", "预测已得到，等待与目标比较"],
      ],
    },
  ],
  backward: [
    {
      node: ["output-0"],
      secondaryNode: ["hidden-0", "hidden-1", "hidden-2"],
      edge: ["w200", "w210", "w220"],
      title: "反向传播：先从输出误差开始",
      explain: "反向传播不是把原始输入往回传，而是把“模型错了多少”变成梯度，从输出层往回走。",
      pulse: { x: 84, y: 52 },
      formulaMain: `${link("dL/dy", ["output-0"])} = partial L / partial y`,
      formulaSub: "输出层先知道自己错了多少",
      formulaExplain: "反向传播的起点不是输入，而是损失函数。因为只有先知道损失，才知道要往哪个方向改。",
      chainMain: `${link("dL/dw200", ["w200"])} = ${link("dL/dy", ["output-0"])} x ${link("dy/dnet", ["output-0"])} x ${link("dnet/dw200", ["w200", "hidden-0"])}`,
      chainSub: `= delta_out x ${link("h0", ["hidden-0"])}`,
      chainExplain: "这就是链式法则第一次展开：总误差对输出层权重的影响，被拆成三段局部影响相乘。",
      values: "预测 y = [0.81]，真实目标 t = [1.00]",
      grads: "dL/dy = y - t = -0.19 -> 这表示输出偏低，相关权重需要朝增大输出的方向更新。",
      params: (lr) => [
        [propagationParamDefs.out0.label, formatUpdate(propagationParamDefs.out0.old, propagationParamDefs.out0.grad, lr)],
        [propagationParamDefs.bout.label, formatUpdate(propagationParamDefs.bout.old, propagationParamDefs.bout.grad, lr)],
      ],
    },
    {
      node: ["hidden-0", "hidden-1", "hidden-2"],
      secondaryNode: ["input-0", "input-1", "output-0"],
      edge: ["w100", "w101", "w111", "w112", "w200", "w210", "w220"],
      title: "反向传播：隐藏层接收梯度信号",
      explain: "每个隐藏节点会收到来自后面层的梯度，知道自己对最终错误贡献了多少，再继续往前传。",
      pulse: { x: 48, y: 48 },
      formulaMain: `${link("delta_l", ["hidden-0", "hidden-1", "hidden-2"])} = (${link("W_(l+1)^T", ["w200", "w210", "w220"])} ${link("delta_(l+1)", ["output-0"])}) odot sigma'(${link("z_l", ["hidden-0", "hidden-1", "hidden-2"])})`,
      formulaSub: "链式法则把后层误差信号传回当前层",
      formulaExplain: "这一步是反向传播最核心的精神：后一层的误差，经过当前层的局部导数修正后，继续往前流。",
      chainMain: `${link("dL/dw100", ["w100"])} = ${link("dL/dout", ["output-0"])} x ${link("dout/dnet10", ["hidden-0"])} x ${link("dnet10/dw100", ["w100", "input-0"])}`,
      chainSub: `= ${link("delta_10", ["hidden-0"])} x ${link("x0", ["input-0"])}`,
      chainExplain: "对隐藏层权重来说，梯度不再只看输出误差，而是要先把输出误差一路传回当前隐藏节点。",
      values: "隐藏表示 h = [0.42, 0.73, 0.18] 不变，但它们现在各自收到了梯度。",
      grads: "dL/dh = [-0.08, 0.03, 0.11] -> 说明不同隐藏节点对最终误差的责任大小不一样。",
      params: (lr) => [
        [propagationParamDefs.hidden0.label, formatUpdate(propagationParamDefs.hidden0.old, propagationParamDefs.hidden0.grad, lr)],
        [propagationParamDefs.hidden1.label, formatUpdate(propagationParamDefs.hidden1.old, propagationParamDefs.hidden1.grad, lr)],
      ],
    },
    {
      node: ["input-0", "input-1"],
      secondaryNode: ["hidden-0", "hidden-1", "hidden-2"],
      edge: ["w100", "w101", "w111", "w112"],
      title: "反向传播：更早层据此调整参数",
      explain: "真正更新的是层间权重和偏置。越靠前的层，需要等梯度一路传回来，才能知道自己该怎么改。",
      pulse: { x: 12, y: 55 },
      formulaMain: `${link("W_new", ["w100", "w101", "w111", "w112", "w200", "w210", "w220"])} = ${link("W_old", ["w100", "w101", "w111", "w112", "w200", "w210", "w220"])} - ${link("eta", [])} x ${link("dL/dW", ["w100", "w101", "w111", "w112", "w200", "w210", "w220"])}`,
      formulaSub: `${link("b_new", ["hidden-0", "hidden-1", "hidden-2", "output-0"])} = ${link("b_old", ["hidden-0", "hidden-1", "hidden-2", "output-0"])} - ${link("eta", [])} x ${link("dL/db", ["hidden-0", "hidden-1", "hidden-2", "output-0"])}`,
      formulaExplain: "最后真正落到参数更新规则上。这里的 eta 就是 learning rate，它直接决定一次更新走多远。",
      chainMain: "eta 越大，W_new 变化越大",
      chainSub: "eta 越小，训练更稳但更慢",
      chainExplain: "这也是为什么学习率是训练里最关键的超参数之一：它直接决定了你每一步迈多大。",
      values: "输入本身通常不更新，但与输入相连的早期权重会更新。",
      grads: "dL/dx = [0.04, -0.02] -> 这不是为了改输入，而是为了继续把责任链条往更早层推。",
      params: (lr) => [
        ["w(input->hidden)", `学习率 = ${lr.toFixed(2)}，所以每次更新幅度 = lr x gradient`],
        ["更新规则", "new_w = old_w - learning_rate x gradient"],
      ],
    },
  ],
}

const propagationTitle = document.querySelector("[data-prop-title]")
const propagationExplain = document.querySelector("[data-prop-explain]")
const propagationValues = document.querySelector("[data-prop-values]")
const propagationGrads = document.querySelector("[data-prop-grads]")
const propagationParams = document.querySelector("[data-prop-params]")
const propagationFormulaMain = document.querySelector("[data-prop-formula-main]")
const propagationFormulaSub = document.querySelector("[data-prop-formula-sub]")
const propagationFormulaExplain = document.querySelector("[data-prop-formula-explain]")
const propagationChainMain = document.querySelector("[data-prop-chain-main]")
const propagationChainSub = document.querySelector("[data-prop-chain-sub]")
const propagationChainExplain = document.querySelector("[data-prop-chain-explain]")
const propagationLrRead = document.querySelector('[data-prop-read="lr"]')
const propagationLrNote = document.querySelector('[data-prop-lr-note]')
const propagationStatus = document.querySelector('[data-prop-status]')
const networkPulse = document.querySelector("[data-network-pulse]")
let propagationTimer = null
let edgePulseTimers = []
let propagationMode = "forward"
let propagationStepIndex = -1

function clearFormulaLinks() {
  document.querySelectorAll(".formula-link").forEach((linkNode) => {
    linkNode.classList.remove("is-hovered")
  })
  document.querySelectorAll("[data-node]").forEach((node) => node.classList.remove("is-linked"))
  document.querySelectorAll("[data-edge]").forEach((edge) => edge.classList.remove("is-linked"))
}

function bindFormulaLinks() {
  clearFormulaLinks()
  document.querySelectorAll(".formula-link").forEach((linkNode) => {
    const apply = (active) => {
      const targets = (linkNode.dataset.linkTargets || "").split(",").map((item) => item.trim()).filter(Boolean)
      linkNode.classList.toggle("is-hovered", active)
      targets.forEach((target) => {
        const node = document.querySelector(`[data-node="${target}"]`)
        if (node) node.classList.toggle("is-linked", active)
        const edge = document.querySelector(`[data-edge="${target}"]`)
        if (edge) edge.classList.toggle("is-linked", active)
      })
    }
    linkNode.onmouseenter = () => apply(true)
    linkNode.onmouseleave = () => apply(false)
  })
}

function clearEdgePulses() {
  edgePulseTimers.forEach((timer) => clearTimeout(timer))
  edgePulseTimers = []
  document.querySelectorAll("[data-edge]").forEach((edge) => {
    edge.classList.remove("is-pulse")
  })
}

function pulseEdgesSequentially(edgeNames) {
  clearEdgePulses()
  edgeNames.forEach((name, index) => {
    const startTimer = setTimeout(() => {
      const edge = document.querySelector(`[data-edge="${name}"]`)
      if (!edge) return
      edge.classList.remove("is-pulse")
      void edge.offsetWidth
      edge.classList.add("is-pulse")
      const endTimer = setTimeout(() => edge.classList.remove("is-pulse"), 720)
      edgePulseTimers.push(endTimer)
    }, index * 180)
    edgePulseTimers.push(startTimer)
  })
}

function renderPropagationStatic() {
  propagationLrRead.textContent = propagationState.learningRate.toFixed(2)
  propagationLrNote.textContent =
    `当前 learning rate = ${propagationState.learningRate.toFixed(2)}。它决定每一步参数改多少：越大改得越猛，越小改得越保守。`
}

function resetPropagation() {
  if (propagationTimer) {
    clearTimeout(propagationTimer)
    propagationTimer = null
  }
  clearEdgePulses()
  document.querySelectorAll("[data-node]").forEach((node) => {
    node.classList.remove("is-active-forward", "is-active-backward", "is-secondary-forward", "is-secondary-backward")
  })
  document.querySelectorAll("[data-edge]").forEach((edge) => {
    edge.classList.remove("is-active-forward", "is-active-backward")
  })
  networkPulse.classList.remove("is-visible", "is-backward")
  propagationTitle.textContent = "还没有播放动画"
  propagationExplain.textContent = "点击上面的按钮，观察“数值”如何向前流动，以及“误差信号”如何向后流动。"
  propagationFormulaMain.innerHTML = "z = W x + b"
  propagationFormulaSub.innerHTML = "a = sigma(z)"
  propagationFormulaExplain.textContent = "前向传播时，最常见的模式是先做线性变换，再过激活函数。反向传播时，则会切到梯度链式法则。"
  propagationChainMain.innerHTML = "dL/dW = dL/dy x dy/dz x dz/dW"
  propagationChainSub.innerHTML = "还没进入反向传播时，这里会显示“当前没有梯度拆解”。"
  propagationChainExplain.textContent = "链式法则的意思不是“突然多出很多公式”，而是把“总误差对某个参数的影响”拆成一段一段局部影响，再连乘起来。"
  propagationValues.textContent = "x = [1.00, 0.50] -> h = [0.42, 0.73, 0.18] -> y = [0.81]"
  propagationGrads.textContent = "dL/dy = [-0.19] -> dL/dh = [-0.08, 0.03, 0.11] -> dL/dx = [0.04, -0.02]"
  propagationParams.innerHTML = `
    <div class="change-item"><strong>w(hidden,0)</strong><span>0.80 -> 0.76</span></div>
    <div class="change-item"><strong>w(out,0)</strong><span>1.10 -> 1.06</span></div>
  `
  propagationStatus.textContent = `当前模式：${propagationMode === "forward" ? "前向传播" : "反向传播"} / Step ${Math.max(0, propagationStepIndex + 1)}`
  renderPropagationStatic()
  bindFormulaLinks()
}

function showPropagationStep() {
  const steps = propagationSteps[propagationMode]
  propagationStatus.textContent = `当前模式：${propagationMode === "forward" ? "前向传播" : "反向传播"} / Step ${Math.max(0, propagationStepIndex + 1)}`
  if (propagationStepIndex < 0 || propagationStepIndex >= steps.length) {
    clearEdgePulses()
    document.querySelectorAll("[data-node]").forEach((node) => {
      node.classList.remove("is-active-forward", "is-active-backward", "is-secondary-forward", "is-secondary-backward")
    })
    document.querySelectorAll("[data-edge]").forEach((edge) => {
      edge.classList.remove("is-active-forward", "is-active-backward")
    })
    networkPulse.classList.remove("is-visible", "is-backward")
    bindFormulaLinks()
    return
  }

  const step = steps[propagationStepIndex]
  const activeClass = propagationMode === "forward" ? "is-active-forward" : "is-active-backward"
  clearEdgePulses()
  document.querySelectorAll("[data-node]").forEach((node) => {
    node.classList.remove("is-active-forward", "is-active-backward", "is-secondary-forward", "is-secondary-backward")
  })
  document.querySelectorAll("[data-edge]").forEach((edge) => {
    edge.classList.remove("is-active-forward", "is-active-backward")
  })
  step.node.forEach((name) => {
    const node = document.querySelector(`[data-node="${name}"]`)
    if (node) node.classList.add(activeClass)
  })
  step.secondaryNode.forEach((name) => {
    const node = document.querySelector(`[data-node="${name}"]`)
    if (node) {
      node.classList.add(propagationMode === "forward" ? "is-secondary-forward" : "is-secondary-backward")
    }
  })
  step.edge.forEach((name) => {
    const edge = document.querySelector(`[data-edge="${name}"]`)
    if (edge) edge.classList.add(activeClass)
  })
  pulseEdgesSequentially(step.edge)

  networkPulse.classList.add("is-visible")
  networkPulse.classList.toggle("is-backward", propagationMode === "backward")
  networkPulse.style.transform = `translate(${step.pulse.x}%, ${step.pulse.y}%)`
  propagationTitle.textContent = step.title
  propagationExplain.textContent = step.explain
  propagationFormulaMain.innerHTML = step.formulaMain
  propagationFormulaSub.innerHTML = step.formulaSub
  propagationFormulaExplain.textContent = step.formulaExplain
  propagationChainMain.innerHTML = step.chainMain
  propagationChainSub.innerHTML = step.chainSub
  propagationChainExplain.textContent = step.chainExplain
  propagationValues.textContent = step.values
  propagationGrads.textContent = step.grads
  propagationParams.innerHTML = step.params(propagationState.learningRate)
    .map(([name, value]) => `<div class="change-item"><strong>${name}</strong><span>${value}</span></div>`)
    .join("")
  bindFormulaLinks()
}

document.querySelectorAll("[data-prop-action]").forEach((button) => {
  button.addEventListener("click", () => {
    const action = button.dataset.propAction
    if (action === "reset") {
      propagationStepIndex = -1
      resetPropagation()
      return
    }
    if (action === "set-forward") {
      propagationMode = "forward"
      propagationStepIndex = -1
      resetPropagation()
      return
    }
    if (action === "set-backward") {
      propagationMode = "backward"
      propagationStepIndex = -1
      resetPropagation()
      return
    }
    const steps = propagationSteps[propagationMode]
    if (action === "next") {
      propagationStepIndex = Math.min(steps.length - 1, propagationStepIndex + 1)
      showPropagationStep()
      return
    }
    if (action === "prev") {
      propagationStepIndex = Math.max(-1, propagationStepIndex - 1)
      if (propagationStepIndex === -1) {
        resetPropagation()
      } else {
        showPropagationStep()
      }
    }
  })
})

document.querySelector('[data-prop-control="lr"]').addEventListener("input", (event) => {
  propagationState.learningRate = Number(event.target.value)
  renderPropagationStatic()
})

resetPropagation()

const attentionTokens = ["The", "curious", "robot", "studied", "attention"]
const tokenVectors = {
  The: [0.1, 0.2, 0.1],
  curious: [0.9, 0.3, 0.2],
  robot: [0.8, 0.9, 0.4],
  studied: [0.3, 0.8, 0.9],
  attention: [0.4, 0.7, 1.0],
}

const attentionState = {
  queryIndex: 4,
  temperature: 1,
  causal: false,
  headIndex: 0,
}

const tokenPicker = document.querySelector("[data-token-picker]")
const sentenceStrip = document.querySelector("[data-sentence-strip]")
const scoreMatrix = document.querySelector("[data-score-matrix]")
const softmaxBars = document.querySelector("[data-softmax-bars]")
const attentionResult = document.querySelector("[data-attention-result]")
const attentionExplain = document.querySelector("[data-attention-explain]")
const attentionFormulaMain = document.querySelector("[data-attention-formula-main]")
const attentionFormulaSub = document.querySelector("[data-attention-formula-sub]")
const headPicker = document.querySelector("[data-head-picker]")
const headHeatmaps = document.querySelector("[data-head-heatmaps]")
const headSummary = document.querySelector("[data-head-summary]")

const multiHeadPatterns = [
  {
    name: "Head 0 / 语义主体",
    summary: "这个头更像在找“句子里谁是当前动作的主要参与者”。它会更偏向 robot、attention 这样语义更重的词。",
    weight: [
      [0.42, 0.18, 0.14, 0.14, 0.12],
      [0.12, 0.33, 0.29, 0.15, 0.11],
      [0.08, 0.13, 0.44, 0.21, 0.14],
      [0.07, 0.09, 0.31, 0.35, 0.18],
      [0.04, 0.08, 0.25, 0.18, 0.45],
    ],
  },
  {
    name: "Head 1 / 动作关系",
    summary: "这个头更像在找“谁和谁发生了动作关系”。它会让 studied 更明显地看向 robot 和 attention。",
    weight: [
      [0.35, 0.2, 0.18, 0.14, 0.13],
      [0.1, 0.25, 0.23, 0.22, 0.2],
      [0.07, 0.11, 0.29, 0.32, 0.21],
      [0.05, 0.06, 0.34, 0.19, 0.36],
      [0.04, 0.06, 0.18, 0.28, 0.44],
    ],
  },
  {
    name: "Head 2 / 局部位置",
    summary: "这个头更像在学局部邻近关系。它通常会更关注自己和相邻 token，类似一种软性的局部窗口。",
    weight: [
      [0.65, 0.2, 0.1, 0.03, 0.02],
      [0.22, 0.4, 0.25, 0.08, 0.05],
      [0.08, 0.24, 0.36, 0.23, 0.09],
      [0.04, 0.08, 0.28, 0.38, 0.22],
      [0.03, 0.05, 0.12, 0.28, 0.52],
    ],
  },
]

function dot(a, b) {
  return a.reduce((sum, value, idx) => sum + value * b[idx], 0)
}

function softmax(values) {
  const maxValue = Math.max(...values)
  const exps = values.map((value) => Math.exp(value - maxValue))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map((value) => value / sum)
}

function linkAttention(text, targets) {
  return `<span class="formula-link" data-att-link-targets="${targets.join(",")}">${text}</span>`
}

function clearAttentionLinks() {
  document.querySelectorAll(".formula-link[data-att-link-targets]").forEach((node) => {
    node.classList.remove("is-hovered")
  })
  document.querySelectorAll(".token-chip, .matrix__cell, .bars__item, .vector-box").forEach((node) => {
    node.classList.remove("is-linked")
  })
}

function bindAttentionLinks() {
  clearAttentionLinks()
  document.querySelectorAll(".formula-link[data-att-link-targets]").forEach((node) => {
    const apply = (active) => {
      const targets = (node.dataset.attLinkTargets || "").split(",").map((item) => item.trim()).filter(Boolean)
      node.classList.toggle("is-hovered", active)
      targets.forEach((target) => {
        const el = document.querySelector(`[data-att-link="${target}"]`)
        if (el) el.classList.toggle("is-linked", active)
      })
    }
    node.onmouseenter = () => apply(true)
    node.onmouseleave = () => apply(false)
  })
}

function renderAttentionLab() {
  tokenPicker.innerHTML = ""
  sentenceStrip.innerHTML = ""
  scoreMatrix.innerHTML = ""
  softmaxBars.innerHTML = ""

  attentionTokens.forEach((token, index) => {
    const button = document.createElement("button")
    button.type = "button"
    button.textContent = `${index}: ${token}`
    button.className = index === attentionState.queryIndex ? "is-active" : ""
    button.addEventListener("click", () => {
      attentionState.queryIndex = index
      renderAttentionLab()
    })
    tokenPicker.appendChild(button)

    const chip = document.createElement("div")
    chip.className = `token-chip ${index === attentionState.queryIndex ? "is-focus" : ""}`
    chip.textContent = token
    chip.dataset.attLink = `token-${index}`
    sentenceStrip.appendChild(chip)
  })

  const queryToken = attentionTokens[attentionState.queryIndex]
  const queryVector = tokenVectors[queryToken]
  const scores = attentionTokens.map((token, keyIndex) => {
    if (attentionState.causal && keyIndex > attentionState.queryIndex) {
      return -1e9
    }
    return dot(queryVector, tokenVectors[token]) / attentionState.temperature
  })
  const weights = softmax(scores)
  const resultVector = [0, 0, 0]

  attentionTokens.forEach((token, index) => {
    const scoreCell = document.createElement("div")
    scoreCell.className = "matrix__cell"
    scoreCell.dataset.attLink = `score-${index}`
    scoreCell.innerHTML = `<strong>${token}</strong><span>${scores[index] < -1e8 ? "masked" : scores[index].toFixed(3)}</span>`
    scoreMatrix.appendChild(scoreCell)

    const bar = document.createElement("div")
    bar.className = "bars__item"
    bar.dataset.attLink = `weight-${index}`
    bar.innerHTML = `<strong>${token}</strong><div>${weights[index].toFixed(3)}</div><div class="bars__track"><div class="bars__fill" style="width:${weights[index] * 100}%"></div></div>`
    softmaxBars.appendChild(bar)

    tokenVectors[token].forEach((value, dim) => {
      resultVector[dim] += value * weights[index]
    })
  })

  attentionResult.textContent = `[${resultVector.map((value) => value.toFixed(3)).join(", ")}]`
  attentionResult.dataset.attLink = "result"
  attentionExplain.textContent =
    `当前 query 是 “${queryToken}”。它先和所有 key 做点积得到 score，再经过 softmax 变成权重。权重大，说明这个 token 对当前 query 更重要。最后对所有 value 做加权求和，得到新的上下文表示。`

  attentionFormulaMain.innerHTML = `${linkAttention("Q", [`token-${attentionState.queryIndex}`])} · ${linkAttention("K", attentionTokens.map((_, index) => `token-${index}`))} / sqrt(d) -> ${linkAttention("score", attentionTokens.map((_, index) => `score-${index}`))}`
  attentionFormulaSub.innerHTML = `${linkAttention("weight = softmax(score)", attentionTokens.map((_, index) => `weight-${index}`))}, ${linkAttention("output = sum(weight x V)", [...attentionTokens.map((_, index) => `weight-${index}`), ...attentionTokens.map((_, index) => `token-${index}`), "result"])}`
  bindAttentionLinks()

  document.querySelector('[data-attention-read="temperature"]').textContent = attentionState.temperature.toFixed(1)
}

function renderMultiHeadLab() {
  headPicker.innerHTML = ""
  headHeatmaps.innerHTML = ""

  multiHeadPatterns.forEach((head, index) => {
    const button = document.createElement("button")
    button.type = "button"
    button.textContent = head.name
    button.className = index === attentionState.headIndex ? "is-active" : ""
    button.addEventListener("click", () => {
      attentionState.headIndex = index
      renderMultiHeadLab()
    })
    headPicker.appendChild(button)
  })

  multiHeadPatterns.forEach((head) => {
    const panel = document.createElement("article")
    panel.className = "heatmap"
    panel.innerHTML = `<h4>${head.name}</h4>`
    const grid = document.createElement("div")
    grid.className = "heatmap__grid"

    head.weight.forEach((row, rowIndex) => {
      const rowNode = document.createElement("div")
      rowNode.className = "heatmap__row"
      const label = document.createElement("div")
      label.className = "heatmap__label"
      label.textContent = attentionTokens[rowIndex]
      rowNode.appendChild(label)

      row.forEach((value) => {
        const cell = document.createElement("div")
        cell.className = "heatmap__cell"
        const alpha = 0.14 + value * 0.86
        cell.style.background = `rgba(106, 195, 255, ${alpha})`
        cell.style.color = value > 0.55 ? "#04111c" : "#f5fbff"
        cell.textContent = value.toFixed(2)
        rowNode.appendChild(cell)
      })

      grid.appendChild(rowNode)
    })

    panel.appendChild(grid)
    headHeatmaps.appendChild(panel)
  })

  const focusHead = multiHeadPatterns[attentionState.headIndex]
  headSummary.textContent = `${focusHead.name}: ${focusHead.summary}`
}

document.querySelector('[data-attention="temperature"]').addEventListener("input", (event) => {
  attentionState.temperature = Number(event.target.value)
  renderAttentionLab()
})

document.querySelector('[data-attention="causal"]').addEventListener("change", (event) => {
  attentionState.causal = event.target.checked
  renderAttentionLab()
})

renderAttentionLab()
renderMultiHeadLab()

const realAttentionFile = document.querySelector('[data-real-attention-file]')
const realAttentionLoad = document.querySelector('[data-real-attention-load]')
const realAttentionPrev = document.querySelector('[data-real-attention-prev]')
const realAttentionNext = document.querySelector('[data-real-attention-next]')
const realAttentionStep = document.querySelector('[data-real-attention-step]')
const realAttentionStatus = document.querySelector('[data-real-attention-status]')
const realAttentionContent = document.querySelector('[data-real-attention-content]')
const realAttentionState = {
  data: null,
  stepIndex: 0,
}

function renderRealAttention(data) {
  realAttentionContent.innerHTML = ""
  if (!data || !Array.isArray(data.tokens) || !Array.isArray(data.layers)) {
    realAttentionStatus.textContent = "attention JSON 结构不正确。"
    return
  }

  const stepData = Array.isArray(data.steps) && data.steps.length > 0 ? data.steps[realAttentionState.stepIndex] : null
  const tokens = stepData ? stepData.context_tokens : data.tokens
  const layers = stepData ? stepData.layers : data.layers

  realAttentionStatus.textContent = `已加载真实样例：backbone = ${data.backbone || "unknown"}，token 数 = ${tokens.length}`
  if (stepData) {
    realAttentionStep.textContent = `当前步骤：${realAttentionState.stepIndex + 1} / ${data.steps.length}，预测下一个 token = ${stepData.predicted_token}`
  } else {
    realAttentionStep.textContent = "当前展示最终整段 attention 权重"
  }

  layers.forEach((layer) => {
    const layerNode = document.createElement("section")
    layerNode.className = "real-attention__layer"
    layerNode.innerHTML = `<h3>Layer ${layer.layer_index}</h3>`
    const heatmaps = document.createElement("div")
    heatmaps.className = "head-heatmaps"

    layer.heads.forEach((head, headIndex) => {
      const panel = document.createElement("article")
      panel.className = "heatmap"
      panel.innerHTML = `<h4>Head ${headIndex}</h4>`
      const grid = document.createElement("div")
      grid.className = "heatmap__grid"

      head.forEach((row, rowIndex) => {
        const rowNode = document.createElement("div")
        rowNode.className = "heatmap__row"
        const label = document.createElement("div")
        label.className = "heatmap__label"
        label.textContent = tokens[rowIndex]
        rowNode.appendChild(label)

        row.forEach((value) => {
          const cell = document.createElement("div")
          cell.className = "heatmap__cell"
          const alpha = 0.14 + Number(value) * 0.86
          cell.style.background = `rgba(106, 195, 255, ${alpha})`
          cell.style.color = Number(value) > 0.55 ? "#04111c" : "#f5fbff"
          cell.textContent = Number(value).toFixed(2)
          rowNode.appendChild(cell)
        })

        grid.appendChild(rowNode)
      })

      panel.appendChild(grid)
      heatmaps.appendChild(panel)
    })

    layerNode.appendChild(heatmaps)
    realAttentionContent.appendChild(layerNode)
  })
}

function applyRealAttentionData(data) {
  realAttentionState.data = data
  realAttentionState.stepIndex = 0
  renderRealAttention(data)
}

realAttentionFile.addEventListener('change', async (event) => {
  const file = event.target.files?.[0]
  if (!file) return
  try {
    const text = await file.text()
    applyRealAttentionData(JSON.parse(text))
  } catch (error) {
    realAttentionStatus.textContent = `读取文件失败：${error.message}`
  }
})

realAttentionLoad.addEventListener('click', async () => {
  try {
    const response = await fetch('./attention-sample.json')
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    applyRealAttentionData(await response.json())
  } catch (error) {
    realAttentionStatus.textContent = `读取内置样例失败：${error.message}。如果你是直接 file:// 打开页面，请改用本地静态服务器或手动选择 JSON 文件。`
  }
})

realAttentionPrev.addEventListener('click', () => {
  if (!realAttentionState.data || !Array.isArray(realAttentionState.data.steps) || realAttentionState.data.steps.length === 0) {
    return
  }
  realAttentionState.stepIndex = Math.max(0, realAttentionState.stepIndex - 1)
  renderRealAttention(realAttentionState.data)
})

realAttentionNext.addEventListener('click', () => {
  if (!realAttentionState.data || !Array.isArray(realAttentionState.data.steps) || realAttentionState.data.steps.length === 0) {
    return
  }
  realAttentionState.stepIndex = Math.min(realAttentionState.data.steps.length - 1, realAttentionState.stepIndex + 1)
  renderRealAttention(realAttentionState.data)
})
