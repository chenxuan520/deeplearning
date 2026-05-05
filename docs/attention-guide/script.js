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

const propagationSteps = {
  forward: [
    {
      node: ["input-0", "input-1"],
      title: "前向传播：输入层接收原始特征",
      explain: "这一步传的不是误差，而是原始输入值。比如图片像素、表格特征或 token embedding，都会先进入输入层。",
      pulse: { x: 12, y: 55 },
      values: "x = [1.00, 0.50] -> h = [?, ?, ?] -> y = [?]",
      grads: "梯度还没有开始传播，当前阶段重点是把输入送进网络。",
      params: [
        ["w(hidden,0)", "0.80 (等待更新)"],
        ["w(out,0)", "1.10 (等待更新)"],
      ],
    },
    {
      node: ["hidden-0", "hidden-1", "hidden-2"],
      title: "前向传播：隐藏层做加权求和和激活",
      explain: "隐藏层把上一层输出乘权重、加偏置，再过激活函数。网络在这里逐步形成更抽象的特征表示。",
      pulse: { x: 48, y: 48 },
      values: "x = [1.00, 0.50] -> h = [0.42, 0.73, 0.18] -> y = [?]",
      grads: "这里仍然是数值流，不是误差信号。隐藏层的输出会作为下一层输入。",
      params: [
        ["z(hidden,0)", "1.00 x 0.80 + 0.50 x -0.20 + 0.10 -> 0.80"],
        ["a(hidden,0)", "sigmoid(0.80) -> 0.69"],
      ],
    },
    {
      node: ["output-0"],
      title: "前向传播：输出层给出预测",
      explain: "最后一层把隐藏特征压缩成预测结果。分类任务里可能是各类别得分，语言模型里则是下一个 token 的 logits。",
      pulse: { x: 84, y: 52 },
      values: "x = [1.00, 0.50] -> h = [0.42, 0.73, 0.18] -> y = [0.81]",
      grads: "前向阶段结束后，模型终于得到一个输出。接下来就可以用 loss 判断它到底偏了多少。",
      params: [
        ["y_pred", "[0.81]"],
        ["loss 前状态", "预测已得到，等待与目标比较"],
      ],
    },
  ],
  backward: [
    {
      node: ["output-0"],
      title: "反向传播：先从输出误差开始",
      explain: "反向传播不是把原始输入往回传，而是把“模型错了多少”变成梯度，从输出层往回走。",
      pulse: { x: 84, y: 52 },
      values: "预测 y = [0.81]，真实目标 t = [1.00]",
      grads: "dL/dy = y - t = -0.19 -> 这表示输出偏低，相关权重需要朝增大输出的方向更新。",
      params: [
        ["w(out,0)", "1.10 -> 1.06 (↓ 0.04)"],
        ["b(out)", "0.20 -> 0.24 (↑ 0.04)"],
      ],
    },
    {
      node: ["hidden-0", "hidden-1", "hidden-2"],
      title: "反向传播：隐藏层接收梯度信号",
      explain: "每个隐藏节点会收到来自后面层的梯度，知道自己对最终错误贡献了多少，再继续往前传。",
      pulse: { x: 48, y: 48 },
      values: "隐藏表示 h = [0.42, 0.73, 0.18] 不变，但它们现在各自收到了梯度。",
      grads: "dL/dh = [-0.08, 0.03, 0.11] -> 说明不同隐藏节点对最终误差的责任大小不一样。",
      params: [
        ["w(hidden,0)", "0.80 -> 0.76 (↓ 0.04)"],
        ["w(hidden,1)", "-0.20 -> -0.18 (↑ 0.02)"],
      ],
    },
    {
      node: ["input-0", "input-1"],
      title: "反向传播：更早层据此调整参数",
      explain: "真正更新的是层间权重和偏置。越靠前的层，需要等梯度一路传回来，才能知道自己该怎么改。",
      pulse: { x: 12, y: 55 },
      values: "输入本身通常不更新，但与输入相连的早期权重会更新。",
      grads: "dL/dx = [0.04, -0.02] -> 这不是为了改输入，而是为了继续把责任链条往更早层推。",
      params: [
        ["w(input->hidden)", "根据 dL/dW 做微调"],
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
const networkPulse = document.querySelector("[data-network-pulse]")
let propagationTimer = null

function resetPropagation() {
  if (propagationTimer) {
    clearTimeout(propagationTimer)
    propagationTimer = null
  }
  document.querySelectorAll("[data-node]").forEach((node) => {
    node.classList.remove("is-active-forward", "is-active-backward")
  })
  networkPulse.classList.remove("is-visible", "is-backward")
  propagationTitle.textContent = "还没有播放动画"
  propagationExplain.textContent = "点击上面的按钮，观察“数值”如何向前流动，以及“误差信号”如何向后流动。"
  propagationValues.textContent = "x = [1.00, 0.50] -> h = [0.42, 0.73, 0.18] -> y = [0.81]"
  propagationGrads.textContent = "dL/dy = [-0.19] -> dL/dh = [-0.08, 0.03, 0.11] -> dL/dx = [0.04, -0.02]"
  propagationParams.innerHTML = `
    <div class="change-item"><strong>w(hidden,0)</strong><span>0.80 -> 0.76</span></div>
    <div class="change-item"><strong>w(out,0)</strong><span>1.10 -> 1.06</span></div>
  `
}

function playPropagation(direction) {
  resetPropagation()
  const steps = propagationSteps[direction]
  const activeClass = direction === "forward" ? "is-active-forward" : "is-active-backward"
  networkPulse.classList.toggle("is-backward", direction === "backward")

  const runStep = (index) => {
    if (index >= steps.length) {
      propagationTimer = setTimeout(() => {
        document.querySelectorAll("[data-node]").forEach((node) => {
          node.classList.remove(activeClass)
        })
        networkPulse.classList.remove("is-visible", "is-backward")
      }, 800)
      return
    }

    const step = steps[index]
    document.querySelectorAll("[data-node]").forEach((node) => {
      node.classList.remove("is-active-forward", "is-active-backward")
    })
    step.node.forEach((name) => {
      const node = document.querySelector(`[data-node="${name}"]`)
      if (node) node.classList.add(activeClass)
    })

    networkPulse.classList.add("is-visible")
    networkPulse.style.transform = `translate(${step.pulse.x}%, ${step.pulse.y}%)`
    propagationTitle.textContent = step.title
    propagationExplain.textContent = step.explain
    propagationValues.textContent = step.values
    propagationGrads.textContent = step.grads
    propagationParams.innerHTML = step.params
      .map(([name, value]) => `<div class="change-item"><strong>${name}</strong><span>${value}</span></div>`)
      .join("")

    propagationTimer = setTimeout(() => runStep(index + 1), 900)
  }

  runStep(0)
}

document.querySelectorAll("[data-prop-action]").forEach((button) => {
  button.addEventListener("click", () => {
    const action = button.dataset.propAction
    if (action === "reset") {
      resetPropagation()
      return
    }
    playPropagation(action)
  })
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
    scoreCell.innerHTML = `<strong>${token}</strong><span>${scores[index] < -1e8 ? "masked" : scores[index].toFixed(3)}</span>`
    scoreMatrix.appendChild(scoreCell)

    const bar = document.createElement("div")
    bar.className = "bars__item"
    bar.innerHTML = `<strong>${token}</strong><div>${weights[index].toFixed(3)}</div><div class="bars__track"><div class="bars__fill" style="width:${weights[index] * 100}%"></div></div>`
    softmaxBars.appendChild(bar)

    tokenVectors[token].forEach((value, dim) => {
      resultVector[dim] += value * weights[index]
    })
  })

  attentionResult.textContent = `[${resultVector.map((value) => value.toFixed(3)).join(", ")}]`
  attentionExplain.textContent =
    `当前 query 是 “${queryToken}”。它先和所有 key 做点积得到 score，再经过 softmax 变成权重。权重大，说明这个 token 对当前 query 更重要。最后对所有 value 做加权求和，得到新的上下文表示。`

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
