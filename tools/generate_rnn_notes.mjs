import { generateTopic } from "./course_note_helpers.mjs";

const article = (sections) => sections.map(([time, title, body]) =>
  `### ${time}　${title}\n\n${body}`
).join("\n\n");

const recurrentShapeDiagram = {
  title: "PyTorch 循环层的张量形状",
  diagram: `
flowchart LR
    X["input<br/>[batch, seq, input_size]"] --> R["RNN / LSTM / GRU"]
    H0["h0<br/>[layers×directions, batch, hidden]"] --> R
    C0["仅 LSTM：c0<br/>形状同 h0"] -.-> R
    R --> O["output<br/>每个时间步<br/>[batch, seq, directions×hidden]"]
    R --> HN["hn<br/>每层最终状态"]
    R -.-> CN["仅 LSTM：cn"]
`,
};

const modelFamilyDiagram = {
  title: "RNN 家族 UML 关系",
  diagram: `
classDiagram
    class 循环序列模型 {
      input_size
      hidden_size
      num_layers
      forward(sequence, state)
    }
    class Vanilla_RNN {
      tanh隐藏状态
    }
    class LSTM {
      遗忘门
      输入门
      输出门
      细胞状态
    }
    class GRU {
      重置门
      更新门
    }
    循环序列模型 <|-- Vanilla_RNN
    循环序列模型 <|-- LSTM
    循环序列模型 <|-- GRU
`,
};

const projectPipelineDiagram = {
  title: "姓名分类项目完整流水线",
  diagram: `
flowchart LR
    A["CSV：name,country"] --> B["字符规范化"]
    B --> C["每字符 57 维 One-Hot"]
    C --> D["Dataset"]
    D --> E["DataLoader"]
    E --> F["RNN / LSTM / GRU"]
    F --> G["18 类 log-probabilities"]
    G --> H["NLLLoss + 反向传播"]
    H --> I["保存权重 / 比较曲线 / Top-k 预测"]
`,
};

const trainingSequenceDiagram = {
  title: "一批数据的训练时序",
  diagram: `
sequenceDiagram
    participant D as DataLoader
    participant M as 循环模型
    participant L as 损失函数
    participant O as 优化器
    D->>M: x 与 label
    M->>L: log_probs 与 label
    O->>O: zero_grad()
    L->>M: backward()
    O->>M: step() 更新参数
`,
};

const lessons = [
  {
    p: 38, slug: "rnn-introduction", title: "RNN 简介：让当前判断带上过去的信息",
    problem: "普通全连接网络看不到词序；RNN 如何一边读序列，一边把前文压进隐藏状态？",
    chain: ["当前 token x_t", "旧状态 h_(t-1)", "共享 RNN 单元", "新状态 h_t", "序列任务输出"],
    extraVisuals: [recurrentShapeDiagram],
    article: article([
      ["0:00–1:57", "本章路线与 RNN 的位置", "老师先给出路线：RNN、LSTM、GRU → 全球姓名分类 → 注意力 → Seq2Seq 英译法，随后才进入 Transformer。这里的意义是先理解“按时间一步步处理”，后面才能理解注意力为何要摆脱串行瓶颈。"],
      ["1:57–5:55", "输入、循环状态与输出", "文本先数值化；循环层在第 t 步同时接收当前输入 x_t 和上一隐藏状态 h_(t-1)，产生 h_t。老师反复指着展开图说明：图上画了多个方框，但它们是同一个单元在不同时间重复使用，同一组权重被共享。"],
      ["5:55–8:51", "展开图怎样读", "h_0 通常初始化为零。读入 x_1 得到 h_1，再把 h_1 带到下一步与 x_2 一起计算。每往后一步，状态就尝试携带更多前文信息。课程把每步概括成“两类输入、两类输出”；在 PyTorch 中要用张量形状理解，而不是误以为真有 12 个独立函数参数。"],
      ["8:51–13:48", "为什么适合语言与意图识别", "语言、语音、时间序列都具有先后依赖。老师以银行助手的意图识别为例：把一句问题逐词读入，最后用最终隐藏表示做多分类。剧情类比帮助理解：看到第 3 集时，脑中应带着前两集信息。"],
      ["13:48–15:25", "本节结论", "RNN 主要处理序列数据；结构仍是输入层、隐藏层、输出层，只是隐藏部分具有循环连接。它能记住多少并不只看维度，还受训练、序列长度和梯度衰减影响。"],
    ]),
    points: ["同一 RNN 单元沿时间共享权重", "h_t 同时依赖 x_t 与 h_(t-1)", "最终状态可用于整句分类"],
    code: `import torch\nfrom rnn_from_scratch.model import SimpleRNNCell\ncell = SimpleRNNCell(6, 8)\nx_t = torch.randn(4, 6)\nh = cell.initial_hidden(batch_size=4)\nprint(cell(x_t, h).shape)`,
    output: "4 个样本，每个当前输入 6 维；更新后的隐藏状态是 [4, 8]。",
    pit: "展开图里的多个 RNN 方框不是多套不同参数，而是同一单元在多个时间步复用。",
    quiz: "第 t 步最少需要哪两类信息？", answer: "当前输入 x_t 和上一隐藏状态 h_(t-1)。"
  },
  {
    p: 39, slug: "rnn-types", title: "RNN 分类：输入输出关系与内部结构是两个维度",
    problem: "“多对一”“多对多”和“LSTM/GRU”不是同一层级的分类，怎样把它们分清？",
    chain: ["先看输入个数", "再看输出个数", "确定任务形态", "再选内部单元", "组合成模型"],
    extraVisuals: [modelFamilyDiagram],
    article: article([
      ["0:00–3:55", "按输入与输出数量分类", "老师要求记住四种任务形态：N→N（等长序列标注或逐步生成）、N→1（文本分类/意图识别）、1→N（由单一条件生成序列，如图像描述的抽象形式）、N→M（翻译、摘要等输入输出长度不同的任务）。英译法属于 N→M。"],
      ["3:55–7:55", "按内部结构分类", "第二个维度才是传统 RNN、LSTM、GRU。LSTM 用遗忘门、输入门、输出门和细胞状态管理长期信息；GRU 用重置门与更新门做较简化的控制。Bi-LSTM/Bi-GRU 表示从正向和反向各读一次再合并。"],
      ["7:55–12:56", "为什么面试回答要先说分类维度", "只回答 N→M 或只回答 LSTM 都不完整。老师用翻译图预告编码器、解码器和中间表示 C；目前只需知道输入输出长度可以不同，后面的 Seq2Seq 会完整展开。"],
      ["12:56–14:14", "练习与收束", "判断题时先问“它在讨论任务接口，还是内部循环单元”。双向模型能利用两侧上下文，但计算和延迟更高，也不适合必须实时只看过去的因果生成。"],
    ]),
    points: ["任务形态和循环单元是两个正交维度", "N→1 常用于整句分类", "双向模型不能用于严格的在线因果场景"],
    code: `tasks = {"sentiment": "N→1", "translation": "N→M", "tagging": "N→N"}\nfor task, shape in tasks.items():\n    print(task, shape)`,
    output: "输出把典型任务映射到输入输出关系；之后仍需另外选择 RNN/LSTM/GRU。",
    pit: "Bi-LSTM 不是一种新的输入输出数量关系，而是内部状态流向的选择。",
    quiz: "姓名国籍分类属于哪种任务形态？", answer: "输入多个字符，输出一个国家类别，因此是 N→1。"
  },
  {
    p: 40, slug: "rnn-structure", title: "RNN 模型结构：公式、共享权重与张量形状",
    problem: "h_t 的公式看懂了，怎样把它对应到 PyTorch 的 input、output、h_n？",
    chain: ["序列张量", "时间步递推", "所有步 output", "最终 h_n", "下游分类层"],
    extraVisuals: [recurrentShapeDiagram],
    article: article([
      ["0:00–3:54", "单步公式", "老师从 h_t = tanh(W_x x_t + W_h h_(t-1) + b) 讲起。W_x 把当前输入投到隐藏空间，W_h 处理旧状态，二者相加再激活。权重在每个时间步相同。"],
      ["3:54–7:52", "output 与 h_n 不要混淆", "PyTorch 循环层返回 output 和 h_n。output 收集最后一层在所有时间步的输出；h_n 收集每一层（以及每个方向）在最终时间步的状态。单层单向时，output 的最后时间步和 h_n[0] 数值相同，但形状和语义组织不同。"],
      ["7:52–11:29", "三类核心维度", "input_size 是每个 token 的特征数，hidden_size 是状态维度，num_layers 是堆叠层数。若 batch_first=True，输入是 [batch, seq, input_size]；否则默认是 [seq, batch, input_size]。老师的课堂示意采用后一种顺序，因此阅读代码必须先检查 batch_first。"],
    ]),
    points: ["input_size 不等于序列长度", "output 含每个时间步，h_n 含最终状态", "batch_first 会改变前两维顺序"],
    code: `import torch\nrnn = torch.nn.RNN(5, 6, batch_first=True)\nx = torch.randn(2, 3, 5)\nout, hn = rnn(x)\nprint(out.shape, hn.shape)`,
    output: "output=[2,3,6]，h_n=[1,2,6]；单层单向所以第一维为 1。",
    pit: "不要只背“236、136”之类示例数字；应从 batch、seq、hidden 的含义推导。",
    quiz: "batch_first=True 时 [2,3,5] 分别是什么？", answer: "2 个样本、每个 3 个时间步、每步 5 维特征。"
  },
  {
    p: 41, slug: "rnn-basic-code", title: "RNN 基础代码：创建层、准备输入、运行并验形状",
    problem: "如何用 nn.RNN 跑通一次前向传播，并用形状证明每个参数理解正确？",
    chain: ["nn.RNN 配置", "input 张量", "h0 张量", "前向调用", "打印 output/hn"],
    extraVisuals: [recurrentShapeDiagram],
    article: article([
      ["0:00–5:51", "建立脚本与解释参数", "老师先写 RNN 概念，再导入 torch/nn。创建 nn.RNN 时依次解释 input_size、hidden_size、num_layers；后两个维度不是“词数”和“类别数”。"],
      ["5:51–10:48", "准备 input 与 h0", "输入张量要同时包含序列长度、批大小和每步特征维度；h0 的形状必须是 [num_layers×directions, batch, hidden_size]。课堂用随机数只是验证接口，不能代表真实文本编码。"],
      ["10:48–15:44", "调用与两个返回值", "执行 output, h_n = rnn(input, h0)。老师逐维核对 output 保存所有时间步，h_n 保存每层最终隐藏状态，并让同学记住单层时 output[-1] 与 h_n[-1] 的关系。"],
      ["15:49–20:25", "从死记数字转向推导", "课堂反复修改张量并打印形状。真正要掌握的是推导规则：改 batch，output/h_n 的 batch 维一起变；改 seq，只改 output 的时间维；改 hidden_size，二者最后一维一起变。"],
    ]),
    points: ["h0 也必须匹配层数、批量和隐藏维度", "随机张量只用于接口测试", "修改一个变量前先预测哪些维度会变"],
    code: `import torch\nrnn = torch.nn.RNN(5, 6)\nx = torch.randn(3, 2, 5)\nh0 = torch.zeros(1, 2, 6)\nout, hn = rnn(x, h0)\nprint(out.shape, hn.shape)`,
    output: "默认非 batch_first：output=[3,2,6]，h_n=[1,2,6]。",
    pit: "h0 的第一维是层数×方向数，不是序列长度。",
    quiz: "把序列长度 3 改为 7，h_n 的形状会变吗？", answer: "不会；output 时间维变为 7，h_n 仍只保留最终状态。"
  },
  {
    p: 42, slug: "change-sequence-length", title: "修改句长：只应改变 output 的时间维",
    problem: "序列从 3 步改为更多步时，哪些张量维度应该变化？",
    chain: ["改 seq_len", "重建 input", "复用 RNN", "比较 output", "确认 hn 不带时间维"],
    extraVisuals: [recurrentShapeDiagram],
    article: article([
      ["0:00–1:53", "只改一个变量做对照", "老师把输入的句子长度改长，其他 input_size、hidden_size、batch 和层数不动，然后要求运行前先猜形状。"],
      ["1:53–3:26", "观察结果", "output 要为每个时间步保留一个隐藏输出，所以时间维随句长改变；h_n 只保存各层最终状态，不保留整条时间轴，因此形状不变。这个小实验是后面“取最后一步做分类”的基础。"],
    ]),
    points: ["seq_len 决定 output 的时间维", "h_n 不包含完整时间轴", "控制变量实验比背结果可靠"],
    code: `import torch\nrnn = torch.nn.RNN(5, 6, batch_first=True)\nfor length in (3, 7):\n    out, hn = rnn(torch.randn(2, length, 5))\n    print(length, out.shape, hn.shape)`,
    output: "output 从 [2,3,6] 变 [2,7,6]；h_n 始终 [1,2,6]。",
    pit: "真实 batch 中句长不同还需 padding/packing；这里只是所有样本同时改长。",
    quiz: "为什么 h_n 不需要 seq_len 这一维？", answer: "它定义上就是每层、每方向的最终状态，而不是所有时间步。"
  },
  {
    p: 43, slug: "change-hidden-size", title: "修改隐藏层与总结：维度代表模型的记忆容量",
    problem: "hidden_size 变大后，模型容量、输出形状和计算量如何一起变化？",
    chain: ["改 hidden_size", "重建循环层", "重建 h0", "比较形状", "权衡容量与成本"],
    extraVisuals: [recurrentShapeDiagram],
    article: article([
      ["0:00–4:49", "修改 hidden_size", "老师把隐藏维度改为新数值，并同步修改 h0。output 与 h_n 的最后一维随之改变；input 最后一维仍由 input_size 决定。"],
      ["4:49–9:44", "层数与批量也按公式推导", "num_layers 增加会让 h_n 第一维增加，output 仍只暴露最后一层；batch 增加会同时改变 input、output、h_n 的批量维。课堂围绕多个数字例子反复核对。"],
      ["9:44–16:35", "容量不是越大越好", "隐藏维度大能容纳更复杂模式，也增加参数、显存、训练时间和过拟合风险。老师最后回扣 input、output、h_n 的关系。实际选型应看验证集，而不是把“记得更多”简单等同于更准确。"],
    ]),
    points: ["hidden_size 控制循环状态宽度", "改层配置后 h0 必须同步", "容量增加伴随计算与过拟合成本"],
    code: `import torch\nfor hidden in (4, 12):\n    rnn = torch.nn.RNN(5, hidden, num_layers=2, batch_first=True)\n    out, hn = rnn(torch.randn(3, 7, 5))\n    print(hidden, out.shape, hn.shape)`,
    output: "hidden=4 时 output [3,7,4]、hn [2,3,4]；hidden=12 时末维都变 12。",
    pit: "旧 h0 不能拿给 hidden_size 已变化的新模型。",
    quiz: "两层单向 RNN 的 h_n 第一维为什么是 2？", answer: "每层各保留一个最终隐藏状态。"
  },
  {
    p: 44, slug: "lstm-diagram-part1", title: "LSTM 图解（上）：遗忘门与输入门管理长期记忆",
    problem: "普通 RNN 的长期信息容易衰减，LSTM 怎样决定忘掉什么、写入什么？",
    chain: ["旧细胞状态 C_(t-1)", "遗忘门 f_t", "输入门 i_t", "候选记忆 C~_t", "新细胞状态 C_t"],
    extraVisuals: [modelFamilyDiagram],
    article: article([
      ["0:00–4:51", "为什么需要长期记忆通道", "老师从普通 RNN 的梯度消失/爆炸讲起。LSTM 引入细胞状态 C，像一条相对稳定的记忆通道；门值经过 sigmoid 落在 0 到 1，用逐元素乘法控制信息比例。"],
      ["4:51–10:57", "遗忘门", "把 x_t 与 h_(t-1) 拼接，经线性层和 sigmoid 得 f_t。f_t 接近 1 的位置更多保留旧记忆，接近 0 的位置更多擦除。它不是整句话只开或关一次，而是对每个隐藏维度分别控制。"],
      ["10:57–16:51", "输入门与候选记忆", "输入门 i_t 决定写入比例；候选记忆 C~_t 通常由 tanh 产生新内容。二者相乘表示“哪些新信息写多少”。"],
      ["16:51–20:57", "更新细胞状态", "C_t = f_t⊙C_(t-1) + i_t⊙C~_t。老师用生活中的保留旧信息、加入新信息解释加法。LSTM 缓解长期依赖，但不能保证彻底消除所有梯度问题。"],
    ]),
    points: ["门是逐元素比例，不是单个布尔开关", "遗忘门处理旧记忆，输入门控制新记忆", "细胞状态通过加法更新"],
    code: `import torch\nx = torch.randn(2, 3, 5)\nlstm = torch.nn.LSTM(5, 7, batch_first=True)\nout, (hn, cn) = lstm(x)\nprint(out.shape, hn.shape, cn.shape)`,
    output: "output=[2,3,7]，h_n 与 c_n 都是 [1,2,7]。",
    pit: "不要把细胞状态叫“记忆门”；LSTM 是三扇门加一个细胞状态。",
    quiz: "f_t 接近 0 意味着什么？", answer: "对应维度的旧细胞记忆被大幅削弱。"
  },
  {
    p: 45, slug: "lstm-diagram-part2", title: "LSTM 图解（下）：输出门产生当前隐藏状态",
    problem: "细胞状态更新后，LSTM 怎样决定当前对外输出哪些信息？",
    chain: ["得到 C_t", "tanh 压缩", "输出门 o_t", "逐元素筛选", "得到 h_t"],
    extraVisuals: [modelFamilyDiagram],
    article: article([
      ["0:00–5:50", "接上半节的细胞状态", "老师先复盘遗忘门、输入门和候选记忆，确认 C_t 已经融合旧记忆与新内容。"],
      ["5:50–11:40", "输出门", "o_t = sigmoid(W_o[x_t,h_(t-1)] + b_o)，决定当前隐藏状态暴露哪些细胞信息。h_t = o_t⊙tanh(C_t)。C_t 是内部长期记忆，h_t 是当前时间步向下一层/下一步提供的表示。"],
      ["11:40–18:34", "三门一状态完整串联", "先算遗忘比例，再算写入比例和候选内容，更新 C_t，最后由输出门产生 h_t。课堂反复让同学用大白话描述每个式子，而不是背字母。"],
      ["18:34–26:10", "优点与局限", "LSTM 对长依赖通常优于普通 RNN，但门多、参数和计算更大，仍然按时间串行。老师预告 GRU 会合并部分结构。实际模型效果要由任务数据验证。"],
    ]),
    points: ["C_t 和 h_t 作用不同", "输出门控制细胞记忆对外暴露比例", "LSTM 仍无法跨时间步完全并行"],
    code: `import torch\ncell = torch.nn.LSTMCell(5, 7)\nx_t = torch.randn(2, 5)\nh = torch.zeros(2, 7); c = torch.zeros(2, 7)\nh, c = cell(x_t, (h, c))\nprint(h.shape, c.shape)`,
    output: "单步产生同形状 h_t 和 c_t，都是 [batch, hidden]。",
    pit: "h_t 不是 C_t 的原样复制；还经过 tanh 和输出门筛选。",
    quiz: "哪一个状态更像长期内部记忆？", answer: "细胞状态 C_t；h_t 是当前对外隐藏表示。"
  },
  {
    p: 46, slug: "bidirectional-lstm", title: "Bi-LSTM：从前后两个方向理解同一位置",
    problem: "只看左侧历史可能不够，怎样让每个位置同时利用右侧上下文？",
    chain: ["原序列正向读取", "原序列反向读取", "两套独立 LSTM", "同位置状态拼接", "下游预测"],
    extraVisuals: [{
      title: "双向 LSTM 的两条信息流",
      diagram: `
flowchart LR
    X1["x1"] --> X2["x2"] --> X3["x3"]
    X3 --> B2["反向状态 3→2"]
    B2 --> B1["反向状态 2→1"]
    X2 --> C["拼接 正向h2 + 反向h2"]
    B2 --> C
    C --> Y["位置 2 的表示"]
`,
    }],
    article: article([
      ["0:00–2:53", "双向的直觉", "老师用“我爱你 / 你爱我”的正反方向说明：正向网络读取过去，反向网络读取未来，两者参数彼此独立。"],
      ["2:53–5:48", "拼接而不是平均", "每个位置得到正向 hidden 和反向 hidden，常沿最后一维拼接，因此输出宽度由 hidden_size 变成 2×hidden_size。最终分类层的输入维度必须同步翻倍。"],
      ["5:48–8:11", "什么时候不能用", "双向模型更适合整句已经可见的分类、标注；流式语音、下一词生成等因果场景无法提前看到未来。代价还包括更多参数、计算和延迟。"],
    ]),
    points: ["两个方向有独立参数", "拼接后特征维通常翻倍", "因果生成不能偷看未来"],
    code: `import torch\nm = torch.nn.LSTM(5, 7, bidirectional=True, batch_first=True)\nout, (hn, cn) = m(torch.randn(2, 4, 5))\nprint(out.shape, hn.shape)`,
    output: "output=[2,4,14]，h_n=[2,2,7]；第一维的 2 是两个方向。",
    pit: "bidirectional=True 不等于把同一个结果简单倒序一次。",
    quiz: "hidden_size=7 时双向 output 最后一维为何是 14？", answer: "同位置的正向 7 维与反向 7 维拼接。"
  },
  {
    p: 47, slug: "lstm-code", title: "LSTM 代码：多一个细胞状态，接口如何变化",
    problem: "把 nn.RNN 换成 nn.LSTM 时，输入输出和初始化需要改哪些地方？",
    chain: ["创建 nn.LSTM", "准备 input", "准备 h0/c0", "解包 output/(hn,cn)", "核对形状"],
    extraVisuals: [recurrentShapeDiagram],
    article: article([
      ["0:00–4:36", "API 参数大体相同", "nn.LSTM 仍接收 input_size、hidden_size、num_layers 等，输入张量形状也与 RNN 相同。"],
      ["4:36–9:29", "状态从一个变成一对", "初始状态是 (h0,c0)，返回值是 output,(h_n,c_n)。h 与 c 形状相同，但语义不同。漏写括号或只接两个普通张量是常见解包错误。"],
      ["9:29–14:23", "运行与形状验证", "老师沿用随机输入，比较 RNN 和 LSTM 的 output 形状；只要配置相同，外部 output 形状一致，多出来的是 c_n。"],
      ["14:23–18:00", "代码迁移的真正原则", "不是复制后只改类名就算完成：任何 forward、初始化、返回值、训练调用都要同步处理 c。若封装统一接口，可以在模型内部隐藏差异。"],
    ]),
    points: ["LSTM 状态是 (h,c)", "外部 output 形状规则与 RNN 相同", "调用处也必须同步修改解包"],
    code: `import torch\nlstm = torch.nn.LSTM(5, 6)\nx = torch.randn(4, 2, 5)\nh0 = torch.zeros(1, 2, 6); c0 = torch.zeros(1, 2, 6)\nout, (hn, cn) = lstm(x, (h0, c0))\nprint(out.shape, hn.shape, cn.shape)`,
    output: "output=[4,2,6]，h_n=c_n=[1,2,6]。",
    pit: "不要写成 out, hn, cn = lstm(...): 第二个返回值本身是一个二元组。",
    quiz: "RNN 迁移成 LSTM 最明显多出的状态是什么？", answer: "细胞状态 c0/c_n。"
  },
  {
    p: 48, slug: "gru-diagram", title: "GRU 图解：两扇门合并记忆管理",
    problem: "GRU 如何用重置门和更新门，在较少结构下保留长依赖？",
    chain: ["重置门 r_t", "筛旧状态", "候选状态 h~_t", "更新门 z_t", "融合旧/新状态"],
    extraVisuals: [modelFamilyDiagram],
    article: article([
      ["0:00–2:56", "为什么已经有 LSTM，还要再学 GRU", `老师仍按“概念、内部结构、API、优缺点”的路线讲解。课程给出的出发点很直接：LSTM 能缓解普通 RNN 的长期依赖问题，但内部有遗忘门、输入门、输出门和独立细胞状态，结构与计算都更复杂。如果能合并一部分功能，同时保留选择性记忆能力，就能得到更轻量的循环单元，这就是本节介绍 GRU 的原因。

GRU 的全称是 **Gated Recurrent Unit（门控循环单元）**。它比普通 RNN 多了门控，比 LSTM 少了一条独立的细胞状态路径。老师特别提醒，“更简单”指状态和门的组织更精简，不等于它退化成普通 RNN，也不等于任何任务上都一定优于 LSTM。

从外部数据流看，每个时间步仍接收当前输入 \`x_t\` 和上一时刻隐藏状态 \`h_(t-1)\`，再产生新的隐藏状态。正因为接口相近，后面写 PyTorch 代码时，\`nn.RNN\` 换成 \`nn.GRU\` 后许多参数和张量形状都能沿用；但老师也强调，API 好换不代表内部原理可以跳过。`],
      ["2:56–5:51", "老师先用更新公式做了一次快速预览", `正式画大图前，老师先展示重置门 \`r_t\`、更新门 \`z_t\` 以及最终隐藏状态的组合公式，目的是让大家先看到 GRU 只有一份需要向后传递的状态 \`h_t\`，没有 LSTM 那样单独的 \`C_t\`。

课堂采用的最终融合写法是：

**融合公式：** \`h_t = (1-z_t) ⊙ h_(t-1) + z_t ⊙ h~_t\`

按这个约定，\`z_t=0\` 表示这一次几乎不更新，主要沿用旧状态；\`z_t=1\` 表示主要采用新的候选状态。老师让同学直接把 0 和 1 代入式子检查，而不是死背一句“越大越保留什么”。这一招很重要，因为有些教材会交换两项，把 \`z_t\` 定义成“保留旧信息”的比例，文字说法就会反过来。`],
      ["5:51–10:47", "从代码积累到读懂结构图", `这里老师插入了一段学习方法上的补充。他说，代码学习常常不是每一天都能立刻感到进步：前期写 \`Hello World\`、张量和简单网络会很慢，等练习量积累起来，某一天会突然发现不同代码之间大量结构都能对应上。GRU 的 API 看起来比前面的 LSTM 容易，正是因为 RNN 的输入、状态、输出和形状规则已经铺垫过了。

老师因此要求大家“多写、多练、一步一步来”，不要因为前期速度慢就判断自己不适合编程。这段不是 GRU 公式的一部分，却解释了课程为什么还要把结构图重新画一遍：复用旧知识不是省略理解，而是把已经掌握的 RNN/LSTM 框架迁移到新模型。

回到图中，老师先标出重置门、更新门、候选隐藏状态和最终隐藏状态，再按总—分顺序讲解。读图时应先认清符号：\`r_t\` 是 reset gate，\`z_t\` 是 update gate，\`h~_t\` 是候选隐藏状态，\`h_t\` 才是本时间步最终向后传递的状态。`],
      ["10:47–18:47", "GRU 怎样把 LSTM 的两份状态合成一份", `老师把 LSTM 和 GRU 放在一起比较：LSTM 用 \`C_t\` 保存较长期的细胞记忆，同时用 \`h_t\` 作为当前对外状态；两者形状通常相同，但职责不同。GRU 取消独立的 \`C_t\`，把“保存、筛选和更新记忆”的职责都放进隐藏状态 \`h_t\` 及其门控计算中。因此，GRU 的核心可以概括成“两扇门加一份隐藏状态”。

为了说明“合并”不是“把能力随便删掉”，老师用了智能门锁的类比：传统方案可能是普通门锁再外接监控设备，新的方案把上锁、监控和报警整合到一个系统里。部件看起来少了，但关键功能仍被重新组织并保留下来。对应到 GRU，就是把 LSTM 部分门和状态路径合并成更轻量的记忆管理方式。

老师又用手写账本与记账 App 作比较。手写账本需要自己记录收入支出，再手工分类、汇总；记账 App 可以在同一套系统中自动识别消费类别并统计交通、餐饮、衣物等支出。这个类比想表达的是：GRU 不再把“长期记忆本”和“当前隐藏状态”分成两本账，而是在一份状态中完成选择性保留与更新。类比只帮助理解职责合并，并不表示门控值真由人工规则决定；它们仍是模型从数据中训练得到的连续向量。`],
      ["18:47–21:45", "第一扇门：重置门决定旧信息怎样参与候选状态", `重置门的典型写法是：

**重置门：** \`r_t = sigmoid(W_r[x_t, h_(t-1)] + b_r)\`

\`sigmoid\` 把每个维度压到 0 到 1。随后 \`r_t ⊙ h_(t-1)\` 对旧隐藏状态逐元素筛选：某一维接近 1，该维旧信息较完整地参与候选状态计算；某一维接近 0，该维历史信息被大幅削弱，相当于在生成新候选时“重置”这部分过去。

老师在图上把它概括为“决定要不要擦除旧记忆”，并让大家比较两个端点。这里要补充一个容易误解的地方：重置门通常不是整句话共用一个 0/1 开关，而是与隐藏状态同宽的一组连续比例，所以模型可以忘掉某些特征，同时保留另一些特征。`],
      ["21:45–24:45", "第二扇门：更新门规划新旧记忆的比例", `更新门同样由当前输入和旧状态共同计算：

**更新门：** \`z_t = sigmoid(W_z[x_t, h_(t-1)] + b_z)\`

老师用家庭资金规划帮助理解：一部分钱作为稳定储备，另一部分拿去投资；比例不能凭感觉混在一起，而要明确分配。对应到 GRU，旧隐藏状态像已经积累的储备，新候选状态像根据当前输入形成的新方案，更新门负责决定最终状态中两者各占多少。

按本节采用的 \`h_t = (1-z_t) ⊙ h_(t-1) + z_t ⊙ h~_t\`，\`z_t\` 越接近 1，写入的新候选越多；越接近 0，保留的旧状态越多。再次强调：看到其他资料时先看最终公式中 \`z_t\` 乘在哪一项，再判断“1”与“0”的含义。`],
      ["24:45–29:45", "候选隐藏状态：先筛过去，再和当前输入合成新方案", `候选隐藏状态可写成：

**候选状态：** \`h~_t = tanh(W_h[x_t, r_t ⊙ h_(t-1)] + b_h)\`

老师把这个式子拆成三步：先用重置门筛一遍旧状态；再把筛过的历史与当前输入 \`x_t\` 合在一起；最后经过线性变换与 \`tanh\`，生成一份“准备写入但尚未定稿”的新记忆。\`tanh\` 把候选值压到 -1 到 1，也提供非线性表达能力。

“候选”两个字不能漏掉。\`h~_t\` 只是根据当前信息提出的新方案，并不会原封不动地成为最终状态；它还要接受更新门的比例控制。重置门管的是“做新方案时参考多少过去”，更新门管的是“最终采用多少新方案”，两扇门的作用因此不能互换。`],
      ["29:47–31:47", "最终隐藏状态：把旧状态与候选状态定稿融合", `最后一步重新回到融合式：

**最终状态：** \`h_t = (1-z_t) ⊙ h_(t-1) + z_t ⊙ h~_t\`

老师逐项解释：第一项是按比例留下的旧隐藏状态，第二项是按比例写入的新候选，二者相加得到本时间步最终状态 \`h_t\`。这个 \`h_t\` 既交给下一个时间步，也可以作为当前时间步的输出参与后续任务。

把 \`z_t\` 看成向量还能得到更准确的直觉：同一时间步里，有些维度可以接近 0，长期保留；另一些维度可以接近 1，快速更新。GRU 能让信息跨很多步传递，靠的不是“永远不忘”，而是模型可以学出较平滑的保留路径。`],
      ["31:47–33:37", "老师最后要求：会画最小图，更要能用自己的话讲", `课程最后把 GRU 放回技术演进中看：在 Transformer 和大模型成为主流之前，RNN、LSTM、GRU 曾是很多序列任务与面试中的重点；后来关注中心发生变化，但门控、状态传递和长期依赖仍是理解序列模型的重要基础。

老师不要求逐字符背公式，而是要求能画出最小结构并顺畅说明四句话：重置门怎样筛旧信息，候选状态怎样融合当前输入，更新门怎样分配新旧比例，最终 \`h_t\` 怎样向后传递。公式忘了某个下标可以查，若连数据流都要临场停很久才能想起来，说明还没有真正掌握。

因此本节最合适的复习方法不是重复看答案，而是合上文章，自己画出 \`h_(t-1)\`、\`x_t\`、\`r_t\`、\`z_t\`、\`h~_t\`、\`h_t\`，再像老师课堂要求的那样，用大白话从头讲一遍。`],
    ]),
    points: ["GRU 两门：reset 与 update", "没有独立细胞状态 C", "公式约定可能互换 z 与 1-z"],
    code: `import torch\ngru = torch.nn.GRU(5, 7, batch_first=True)\nout, hn = gru(torch.randn(2, 4, 5))\nprint(out.shape, hn.shape)`,
    output: "output=[2,4,7]，h_n=[1,2,7]；接口更接近普通 RNN。",
    pit: "不要凭门数推断准确率；数据规模和超参数会改变结果。",
    quiz: "GRU 与 LSTM 在状态接口上的关键差别？", answer: "GRU 只有隐藏状态 h；LSTM 还有独立细胞状态 c。"
  },
  {
    p: 49, slug: "gru-code", title: "GRU 代码：替换循环层并验证接口",
    problem: "为何 GRU 代码几乎能沿用 RNN，而不能完全照搬 LSTM？",
    chain: ["nn.GRU 配置", "input/h0", "运行前向", "得到 output/hn", "比较三模型"],
    extraVisuals: [recurrentShapeDiagram],
    article: article([
      ["0:00–5:44", "概念与优缺点复盘", "老师先写 GRU 全称、两扇门和长序列优势，再指出它仍无法跨时间步并行。课堂对“GRU 一定弱于 LSTM”的绝对说法过强：不同任务可能各有胜负，应实测。"],
      ["5:44–10:38", "五步 API 演示", "创建 nn.GRU、创建输入、创建 h0、运行、打印。参数 input_size/hidden_size/num_layers 与 RNN 一致。"],
      ["10:38–13:37", "输出形状", "output 含所有时间步，h_n 含每层最终状态；因为没有 c_n，解包与普通 RNN 相同。老师再次用形状表提醒不要混淆时间维、批量维和隐藏维。"],
      ["13:37–16:49", "选择题与纠错", "LSTM 是三门一状态，不存在“记忆门”；GRU 和 LSTM 都有门控。真正写项目时，可把公共分类头封装，让三种循环主干可切换。"],
    ]),
    points: ["GRU 调用接口与 RNN 接近", "LSTM 才返回 c_n", "同形状不代表内部计算相同"],
    code: `import torch\nfor cls in (torch.nn.RNN, torch.nn.GRU):\n    m = cls(5, 6, batch_first=True)\n    out, hn = m(torch.randn(2, 3, 5))\n    print(cls.__name__, out.shape, hn.shape)`,
    output: "两者外部形状相同，但内部更新方程不同。",
    pit: "只改 nn.RNN 为 nn.GRU 后，变量名、注释和保存路径也要同步，避免训练/加载错模型。",
    quiz: "GRU forward 默认返回几个顶层结果？", answer: "两个：output 和 h_n。"
  },
  {
    p: 50, slug: "name-classification-requirement", title: "姓名分类需求：从名字预测 18 个国家类别",
    problem: "怎样把一个现实需求拆成数据、编码、模型、训练、评估和预测六个部分？",
    chain: ["姓名+国家数据", "字符 One-Hot", "三类循环模型", "18 类训练", "曲线比较与预测"],
    extraVisuals: [projectPipelineDiagram],
    article: article([
      ["0:00–4:54", "需求与数据", "老师提出注册场景：用户输入姓名，系统估计国家/地区，可作为后续流程的弱提示。数据约 20,074 条，两列分别是姓名和国家，共 18 类。预测国籍具有偏见和误判风险，不能作为身份事实或限制用户权益的依据。"],
      ["4:54–7:52", "为什么用字符 One-Hot", "项目刻意以字符为单位：52 个大小写英文字母加 5 个常用符号，共 57 维。一个长度 L 的姓名表示为 [L,57]。这是理解型案例，不是现代生产方案的唯一选择。"],
      ["7:52–10:49", "三模型对比", "同一数据分别训练 RNN、LSTM、GRU，对比损失、准确率和耗时。老师展示的 GRU 结果仅属于本次配置，不能推广为所有任务的固定排名。"],
      ["10:49–12:36", "形状自测", "每个字符是 57 维 One-Hot；5 字符姓名是 [5,57]，批量为 1 时模型内部是 [1,5,57]（若 batch_first=True）。"],
    ]),
    points: ["18 类多分类", "每字符 57 维 One-Hot", "模型选择必须在同一数据和评估协议下比较"],
    code: `name_length = 5\nalphabet_size = 57\nprint((name_length, alphabet_size))`,
    output: "单个 5 字符姓名的字符矩阵形状是 [5,57]。",
    pit: "姓名只能提供弱统计线索，不能可靠决定真实国籍、民族或身份。",
    quiz: "为什么 5 个字符不是一个 5×1 的整数就直接送入线性层？", answer: "整数只是类别 ID；本案例用每字符 57 维 One-Hot 防止把 ID 大小误当连续数值。"
  },
  {
    p: 51, slug: "alphabet-and-countries", title: "全局字母表与国家名：固定输入列和输出列",
    problem: "如何让训练、验证和预测阶段始终使用同一字符编号和类别编号？",
    chain: ["定义 57 字符表", "建立 char→index", "定义 18 国家表", "建立 country→index", "全阶段冻结映射"],
    extraVisuals: [projectPipelineDiagram],
    article: article([
      ["0:00–5:54", "先搭项目骨架", "老师回顾导包、数据处理、模型、训练、可视化和预测，并强调本案例同时比较三种循环模型。"],
      ["6:00–10:57", "字符表", "使用 ascii_letters 得 52 个大小写字母，再追加空格、句点、逗号、分号、单引号等 5 个符号，长度为 57。课堂逐字符打印核对。"],
      ["11:59–16:19", "国家类别表", "18 个国家名按固定顺序保存，列表位置就是标签 ID。老师也提到可从数据中去重生成；工程上应排序或保存映射，否则不同运行顺序可能改变标签含义。"],
    ]),
    points: ["字符表决定 One-Hot 列", "国家表决定分类输出列", "映射必须随模型一起保存"],
    code: `import string\nletters = string.ascii_letters + " .,;'"\nchar_to_id = {c:i for i,c in enumerate(letters)}\nprint(len(letters), char_to_id["A"])`,
    output: "字符表长度为 57；每个字符获得稳定索引。",
    pit: "从 set 直接生成类别列表会造成顺序不稳定。",
    quiz: "模型权重相同但国家列表顺序变了，会怎样？", answer: "同一个输出下标会被解释成另一个国家，预测语义完全错误。"
  },
  {
    p: 52, slug: "read-name-data", title: "读取数据：把姓名与国家分别保存并处理异常行",
    problem: "怎样逐行读取 CSV，同时保证特征和标签永远一一对应？",
    chain: ["打开 UTF-8 文件", "逐行切分", "校验字段", "过滤/规范姓名", "追加 X 与 y"],
    extraVisuals: [projectPipelineDiagram],
    article: article([
      ["0:00–2:58", "函数输入输出", "老师定义 read_data(path)，返回姓名列表 X 与国家列表 y。左列是特征，右列是标签。还提出过滤过短姓名的想法，因为单个字母几乎无从判断。"],
      ["2:58–6:21", "逐行读取", "使用 with open 自动关闭文件，逐行 strip，再按分隔符切开。真实 CSV 若字段可能含逗号，应使用 csv 模块而不是简单 split。"],
      ["6:22–11:20", "清洗与核对", "合法行才同时 append X 和 y；打印数量和样例，确认两列表长度一致。过滤规则必须只依赖输入本身，且训练与预测一致。"],
    ]),
    points: ["X/y 必须同步追加", "异常行应记录而非悄悄吞掉", "生产 CSV 优先用 csv.reader"],
    code: `import csv\nrows = ["Smith,English", "Zhang,Chinese"]\nparsed = list(csv.reader(rows))\nX = [r[0] for r in parsed]; y = [r[1] for r in parsed]\nprint(X, y)`,
    output: "得到两个等长列表，索引 i 的姓名与国家仍对应。",
    pit: "单纯 line.split(',') 遇到带逗号字段会错列。",
    quiz: "过滤一条姓名时为什么标签也必须一起过滤？", answer: "否则 X 与 y 从该位置起错位，模型学到错误监督。"
  },
  {
    p: 53, slug: "dataset", title: "Dataset：把变长姓名转成字符 One-Hot 张量",
    problem: "Dataset 的 __getitem__ 怎样把一条字符串即时转换成模型可用的张量和标签？",
    chain: ["索引一条姓名", "逐字符查表", "构造 [L,57]", "国家查 ID", "返回 x/y 张量"],
    extraVisuals: [{
      title: "Dataset 类职责 UML",
      diagram: `
classDiagram
    class NameDataset {
      names
      countries
      char_to_id
      country_to_id
      __len__()
      __getitem__(index)
      name_to_tensor(name)
    }
    class DataLoader {
      batch_size
      shuffle
      collate_fn
    }
    NameDataset --> DataLoader : 提供单样本
`,
    }],
    article: article([
      ["0:00–4:59", "继承 Dataset 的三个核心部分", "老师定义构造函数保存 X/y，__len__ 返回样本数，__getitem__ 按索引返回一条特征和标签。"],
      ["4:59–10:54", "字符 One-Hot", "创建 [姓名长度,57] 全零张量，遍历字符，查字符索引并把对应位置置 1。未知字符需要明确策略：规范化、UNK 列或丢弃；直接 find 返回 -1 会误把最后一列置 1。"],
      ["10:56–16:56", "标签转 ID", "国家名在固定类别列表中的位置作为整数标签。NLLLoss/CrossEntropyLoss 要求标签是 long 类型类别索引，不是 One-Hot。"],
      ["16:56–21:02", "变长样本的批处理难题", "不同姓名长度不同。课堂选择 batch_size=1，从而避免默认 collate 无法堆叠。更一般方案是 padding+mask、pack_padded_sequence 或自定义 collate_fn。"],
    ]),
    points: ["特征是浮点 One-Hot，标签是 long 类别 ID", "未知字符不能误映射到 -1", "变长序列需要专门的 batch 策略"],
    code: `import torch\nalphabet = "abc"\ndef encode(name):\n    x = torch.zeros(len(name), len(alphabet))\n    for i, ch in enumerate(name): x[i, alphabet.index(ch)] = 1\n    return x\nprint(encode("cab"))`,
    output: "输出 [3,3] 矩阵，每行恰有一个 1。",
    pit: "Python 的 list[-1] 合法；未找到字符若返回 -1，会静默污染最后一列。",
    quiz: "为什么标签不需要 18 维 One-Hot？", answer: "交叉熵类损失通常直接接收类别索引。"
  },
  {
    p: 54, slug: "dataloader", title: "DataLoader：为变长姓名组织训练迭代",
    problem: "为什么本案例先用 batch_size=1，shuffle 又在什么时候生效？",
    chain: ["Dataset 单样本", "DataLoader 配置", "shuffle 索引", "产生 x/y", "训练循环消费"],
    extraVisuals: [projectPipelineDiagram],
    article: article([
      ["0:00–3:51", "从 Dataset 到 DataLoader", "老师用 DataLoader(dataset,batch_size=1,shuffle=True) 包装。每轮遍历由加载器产生姓名张量和标签。"],
      ["3:51–7:48", "为何批量只能先写 1", "默认 collate 用 stack 合并样本，姓名长度不一会报错；batch_size=1 时每批只有一种长度。课堂这是降低复杂度，不代表 RNN 只能单样本训练。"],
      ["7:48–11:37", "打印形状做抽查", "遍历几条数据，确认 x 形如 [1,L,57]、y 形如 [1]。shuffle=True 每轮顺序可能不同，因此不能把打印的第一条当固定样例。"],
    ]),
    points: ["DataLoader 不负责文本编码本身", "batch_size=1 是变长序列的简化方案", "shuffle 只应作用训练集"],
    code: `from torch.utils.data import DataLoader\n# loader = DataLoader(dataset, batch_size=1, shuffle=True)\nprint("变长序列若批量>1，需要 padding 或自定义 collate_fn")`,
    output: "提示本节的设计约束。",
    pit: "验证/测试集通常不要 shuffle，以便复现实例顺序。",
    quiz: "batch_size=8 为什么可能报 stack 大小不一致？", answer: "8 个姓名的 L 不同，原始 [L,57] 张量无法直接堆叠。"
  },
  {
    p: 55, slug: "log-softmax", title: "LogSoftmax：旧式 NLLLoss 与现代 CrossEntropyLoss 的关系",
    problem: "模型最后到底输出 logits、概率还是 log-probabilities，损失函数必须怎样配套？",
    chain: ["分类 logits", "LogSoftmax", "log-probabilities", "NLLLoss", "标量损失"],
    extraVisuals: [{
      title: "输出层与损失的两种合法组合",
      diagram: `
flowchart TB
    A["Linear 输出 logits"] --> B["CrossEntropyLoss"]
    A --> C["LogSoftmax(dim=-1)"]
    C --> D["NLLLoss"]
    B --> E["标量损失"]
    D --> E
`,
    }],
    article: article([
      ["0:00–4:44", "Softmax 与 LogSoftmax", "老师先回顾 Softmax 把 18 个分数归一化成概率；LogSoftmax 返回概率的对数，数值通常非正，但 exp 后和为 1。"],
      ["4:44–9:42", "旧式组合", "课程模型末尾使用 LogSoftmax(dim=-1)，训练使用 NLLLoss。dim=-1 表示沿最后的类别维处理。"],
      ["9:42–13:24", "现代组合与重复计算坑", "更常见写法是模型直接返回 logits，损失用 CrossEntropyLoss，它内部已经包含 LogSoftmax+NLLLoss。两种组合都对，但不能模型先 Softmax 再 CrossEntropyLoss，也不能重复 LogSoftmax。"],
    ]),
    points: ["NLLLoss 要 log-probabilities", "CrossEntropyLoss 要原始 logits", "预测概率可对 logits 做 softmax"],
    code: `import torch\nlogits = torch.tensor([[1.0, 2.0, 0.0]])\nlogp = torch.log_softmax(logits, dim=-1)\nprint(logp, logp.exp().sum(-1))`,
    output: "logp 是对数概率；取 exp 后每行和为 1。",
    pit: "负的 log-probability 不是负概率。",
    quiz: "使用 CrossEntropyLoss 时模型 forward 还要 LogSoftmax 吗？", answer: "不要；直接返回 logits。"
  },
  {
    p: 56, slug: "rnn-model", title: "RNN 分类模型：取最后时间步映射到 18 类",
    problem: "怎样把变长字符序列的循环输出变成一个国家分类结果？",
    chain: ["输入 [L,57]", "补 batch 维", "RNN 得所有时间步", "取最后隐藏表示", "Linear→18→LogSoftmax"],
    extraVisuals: [recurrentShapeDiagram, projectPipelineDiagram],
    article: article([
      ["0:00–5:33", "模型类的组成", "老师定义 RNN 层、Linear 分类层和 LogSoftmax。input_size=57，hidden_size 是可调容量，output_size=18。"],
      ["5:34–10:48", "旧式损失配套", "模型保留 LogSoftmax，因此训练要配 NLLLoss；若换 CrossEntropyLoss，应删除 LogSoftmax。"],
      ["11:42–16:11", "输入加 batch 维", "Dataset 单条姓名原本 [L,57]，进入模型前用 unsqueeze 增加 batch 维。要先确定 batch_first；课程代码把单条序列放到循环层要求的顺序。"],
      ["16:14–19:54", "最后一步做整句分类", "RNN 返回所有时间步 output 和 h_n。老师取最后时间步表示，送入 Linear 从 hidden_size 映射到 18 类，再 LogSoftmax。对有 padding 的批量，不能盲取 output[:,-1]，应按真实长度取最终有效状态或用 h_n/packing。"],
      ["20:23–21:37", "初始化隐藏状态", "用全零张量初始化 h0，形状由层数、batch 和 hidden_size 决定。更稳妥是在输入同一 device/dtype 上创建。"],
    ]),
    points: ["Linear 把隐藏维映射到类别维", "取最后有效时间步，不是 padding 最后一步", "初始化状态应匹配 device"],
    code: `import torch\nfrom rnn_from_scratch.model import NameClassifier\nmodel = NameClassifier(57, 128, 18, kind="rnn")\nprint(model(torch.randn(1, 6, 57)).shape)`,
    output: "1 个 6 字符姓名，输出 [1,18] 对数概率。",
    pit: "padding 后直接取序列最后位置可能取到 PAD 状态。",
    quiz: "为什么 Linear 的 out_features 是 18？", answer: "每一列对应一个国家类别。"
  },
  {
    p: 57, slug: "test-rnn", title: "测试 RNN：用随机输入把形状链走通",
    problem: "训练前怎样证明模型前向传播、隐藏状态和 18 类输出没有维度错误？",
    chain: ["实例化模型", "造 [L,57] 输入", "初始化 h0", "前向运行", "检查 [1,18]"],
    extraVisuals: [recurrentShapeDiagram],
    article: article([
      ["0:00–5:58", "测试配置", "老师用 57 输入维、128 隐藏维、18 输出维实例化模型。随机 [L,57] 只用于形状冒烟测试，L 可用某个姓名长度举例。"],
      ["5:58–10:56", "模型结构与测试张量", "打印模型确认 RNN、Linear、LogSoftmax 层；创建随机输入并核对每个字符 57 维。"],
      ["10:57–16:52", "为什么最终是 18 维", "所有时间步原本产生 hidden_size=128 的表示，取最后一步后通过 Linear(128,18)，所以分类输出是 18，不再是 128。"],
      ["17:05–20:07", "从分数到类别", "模型给出 18 个 log-probabilities；可用 argmax 得最高类别。若需要普通概率，用 exp（对于 log-softmax 输出）而不是再错误套一次任意归一化。"],
    ]),
    points: ["先跑形状测试再训练", "Linear 负责 128→18", "argmax 返回类别索引"],
    code: `import torch\nfrom rnn_from_scratch.model import NameClassifier\nm = NameClassifier(57,128,18,kind="rnn")\nlogp = m(torch.randn(1,6,57))\nprint(logp.shape, int(logp.argmax(-1)))`,
    output: "形状 [1,18]，并打印当前随机模型猜出的类别索引。",
    pit: "随机未训练模型的类别没有业务意义，只能检查接口。",
    quiz: "隐藏维 128 为什么不会让最终输出也是 128？", answer: "分类 Linear 把 128 维映射为 18 类。"
  },
  {
    p: 58, slug: "lstm-gru-models", title: "搭建 LSTM 与 GRU 模型：复用分类头，隔离状态差异",
    problem: "三模型哪些代码可以共用，哪些状态差异必须单独处理？",
    chain: ["复制公共分类骨架", "替换循环主干", "LSTM 解包 h/c", "GRU 只解包 h", "统一 18 类输出"],
    extraVisuals: [modelFamilyDiagram, projectPipelineDiagram],
    article: article([
      ["0:00–3:55", "从 RNN 模型迁移到 LSTM", "构造参数、Linear 和输出层可复用；循环层换成 nn.LSTM，forward 返回 state=(h,c)，初始化也要一对状态。"],
      ["3:55–6:54", "LSTM 返回值", "最终分类仍可使用 h_n 的最后一层，c_n 不直接送分类头，但必须正确接收。"],
      ["6:54–9:52", "GRU 迁移更接近 RNN", "循环层换为 nn.GRU，其余状态接口仍是单个 h；课堂指出主要改动很少。"],
      ["9:52–10:23", "复用与维护", "复制能快速教学，但生产代码更适合用一个 kind 参数选择循环类，公共前向和分类头只写一次，降低三份代码漂移风险。"],
    ]),
    points: ["LSTM 独有 c 状态", "三模型可共享分类 Linear", "统一封装便于公平比较"],
    code: `import torch\nfrom rnn_from_scratch.model import NameClassifier\nx=torch.randn(2,5,57)\nfor kind in ("rnn","lstm","gru"):\n    print(kind, NameClassifier(57,32,18,kind=kind)(x).shape)`,
    output: "三种模型都输出 [2,18]，内部状态实现不同。",
    pit: "复制三份模型后很容易只修一份 bug；统一参数化更安全。",
    quiz: "哪种模型的 state 是二元组？", answer: "LSTM，包含 h_n 和 c_n。"
  },
  {
    p: 59, slug: "test-three-models", title: "测试三种模型：用同一数据管道公平验形状",
    problem: "怎样确保 RNN/LSTM/GRU 的差异来自模型，而不是数据和参数接错？",
    chain: ["读取同一数据", "构建同一 Dataset", "同一 DataLoader", "实例化三模型", "逐一前向对比"],
    extraVisuals: [projectPipelineDiagram],
    article: article([
      ["0:00–4:03", "实例化与真实数据", "老师不再只用随机张量，而是读取姓名数据、构建 Dataset 和 DataLoader。三模型共享 57/128/18 配置。"],
      ["4:03–8:06", "加载器与变长限制", "batch_size=1 继续避免姓名长度不同的堆叠问题；shuffle 可打乱训练顺序。"],
      ["8:06–13:08", "三模型对象", "分别创建 RNN、LSTM、GRU，打印结构核对输入、隐藏和输出维度完全一致。"],
      ["13:08–20:51", "抽一条样本看形状", "x 是 [1,L,57]，y 是 [1]。老师逐维解释 1 是批量、L 是姓名字符数、57 是字符特征。"],
      ["20:52–34:27", "逐模型前向与状态处理", "送入模型时应避免重复增加/删除 batch 维。RNN/GRU 接 h，LSTM 接 (h,c)。最终都得到 18 类输出。公平比较还应固定随机种子、数据划分、训练轮数和优化器配置。"],
    ]),
    points: ["三模型必须用同一数据划分", "先验证每个 x/y 形状", "控制随机性才能比较"],
    code: `import torch\nfrom rnn_from_scratch.model import NameClassifier\ntorch.manual_seed(1)\nx=torch.randn(1,8,57)\nfor kind in ("rnn","lstm","gru"):\n    print(kind, NameClassifier(57,128,18,kind=kind)(x).shape)`,
    output: "每个模型都产生 [1,18]。",
    pit: "只比较一次随机初始化的输出没有意义；正式比较要训练多次并报告波动。",
    quiz: "模型对比至少应固定哪些条件？", answer: "数据划分、编码、隐藏维、训练轮数、优化器等，并控制/记录随机种子。"
  },
  {
    p: 60, slug: "train-rnn", title: "训练 RNN：外层 epoch、内层 batch 与五步反向传播",
    problem: "一轮训练中数据、损失、梯度和参数到底按什么顺序流动？",
    chain: ["取一批 x/y", "forward", "计算 NLLLoss", "zero_grad/backward/step", "累计指标并保存"],
    extraVisuals: [trainingSequenceDiagram],
    article: article([
      ["0:00–7:19", "训练函数输入与准备", "老师定义训练轮数、模型、损失函数、优化器，并准备记录损失、准确率和耗时。模型末尾为 LogSoftmax，因此使用 NLLLoss。"],
      ["7:51–15:59", "双重循环", "外层控制 epoch，内层遍历 DataLoader。每批取 x/y，初始化状态或由模型内部处理，然后前向得到 18 类 log-probabilities。"],
      ["16:11–27:14", "反向传播固定顺序", "先 optimizer.zero_grad() 清掉上批梯度；loss.backward() 计算梯度；optimizer.step() 更新参数。若漏清零，PyTorch 默认会累积梯度。"],
      ["27:14–37:14", "损失与准确率统计", "每批损失累加；argmax 预测与标签比较得到正确数。epoch 平均损失应按样本或批次口径明确，不能混算。进度条只负责显示。"],
      ["37:14–44:38", "保存模型与返回曲线数据", "训练结束保存 state_dict，并返回损失、准确率、耗时等用于对比。保存目录要存在，模型结构配置也应一并记录。"],
    ]),
    points: ["zero_grad→backward→step 顺序固定", "训练指标口径要明确", "保存 state_dict 还需保存模型配置"],
    code: `# 训练核心顺序\noptimizer.zero_grad()\nlogp = model(x)\nloss = criterion(logp, y)\nloss.backward()\noptimizer.step()`,
    output: "每批先算预测和损失，再反传并更新一次参数。",
    pit: "不要在 forward 前忘记 model.train()，评估时要 model.eval()。",
    quiz: "为什么每批都要 zero_grad？", answer: "PyTorch 梯度默认累积，不清零会把多批梯度叠在一起。"
  },
  {
    p: 61, slug: "train-lstm", title: "训练 LSTM：复用训练循环，正确管理 h 与 c",
    problem: "LSTM 训练与 RNN 训练相比，真正需要改变的是什么？",
    chain: ["同一 DataLoader", "LSTM forward", "h/c 状态", "同一损失与优化", "保存 LSTM 权重"],
    extraVisuals: [trainingSequenceDiagram],
    article: article([
      ["0:00–2:49", "训练骨架完全相同", "读取批次、前向、算损失、清梯度、反传、更新、统计指标的顺序不变。"],
      ["2:58–5:57", "状态差异藏进模型", "若模型 forward 已在内部正确处理 (h,c)，外部训练函数几乎只需换模型对象。状态跨样本复用时必须 detach 并处理边界；本案例每个姓名独立，通常每条重新初始化。"],
      ["5:57–8:03", "保存与命名", "LSTM 权重应使用独立路径，不能覆盖 RNN；加载时也必须先创建完全相同配置的 LSTM 结构。"],
    ]),
    points: ["优化步骤不因循环单元改变", "独立姓名不应串联隐藏状态", "保存路径和模型类型要匹配"],
    code: `from rnn_from_scratch.model import NameClassifier\nmodel = NameClassifier(57,128,18,kind="lstm")\nprint(model.kind)`,
    output: "模型主干明确为 LSTM，训练循环可复用。",
    pit: "把前一个姓名的状态传给下一个姓名会制造不存在的序列关系。",
    quiz: "本项目为什么每个姓名重新初始化状态？", answer: "不同姓名是独立样本，不是同一长序列的连续片段。"
  },
  {
    p: 62, slug: "train-gru", title: "训练 GRU：同一协议下比较速度与效果",
    problem: "怎样训练 GRU 并避免把一次实验结果误当成模型定律？",
    chain: ["实例化 GRU", "复用训练循环", "记录每轮指标", "保存权重", "与另两模型同协议比较"],
    extraVisuals: [trainingSequenceDiagram],
    article: article([
      ["0:00–2:52", "把模型对象换成 GRU", "训练函数的输入输出和优化步骤保持不变，GRU 内部只有 h 状态。"],
      ["2:53–5:44", "记录耗时、损失和准确率", "老师为后续三张对比图保存数据。耗时会受硬件、首次初始化、后台负载影响，至少应多次运行取统计量。"],
      ["5:44–7:27", "结果解释", "课程这次实验中 GRU 收敛较快、准确率较高，但这只是一个数据集和超参数组合。公平比较最好为每种模型单独调参，并报告均值与方差。"],
    ]),
    points: ["GRU 可复用统一训练函数", "训练时间要多次测量", "单次曲线不能推出普遍优劣"],
    code: `from rnn_from_scratch.model import NameClassifier\nmodel = NameClassifier(57,128,18,kind="gru")\nprint(model.kind)`,
    output: "输出 gru，表明公共分类器已切换循环主干。",
    pit: "不同模型若学习率等超参数不合适，固定配置比较只能说明“该配置下”的结果。",
    quiz: "课程中 GRU 最好，能否断言所有任务都选 GRU？", answer: "不能；需按数据、预算和验证结果重新比较。"
  },
  {
    p: 63, slug: "visualize-comparison", title: "可视化三模型：损失、时间和准确率要一起看",
    problem: "一条曲线不能决定模型，怎样从三类指标做有边界的结论？",
    chain: ["收集同口径记录", "绘损失曲线", "绘训练耗时", "绘准确率曲线", "综合选择"],
    extraVisuals: [{
      title: "模型选择决策图",
      diagram: `
flowchart TB
    A["验证集指标"] --> D{"任务优先级"}
    B["训练/推理耗时"] --> D
    C["参数量与稳定性"] --> D
    D --> E["选择候选模型"]
    E --> F["在独立测试集只评一次"]
`,
    }],
    article: article([
      ["0:00–5:40", "准备绘图数据", "老师将三模型的 epoch 损失、准确率和总耗时组织为列表，配置中文字体并分别绘图。曲线的横轴、平滑方式和数据划分必须一致。"],
      ["5:40–11:48", "损失曲线", "损失下降代表优化目标改善；收敛快不等于最终泛化最好。若只画训练损失，无法判断过拟合，应同时记录验证损失。"],
      ["11:49–15:17", "训练时间", "课程运行中 RNN、LSTM、GRU 耗时不同。老师指出不同电脑结果会变，因此时间图属于特定硬件环境。"],
      ["15:18–18:17", "准确率与综合结论", "本次 GRU 准确率更高，但多分类还应看混淆矩阵、宏平均 F1 和各国家样本数。作业要求保存三张图；学习重点是能解释图，而非只交截图。"],
    ]),
    points: ["训练与验证指标要分开", "时间结论绑定硬件和实现", "类别不平衡时准确率不够"],
    code: `records = {"rnn":{"acc":.70,"sec":20}, "lstm":{"acc":.73,"sec":48}, "gru":{"acc":.75,"sec":17}}\nprint(max(records, key=lambda k: records[k]["acc"]))`,
    output: "示例按准确率选 gru；真实选择还要加入速度、稳定性等约束。",
    pit: "不能在测试集上反复挑模型，否则测试集也被用来调参。",
    quiz: "为什么还要看验证损失？", answer: "训练损失下降时验证损失可能上升，提示过拟合。"
  },
  {
    p: 64, slug: "rnn-prediction", title: "RNN 预测：姓名转张量、加载权重并取 Top-k",
    problem: "上线预测怎样复用训练时字符表、模型结构和类别表，输出可解释的前三名？",
    chain: ["输入姓名", "同字符表编码", "创建同结构模型", "加载 state_dict", "no_grad 前向与 Top-k"],
    extraVisuals: [{
      title: "推理调用序列",
      diagram: `
sequenceDiagram
    participant U as 输入姓名
    participant E as 字符编码器
    participant M as 已加载模型
    participant K as Top-k解析器
    U->>E: 原始字符串
    E->>M: [1,L,57]
    M->>K: 18 类 log-probabilities
    K-->>U: 类别名与分数
`,
    }],
    article: article([
      ["0:00–2:58", "先总结训练再进入预测", "老师回顾数据读取、Dataset、DataLoader、模型、双层循环、反向传播、保存和三模型曲线，随后根据本次实验偏向 GRU。"],
      ["2:58–5:56", "预测必须走同一编码", "直接把字符串扔给模型不行。定义 line_to_tensor，把每个字符在同一 57 字符表中的列置 1。大小写、未知字符和规范化规则必须与训练一致。"],
      ["5:56–10:56", "实现字符矩阵", "创建 [L,57] 零张量，enumerate 遍历字符，查索引并置 1，返回结果。课堂写法依赖全局字符表；更易测试的设计是把编码器对象显式传入。"],
      ["10:56–14:06", "重建模型与加载权重", "先用 57/128/18 创建同结构模型，再 load_state_dict(torch.load(path,...))，切到 eval，并放在 torch.no_grad() 中推理。"],
      ["14:06–20:05", "Top-k 与类别解析", "topk(3,dim=-1) 返回前三个值和索引；逐项用 country 列表把索引翻译成类别名。课程模型输出 LogSoftmax，所以值是对数概率；要显示普通概率需 exp。"],
      ["20:05–23:11", "结果受训练程度限制", "老师尝试多个姓名并指出只训练一轮时预测不可靠。训练轮数增加也不保证公平可靠，仍需独立测试集和伦理边界。LSTM/GRU 预测代码可复用同一流程。"],
    ]),
    points: ["编码映射必须与训练一致", "推理要 eval + no_grad", "LogSoftmax 输出需 exp 才是普通概率"],
    code: `import torch\nfrom rnn_from_scratch.model import NameClassifier\nm=NameClassifier(57,128,18,kind="rnn").eval()\nwith torch.no_grad():\n    values, ids = m(torch.randn(1,5,57)).topk(3, dim=-1)\nprint(ids.shape, values.exp())`,
    output: "得到 3 个类别索引与对应概率形式的分数。",
    pit: "只保存权重、不保存字符表和国家表，模型无法被正确解释。",
    quiz: "为什么 topk 的 log 值不能直接写成 80%？", answer: "它是对数概率；要先 exp，再按百分比显示。"
  },
  {
    p: 65, slug: "name-classifier-summary", title: "姓名分类总结：按单模型跑通，再统一重构",
    problem: "面对数百行项目代码，零基础应按什么顺序练习才不会迷失？",
    chain: ["先只跑 RNN", "数据→模型→训练→预测", "再独立迁移 LSTM", "再迁移 GRU", "最后统一重构与优化"],
    extraVisuals: [projectPipelineDiagram],
    article: article([
      ["0:00–1:52", "不要三模型同时抄", "老师建议先忽略 LSTM/GRU，只用 RNN 从数据处理、建模、训练到预测完整跑一遍。先建立一条能闭环的主线。"],
      ["1:52–3:47", "再分别迁移", "RNN 熟练后，单独建立 LSTM 版本，再建立 GRU 版本；最后才把三者合并成课堂的统一对比版本。这样每次只改变一个模型因素，错误更好定位。"],
      ["3:48–4:59", "优化清单", "老师要求把可优化点真正落地：使用 CrossEntropyLoss 简化输出、直接返回 DataLoader、把损失/准确率/耗时持久化后再绘图，并改进其他重复代码。绘图函数可复用，但数据、模型、训练和预测应自己写。"],
    ]),
    points: ["先单模型闭环，再做抽象", "每次迁移只改必要部分", "优化应伴随测试而不是只改写法"],
    code: `stages = ["RNN 闭环", "LSTM 迁移", "GRU 迁移", "统一接口", "测试与持久化"]\nfor i,s in enumerate(stages,1): print(i,s)`,
    output: "按难度递增打印五个练习阶段。",
    pit: "一开始把三模型、绘图、优化全部混在一起，会让任何错误都难以定位。",
    quiz: "最适合零基础的第一步是什么？", answer: "只用 RNN 完整跑通数据处理→模型→训练→预测。"
  },
];

const beginnerExamples = {
  38: ["读“这部电影不太好看”时，看到“好看”不能忘记前面的“不”。RNN 的隐藏状态就是一张随阅读不断改写的临时便签。", "先只手推一个时间步：当前词向量 x_t 和旧便签 h_(t-1) 进入同一单元，得到新便签 h_t。"],
  39: ["“一句话对应一个标签”和“每个词对应一个标签”是输入输出关系；RNN、LSTM、GRU 是内部怎样保存记忆。它们是两套不同的分类角度。", "先给任务画输入和输出的数量，再决定内部单元，不要把 many-to-one 与 LSTM 当成同一级名称。"],
  40: ["展开图画出很多方框，像把同一个演员在不同时间的照片排成一列；演员没换，权重也没有换。", "在每次矩阵乘法前写维度，尤其分清 input_size 是当前词宽度，hidden_size 是记忆宽度。"],
  41: ["把 RNN 当成一个有两个出口的黑盒：output 给出每个时间步的结果，hn 只保存每层最后的状态。", "用 batch_first=True 后，牢牢记住输入 [B,L,Din]、output [B,L,H]、hn [layers,B,H]。"],
  42: ["一句话从 7 个词换成 12 个词，只是要多读 5 步，不应改变每个词的输入宽度和记忆宽度。", "改 seq_len 前先预测：output 的 L 会变，hn 没有 L 这一维，所以形状不随句长增加。"],
  43: ["hidden_size 像便签能写多少格。16 格改成 64 格，记忆和 output 的最后一维都会变，参数量也会增加。", "只改 hidden_size，逐项检查 output、hn 和后续 Linear 层的输入维是否同步。"],
  44: ["普通 RNN 的便签反复覆盖，长句前面的信息容易淡掉。LSTM 增加一条长期记忆通道，并用门决定忘掉什么、写入什么。", "第一次只跟踪 c：旧 c 先乘遗忘门，再加候选信息乘输入门；暂时不要背所有字母。"],
  45: ["长期记忆 c 像私人笔记，不会全部直接说出口；输出门决定当前时刻从笔记里展示多少，形成 h_t。", "按顺序读：更新 c_t → tanh 压缩 → 乘输出门 → 得 h_t，不要把 c_t 和 h_t 当同一个张量。"],
  46: ["判断句中“银行”的含义时，左边的“去”与右边的“办理贷款”都重要。双向 LSTM 让当前位置同时获得前文和后文。", "两个方向各产生 H 维，默认拼接后 output 最后一维是 2H；hn 的方向则在第一维分开保存。"],
  47: ["LSTM 接口比 RNN 多带一只箱子：h 是当前可见摘要，c 是更长期的细胞状态。", "运行前写出返回值 output,(hn,cn)，避免仍按 output,hn 两项去解包。"],
  48: ["GRU 也管理新旧记忆，但把 LSTM 的三扇门简化为更新门和重置门，并取消独立的 c。", "先抓更新门的直觉：它决定新状态中保留多少旧 h、写入多少候选信息。"],
  49: ["PyTorch 的 GRU 对外形状与普通 RNN 很像，所以替换层以后，主要变化在内部公式而非数据管道。", "把 nn.RNN 换成 nn.GRU 后，继续验证 output 和 hn；不要凭接口相似就跳过形状测试。"],
  50: ["姓名分类项目的输入是一个字符序列，输出是 18 个国家类别之一，本质是 many-to-one 分类。", "先写清 X、Y 和评价方式：X=姓名字符，Y=国家索引；这一步没说清前不要开始搭模型。"],
  51: ["字符表和国家表像两把固定尺子：字符列决定输入含义，国家顺序决定输出第几列代表哪个国家。", "保存字符表与 country 列表；训练、验证和预测时必须使用完全相同的顺序。"],
  52: ["CSV 中可能有空行、额外逗号或无法解析的字符。读数据不是“文件能打开”就完成，而是要把异常行变成可追踪的决定。", "先打印前几条 name,country 和异常计数，再确认标签是否都能映射到国家表。"],
  53: ["名字“Li”可以看成长度 2、每个字符 57 维的表，所以单样本张量形状是 [2,57]。", "在 __getitem__ 中按顺序检查原字符串→字符 ID→One-Hot→标签 ID，每一步都打印一个真实样本。"],
  54: ["不同姓名长度不一，不能直接堆成普通 [B,L,D]；最简单的教学方案是 batch_size=1，正式批量则需 padding 或 pack。", "先理解 DataLoader 只负责取样、打乱和组批，不会自动替你设计变长序列策略。"],
  55: ["模型输出 18 个分数。CrossEntropyLoss 可以直接接 logits；旧写法是先 LogSoftmax 得 log 概率，再交给 NLLLoss。", "把两条配对记住：logits+CrossEntropyLoss，或 log_probs+NLLLoss；不要重复做两次 LogSoftmax。"],
  56: ["姓名分类只需要一个最终标签，因此模型读完整个字符序列后，取最后时间步表示，再由 Linear 映射到 18 类。", "沿形状追踪 [B,L,57]→[B,L,H]→[B,H]→[B,18]，每个箭头都找到对应代码。"],
  57: ["测试随机输入不是测试准确率，而是做水管通水：先确保尺寸、设备和返回值能走通，再接真实数据。", "用一个很小的 B、L 手算期望形状；断言通过后才进入训练循环。"],
  58: ["RNN、LSTM、GRU 可以共用同一个分类头，但它们返回的状态结构不同，尤其 LSTM 多一个 c。", "把差异封装在模型内部，让外部统一只拿 [B,18] 分类结果。"],
  59: ["公平比较三种模型像让三名选手跑同一条赛道：数据划分、字符表、隐藏维和评价方式都要一致。", "先用同一批随机张量验证三者输出都是 [B,18]，再谈谁训练得更好。"],
  60: ["一次训练像批改一叠练习：清旧梯度、做预测、算错误、反向找责任、更新参数，然后处理下一批。", "把 zero_grad→forward→loss→backward→step 写成固定五步，并在第一批打印 loss 与形状。"],
  61: ["训练 LSTM 的总体五步不变，只是内部前向会同时维护 h 和 c；若模型内部初始化状态，外部循环不必到处处理它们。", "先复用已验证的训练协议，只替换模型，防止同时改数据、损失和优化器导致无法定位差异。"],
  62: ["GRU 训练仍是同一份试卷和评分标准。速度更快或参数更少不代表在所有数据上都必然更准。", "固定随机种子、数据顺序和指标，记录训练时间与验证准确率，而不是只看最后一次 loss。"],
  63: ["损失下降说明训练集上的错误在减小，验证准确率说明能否泛化，耗时说明成本；三条信息回答不同问题。", "绘图前先把每个 epoch 的原始数值保存下来，检查横轴是否对齐、三模型是否使用同一尺度。"],
  64: ["预测一个新姓名仍要经过训练时完全相同的字符规范化和编码，然后加载同结构权重，最后把类别索引翻译回国家名。", "按顺序检查 eval、no_grad、Top-k 索引和 exp；LogSoftmax 的负数不能直接当百分比。"],
  65: ["面对完整项目，先让 RNN 从数据到预测闭环，再逐个替换 LSTM、GRU。一次只改变一处，错误才有清楚来源。", "把项目拆成数据、模型、训练、预测四个可单测模块；最后才抽象公共代码和画对比图。"]
};

for (const lesson of lessons) {
  const [scene, focus] = beginnerExamples[lesson.p];
  const steps = lesson.chain.map((step, index) => `${index + 1}. ${step}`).join("\n");
  lesson.beginner = `### 先用一个场景理解

${scene}

### 沿数据流一步一步走

${steps}

上面每一步都对应流程图的一段。读图时不断问自己：“此刻张量里装的是什么，形状是什么，下一步为什么需要它？”

### 第一次看代码只盯住这里

${focus}

运行代码前先写出预期形状，运行后逐维核对。数值可以暂时算不出，但 B（批量）、L（长度）、D/H（特征或隐藏宽度）为什么出现，必须能说清。

### 本节边界

${lesson.pit}

本节过关不是背公式，而是能从第 1 步讲到最后一步，并指出哪一个状态把前文带到了后面。`;
}

generateTopic({
  topic: "RNN、LSTM、GRU 与姓名分类",
  folder: "rnn",
  lessons,
  stages: [
    { title: "一、普通 RNN：结构、形状与代码", start: 1, end: 6 },
    { title: "二、LSTM、双向与 GRU", start: 7, end: 12 },
    { title: "三、全球姓名分类项目", start: 13, end: 28 },
  ],
  codePackage: {
    label: "rnn_from_scratch 配套实现",
    href: "../../rnn_from_scratch/README.md",
    test: "python3 -m unittest rnn_from_scratch.test_model -v",
  },
});
