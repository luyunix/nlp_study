import { generateTopic } from "./course_note_helpers.mjs";

const article = (sections) => sections.map(([time, title, body]) =>
  `### ${time}　${title}\n\n${body}`
).join("\n\n");

const coreFlow = {
  title: "注意力的三步主流程",
  diagram: `
flowchart LR
    Q["Q：当前想找什么"] --> S["Q 与每个 K 算分"]
    K["K：每份内容的检索标签"] --> S
    S --> P["Softmax：变成总和为 1 的权重"]
    P --> W["按权重汇总 V"]
    V["V：真正要取的内容"] --> W
    W --> C["context：本次专属信息"]
`,
};

const qkvUml = {
  title: "Q、K、V 的职责 UML",
  diagram: `
classDiagram
    class Query_Q {
      当前需求
      决定本次查什么
    }
    class Key_K {
      可匹配的索引
      与Q计算相关性
    }
    class Value_V {
      实际内容向量
      被权重加权汇总
    }
    class Attention {
      score(Q,K)
      softmax(scores)
      weighted_sum(weights,V)
    }
    Query_Q --> Attention
    Key_K --> Attention
    Value_V --> Attention
`,
};

const seq2seqUml = {
  title: "Encoder、Attention、Decoder 的模块关系",
  diagram: `
classDiagram
    class Encoder {
      输入 token 序列
      输出每个位置 h1...hL
    }
    class Attention {
      Decoder状态作为Q
      Encoder状态作为K和V
      输出context
    }
    class Decoder {
      上一步输出
      上一步隐藏状态
      context
      预测下一个token
    }
    Encoder --> Attention
    Attention --> Decoder
    Decoder --> Attention : 下一时间步更新Q
`,
};

const shapeFlow = {
  title: "单查询注意力的形状链",
  diagram: `
flowchart LR
    Q["Q [B,H]"] --> U["unsqueeze → [B,H,1]"]
    K["K [B,L,H]"] --> B1["bmm(K,Q)"]
    U --> B1
    B1 --> S["scores [B,L]"]
    S --> P["weights [B,L]"]
    P --> U2["unsqueeze → [B,1,L]"]
    V["V [B,L,Dv]"] --> B2["bmm(weights,V)"]
    U2 --> B2
    B2 --> C["context [B,Dv]"]
`,
};

const lessons = [
  {
    p: 66, slug: "attention-introduction", title: "注意力机制介绍：把有限精力分给更相关的信息",
    problem: "模型面对一长串输入时，为什么不该把每个位置都看得一样重？",
    chain: ["大量输入信息", "当前任务产生需求", "计算各位置重要度", "动态分配权重", "汇总关键内容"],
    extraVisuals: [coreFlow],
    article: article([
      ["0:00–3:51", "本章会学什么", "老师先给路线：注意力概念、Q/K/V、实现步骤、Seq2Seq 中的用法、常见打分公式、bmm，再手写代码。注意力最早在机器翻译中大放异彩，随后成为 Transformer 的核心。"],
      ["3:51–8:39", "从人的视觉与学习说起", "人看图片时不会逐像素平均观察，而会先看脸、标题或当前关心的物体。老师把它概括为：用有限精力从海量信息中筛出高价值部分。注意力不是只留下一个位置，而是给不同位置不同权重。"],
      ["8:39–13:35", "RNN 的长序列与串行局限", "普通 RNN 依靠隐藏状态逐步压缩历史，序列很长时早期细节容易衰减，而且时间步必须串行。注意力让当前预测能直接回看编码器的多个位置，不必只依赖一个最终压缩状态。需要校正的是：注意力本身不自动让所有 Seq2Seq 计算完全并行；Transformer 去掉循环后才充分并行。"],
      ["13:35–18:31", "翻译时权重是动态的", "以 “welcome to Wuhan” 为例：生成“欢迎”时更关注 welcome，生成“来”时更关注 to，生成“武汉”时更关注 Wuhan。每个目标词都重新得到一组权重；它们不是手写常数，而是训练学出的概率分布。"],
      ["18:31–24:29", "一句话总结", "注意力让模型针对当前任务，对输入不同部分动态分配关注比例，再按比例汇总信息。软注意力给所有位置连续权重；硬注意力更像离散选择，后面单独区分。"],
    ]),
    points: ["权重随当前查询动态变化", "注意力缓解单一固定向量的信息瓶颈", "权重表示模型内部贡献，不等同因果解释"],
    code: `scores = [2.0, 0.5, -1.0]\nimport torch\nweights = torch.softmax(torch.tensor(scores), dim=0)\nprint(weights, weights.sum())`,
    output: "三个分数变成非负权重，且总和为 1；分数最高的位置权重最大。",
    pit: "不要把注意力权重直接解释为“模型作决定的唯一原因”。",
    quiz: "生成不同目标词时，输入端的注意力权重会一样吗？", answer: "通常不一样；查询变了，相关性和权重也会重新计算。"
  },
  {
    p: 67, slug: "qkv-introduction", title: "Q、K、V：问题、索引与实际内容",
    problem: "Q/K/V 三个字母分别做什么，为什么 K 和 V 要成对出现？",
    chain: ["Q 表达当前需求", "K 提供可匹配索引", "Q·K 得相关分", "权重选择对应 V", "得到专属信息"],
    extraVisuals: [qkvUml, coreFlow],
    article: article([
      ["0:00–2:49", "三个角色", "老师要求先记大白话：Q 是当前要解决的问题；K 是用来与 Q 匹配的索引；V 是匹配后真正取出的内容。它们分工协作完成“聚焦关键信息”。"],
      ["2:49–6:45", "文章搜索类比", "Q 是搜索词，K 可以是每篇文章的标题/标签，V 是文章内容。先比较 Q 与所有 K，标题越相关，对应文章 V 的权重越高。K 与 V 的序列位置必须一一对应。"],
      ["6:45–10:44", "翻译中的 Q/K/V", "解码器当前隐藏状态常充当 Q；编码器每个位置的隐藏状态常同时充当 K 和 V。Q 与每个 K 算分，再用概率加权所有 V，得到当前目标词专用的 context。"],
      ["10:44–13:51", "不是数据库查找", "注意力通常不是只取最高 K 对应的一条 V，而是软加权求和。Q/K/V 的维度可以通过不同线性层投影，三者也不要求原始来源完全不同。"],
    ]),
    points: ["Q=需求，K=匹配索引，V=实际内容", "K_i 与 V_i 必须位置对应", "软注意力会混合多条 V"],
    code: `items = [{"key":"财经", "value":"财经文章全文"}, {"key":"汽车", "value":"汽车文章全文"}]\nquery = "买车"\nprint(query, [x["key"] for x in items])`,
    output: "示例把检索需求、可匹配标签和真实内容分开。",
    pit: "V 不是 Q 与 K 的相似度；相似度只负责产生 V 的权重。",
    quiz: "为什么不能打乱 V 的顺序而保留 K 不动？", answer: "每个 K 的权重会套到错误内容 V 上，索引与内容失去对应。"
  },
  {
    p: 68, slug: "attention-steps", title: "注意力实现步骤：算分、归一化、加权求和",
    problem: "从 Q/K/V 到 context 的计算到底分哪三步，每步的形状是什么？",
    chain: ["Q 与每个 K 打分", "得到 scores", "Softmax 得 weights", "weights×V", "求和得 context"],
    extraVisuals: [coreFlow, shapeFlow],
    article: article([
      ["0:00–6:57", "第一步：匹配分", "对一个 Q 与 L 个 K 分别计算相关性，得到 L 个 scores。可用点积、线性拼接、加法网络等。分数可正可负，尚不是概率。"],
      ["6:57–13:57", "第二步：Softmax", "沿 L 个输入位置做 Softmax，得到非负且和为 1 的 weights。维度一定是序列位置维；若写错 dim，可能在 batch 或特征维归一化。"],
      ["13:57–22:57", "第三步：加权求和", "每个 weight_i 乘对应 V_i，再把 L 项相加，得到一个 context。context 的最后维度等于 V 的特征维 D_v。"],
      ["22:57–30:58", "为什么能训练", "初始权重可能乱，但预测损失通过反向传播更新打分函数和前面编码器，让与任务更相关的位置逐步得到更合适权重。"],
      ["30:58–37:25", "零基础形状口诀", "单查询：Q[B,H]，K[B,L,H]，scores/weights[B,L]，V[B,L,Dv]，context[B,Dv]。B 是批量，L 是输入长度，H 是匹配维度，Dv 是内容维度。"],
    ]),
    points: ["scores 不是概率", "Softmax 要沿输入位置维", "context 维度跟 V 的内容维度走"],
    code: `import torch\nq=torch.randn(2,8); k=torch.randn(2,5,8); v=torch.randn(2,5,6)\nscores=torch.bmm(k,q.unsqueeze(-1)).squeeze(-1)\nw=torch.softmax(scores,dim=-1)\ncontext=torch.bmm(w.unsqueeze(1),v).squeeze(1)\nprint(scores.shape,w.shape,context.shape)`,
    output: "scores/weights=[2,5]，context=[2,6]。",
    pit: "Q 和 K 的匹配维必须相同；V 的内容维可以不同。",
    quiz: "有 5 个输入词时，一个查询会有几个注意力权重？", answer: "5 个，每个输入位置一个。"
  },
  {
    p: 69, slug: "seq2seq-task", title: "Seq2Seq 任务：编码器把输入交给解码器逐词生成",
    problem: "输入和输出长度不同的翻译任务，Encoder 与 Decoder 各自负责什么？",
    chain: ["源语言序列", "Encoder 状态序列", "中间表示", "Decoder 逐步生成", "目标语言序列"],
    extraVisuals: [seq2seqUml],
    article: article([
      ["0:00–5:59", "句子到句子的任务", "老师用机器翻译说明 Seq2Seq：输入 N 个词，输出 M 个词，N 与 M 可以不同。也可用于摘要、对话等。"],
      ["5:59–12:55", "Encoder", "编码器逐词读取源句，产生隐藏状态。普通框架常只把最终状态压成固定中间向量 C 交给解码器。"],
      ["12:55–21:53", "Decoder", "解码器用 C、上一步隐藏状态和上一步生成词预测下一词。训练时常用真实上一词（teacher forcing），推理时只能使用自己刚生成的词。"],
      ["21:53–29:56", "固定向量瓶颈", "所有源句信息都塞进一个 C，长句容易丢细节。注意力的动机正是让解码器每一步直接查看全部编码器状态。"],
      ["29:56–33:59", "边界符", "实际序列需要 SOS/EOS/PAD 等特殊 token，解码从 SOS 开始，生成 EOS 停止。"],
    ]),
    points: ["Encoder 读源句，Decoder 生成目标句", "N 与 M 不必相等", "训练与推理的上一词来源不同"],
    code: `source=["welcome","to","Wuhan"]\ntarget=["欢迎","来","武汉","<EOS>"]\nprint(len(source),len(target))`,
    output: "源长 3、目标长 4，说明 Seq2Seq 不要求等长。",
    pit: "不能用目标句未来词帮助当前推理，否则发生信息泄漏。",
    quiz: "解码为什么需要 EOS？", answer: "输出长度未知，模型用 EOS 表示生成结束。"
  },
  {
    p: 70, slug: "seq2seq-with-attention", title: "Seq2Seq 加入注意力：每个目标词拥有自己的 context",
    problem: "固定 C 为什么不够，C₁、C₂、C₃ 又怎样随解码步变化？",
    chain: ["编码器保存 h1…hL", "当前解码状态作 Q", "算本步 weights", "得到本步 C_t", "预测目标 token"],
    extraVisuals: [seq2seqUml, coreFlow],
    article: article([
      ["0:00–4:58", "固定 C 的问题", "普通 Encoder-Decoder 每次预测都使用同一个 C，等于认为所有源词对所有目标词贡献相同。长句信息被挤进一个向量。"],
      ["4:58–9:58", "动态 C_t", "加入注意力后，第 1、2、3 个目标词分别使用 C_1、C_2、C_3。每个 C_t 都由当前解码状态与编码器各位置重新打分得到。"],
      ["9:58–14:57", "翻译对齐直觉", "生成“欢迎”时 welcome 权重大，生成“来”时 to 权重大；但仍会给其他词小权重以处理短语、一词多义和语序变化。"],
      ["14:57–18:03", "计算闭环", "Decoder 状态→Q；Encoder 状态→K/V；Softmax 得权重；加权 V 得 C_t；Decoder 用 C_t 和历史生成新词，下一步状态再形成新 Q。"],
    ]),
    points: ["每个解码步都有独立 context", "注意力是软对齐，不要求一一对应", "当前 Decoder 状态决定本步查询"],
    code: `contexts=["C_欢迎","C_来","C_武汉"]\nfor step,c in enumerate(contexts,1): print(step,c)`,
    output: "三个目标时间步明确使用不同 context。",
    pit: "注意力不要求源词数与目标词数相同。",
    quiz: "为什么生成下一个词时权重会变化？", answer: "Decoder 状态和已生成历史变化，使 Q 变化。"
  },
  {
    p: 71, slug: "plain-encoder-decoder", title: "普通 Encoder-Decoder：一个 C 服务所有解码步骤",
    problem: "在加入注意力前，先准确看懂传统框架的信息流和瓶颈。",
    chain: ["Encoder 逐词编码", "最终状态形成 C", "Decoder 接收 C", "结合历史状态", "逐步输出 y1…yM"],
    extraVisuals: [seq2seqUml],
    article: article([
      ["0:00–3:58", "编码器的非线性压缩", "老师把 Encoder 看作 RNN/GRU：每步输入词向量并更新隐藏状态，最终得到固定中间语义向量 C。"],
      ["3:58–6:55", "解码器依赖 C 与历史", "预测 y_1 用 C 与初始状态；预测 y_2 还要带 y_1 的信息；预测第 10 个词要带前 9 个生成历史。"],
      ["6:55–8:11", "关键判断", "这仍没有注意力，因为每个解码步拿到的 C 完全相同。把普通框架看懂，才能看到后面唯一重要变化：C 从固定变为动态。"],
    ]),
    points: ["普通框架只传固定 C", "Decoder 仍有自回归历史", "固定 C 是主要信息瓶颈"],
    code: `fixed_c="same context"\nprint([fixed_c for _ in range(3)])`,
    output: "三个解码步拿到完全相同的 C。",
    pit: "“没有注意力”不等于 Decoder 没有隐藏状态。",
    quiz: "普通 Seq2Seq 每步变化的是什么、不变的是什么？", answer: "Decoder 历史状态变化；传入的固定 C 不变。"
  },
  {
    p: 72, slug: "attentive-encoder-decoder", title: "带注意力 Encoder-Decoder：从公式看 Cᵢ 怎样生成",
    problem: "Cᵢ = Σ αᵢⱼ hⱼ 中每个下标和函数到底表示什么？",
    chain: ["Encoder 得 h_j", "Decoder 第 i 步状态", "算 α_ij", "Σ α_ij h_j", "得到 C_i 并生成 y_i"],
    extraVisuals: [seq2seqUml, coreFlow],
    article: article([
      ["0:00–4:54", "同一个翻译例子", "老师用 Tom / chase / Jerry 展示：普通框架贡献相同不合理，注意力给三个源词 0.6/0.2/0.2 等权重。数字只是示意，真实权重由模型计算。"],
      ["4:54–8:51", "F1、F2、g 的大白话", "F2 表示 Encoder 对源词的编码，得到 h_j；α_ij 表示生成第 i 个目标词时，第 j 个源位置的重要程度；g 是加权求和。F1 表示 Decoder 根据 C_i 与历史生成新词的规则。"],
      ["8:52–14:50", "公式逐项读", "C_i = Σ_{j=1}^{Lx} α_ij h_j。Lx 是源句长度；h_j 是第 j 个源词的编码；α_ij 是本目标步对该源词的权重；求和后得到一个与源长度无关的固定维 context。"],
      ["14:50–20:53", "权重矩阵", "若目标 3 步、源 3 词，所有 α 可排成 3×3 矩阵：每一行对应一个目标步，每行通常和为 1。不是说翻译只能 3 对 3。"],
      ["20:53–24:40", "C₁、C₂、C₃ 的关系", "三个 context 使用相同计算规则，但 Decoder 查询不同，所以权重行不同，最终 C 不同。"],
    ]),
    points: ["i 指目标位置，j 指源位置", "每个 C_i 是全部 h_j 的加权和", "权重矩阵形状是 [目标长度,源长度]"],
    code: `import torch\nalpha=torch.tensor([.6,.2,.2])\nh=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])\nprint(alpha @ h)`,
    output: "得到本目标步的 2 维 context：[0.8,0.4]。",
    pit: "α_ij 是标量权重，h_j 是向量；不要把两者维度混为一谈。",
    quiz: "目标长 5、源长 3，注意力矩阵多大？", answer: "5×3，每个目标步一行、每个源位置一列。"
  },
  {
    p: 73, slug: "attention-probabilities", title: "注意力概率分布：Decoder 状态如何与所有 Encoder 状态比较",
    problem: "示例里的 0.6、0.2、0.2 是怎样一步步算出来的？",
    chain: ["Encoder 状态 h1…hL", "当前 Decoder 状态 s", "逐位置 score(s,hj)", "Softmax", "权重加权 h_j 得 C"],
    extraVisuals: [seq2seqUml, coreFlow],
    article: article([
      ["0:00–4:54", "核心问题", "老师明确要算的是：生成当前目标词时，应重点关注 Encoder 的哪些词。源长与目标长可以不同，不能靠位置硬对齐。"],
      ["4:54–10:52", "两边的隐藏状态", "Encoder 每个 h_j 同时含当前词表示和左侧上下文；Decoder 当前 s_i 含已生成历史。注意力拿 s_i 与每个 h_j 比较。"],
      ["10:52–13:42", "第一、二步", "score(s_i,h_j) 得到匹配分；把所有 j 的分数组成向量，经 Softmax 得 α_i1…α_iL。分数高的位置概率通常高。"],
      ["13:44–16:42", "第三步", "用 α_ij 给 Encoder 状态加权求和得到专属 C_i。每个 Decoder 步的 s_i 不同，因此分数、概率、C_i 都会变。"],
      ["16:42–21:22", "面试式总结", "一：当前 Decoder 状态与所有 Encoder 状态算匹配；二：Softmax 转成关注比例；三：用比例加权 Encoder 状态得到当前步 context，再交给 Decoder 生成词。"],
    ]),
    points: ["源/目标不需要位置对齐", "Decoder 当前状态决定查询", "Softmax 权重沿源位置求和为 1"],
    code: `import torch\nscores=torch.tensor([[2.0,.5,.5]])\nweights=torch.softmax(scores,dim=-1)\nprint(weights,weights.sum(-1))`,
    output: "得到一行三列的概率，和为 1。",
    pit: "Softmax 前的匹配分不需要位于 0~1。",
    quiz: "为什么下一目标步要重新算概率？", answer: "Decoder 状态变了，代表当前需求的 Q 变了。"
  },
  {
    p: 74, slug: "attention-types", title: "软注意力、硬注意力与自注意力",
    problem: "三种“注意力”名称分别按什么标准区分，最容易混在哪里？",
    chain: ["判断权重连续或离散", "软/硬注意力", "判断 QKV 来源", "自注意力", "选择可训练方式"],
    extraVisuals: [qkvUml],
    article: article([
      ["0:00–4:51", "软注意力", "前面一直使用软注意力：对所有源位置给连续权重，加权求和，整体可微，能用普通反向传播训练。"],
      ["4:51–8:41", "硬注意力", "硬注意力更像选中一个/少数位置，权重近似 0/1。离散采样通常不可直接求导，训练可能需要强化学习、采样估计或可微近似。课程只作了解。"],
      ["8:41–11:36", "自注意力", "自注意力按 Q/K/V 是否来自同一序列区分。常见 Q=XW_Q、K=XW_K、V=XW_V；来源相同但投影参数不同，所以“Q=K=V”更准确地说是来自同一 X，不一定数值完全相等。"],
      ["11:36–14:54", "两个维度别混", "软/硬描述权重选择方式；自注意力描述信息来源。自注意力通常也是软注意力，因此这些名称不是互斥的同一分类表。"],
    ]),
    points: ["软/硬看权重方式", "自注意力看 QKV 来源", "自注意力中的 Q/K/V 常由同一 X 不同投影得到"],
    code: `print("soft: [0.6,0.2,0.2]")\nprint("hard: [1,0,0]")\nprint("self: Q,K,V all projected from X")`,
    output: "三行分别显示分类依据。",
    pit: "“Q=K=V”不代表经过 W_Q/W_K/W_V 后三个张量数值仍完全相同。",
    quiz: "自注意力能同时也是软注意力吗？", answer: "可以，而且 Transformer 的常见自注意力就是连续 Softmax 权重。"
  },
  {
    p: 75, slug: "attention-scoring-rules", title: "常见注意力计算规则：拼接式、加法式与缩放点积",
    problem: "不同打分公式只是写法不同吗，为什么缩放点积要除以 √dₖ？",
    chain: ["选 score(Q,K)", "得到 logits", "可选缩放/非线性", "Softmax", "权重×V"],
    extraVisuals: [coreFlow],
    article: article([
      ["0:00–4:47", "共同主干", "无论公式怎样，主干都一样：Q 与 K 算相关分→Softmax→概率乘 V→汇总。一般注意力常见拼接/加法打分；自注意力常用缩放点积。"],
      ["4:47–11:43", "拼接式打分", "把 Q 与 K_j 拼接，经 Linear 得标量分：score=vᵀtanh(W[Q;K_j]) 等。课堂把“拼接→线性变换→Softmax→乘 V”拆成三部分。"],
      ["11:43–16:27", "加法/非线性变体", "可以在打分网络中加入 tanh、额外线性层和求和，让函数学习更复杂关系。网络更深不保证更好，需看任务与计算成本。"],
      ["16:46–22:55", "缩放点积", "score=QKᵀ/√d_k。点积把对应维相乘后求和；d_k 大时点积幅度容易变大，使 Softmax 极端饱和、梯度变小。除以 √d_k 缩小尺度，使概率较稳定。"],
      ["22:55–25:37", "分类口径纠正", "课堂用 Q/K/V 相等区分自注意力；更严谨是它们来自同一序列的投影。前两种也可用于自注意力，第三种也能用于交叉注意力；公式与注意力来源不是严格绑定。"],
    ]),
    points: ["打分函数输出 logits", "缩放防止大维点积导致 Softmax 饱和", "打分公式与 QKV 来源是两个选择"],
    code: `import math, torch\nq=torch.randn(2,8); k=torch.randn(2,5,8)\nscores=torch.bmm(k,q.unsqueeze(-1)).squeeze(-1)/math.sqrt(8)\nprint(scores.shape)`,
    output: "每个查询对 5 个 key 得 5 个缩放分数。",
    pit: "√d_k 中的 d_k 是 Q/K 匹配维，不是序列长度。",
    quiz: "为什么不是除以 d_v？", answer: "Softmax 前的点积发生在 Q/K 匹配空间，尺度由 d_k 决定。"
  },
  {
    p: 76, slug: "bmm", title: "bmm：一次完成一批三维矩阵乘法",
    problem: "torch.bmm 与 torch.matmul 有什么差别，注意力为什么常用它？",
    chain: ["准备 [B,N,M]", "准备 [B,M,P]", "同批逐项相乘", "得到 [B,N,P]", "不做 batch 广播"],
    extraVisuals: [shapeFlow],
    article: article([
      ["0:00–2:53", "为什么先补 bmm", "注意力要对一个 batch 的矩阵批量做乘法。老师在写模块前先比较 torch.matmul 与 torch.bmm。"],
      ["2:55–7:54", "matmul 更通用", "matmul 支持多种维度，并可对 batch 维广播。例如 [10,3,4] 与 [1,4,5] 可以广播为 10 批得到 [10,3,5]。"],
      ["7:54–11:52", "bmm 的严格输入", "bmm 专门接受两个三维张量 [B,N,M] 与 [B,M,P]，batch B 必须相等，不做广播，输出 [B,N,P]。表达意图明确。速度优势要在实际硬件和规模上基准测试，不能绝对断言总更快。"],
      ["11:52–18:45", "课堂实验", "老师构造 [10,3,4] 与 [10,4,5]，两者都得 [10,3,5]；把第二个 batch 改 1，matmul 可广播，bmm 报 batch 不匹配。"],
      ["18:45–20:01", "在注意力中的位置", "K[B,L,H]×Q[B,H,1]→scores[B,L,1]；weights[B,1,L]×V[B,L,Dv]→context[B,1,Dv]，正好都是三维批量乘法。"],
    ]),
    points: ["bmm 只接三维", "batch 维必须一致", "内侧矩阵维必须匹配"],
    code: `import torch\na=torch.randn(10,3,4); b=torch.randn(10,4,5)\nprint(torch.bmm(a,b).shape)`,
    output: "输出 [10,3,5]。",
    pit: "bmm 不会把 batch=1 自动广播到 batch=10。",
    quiz: "[B,1,L] bmm [B,L,D] 输出什么？", answer: "[B,1,D]。"
  },
  {
    p: 77, slug: "attention-code", title: "注意力代码实现：从 Q/K/V 到增强后的查询",
    problem: "怎样把注意力三步写成可训练 nn.Module，并逐层追踪形状？",
    chain: ["定义打分层", "Q 与每个 K 算分", "Softmax 得 weights", "bmm 汇总 V", "拼 Q/context 后投影"],
    extraVisuals: [shapeFlow, qkvUml],
    article: article([
      ["0:00–3:52", "任务与模块接口", "课程把单个 Q 变成包含上下文的“增强 Q”。forward 接 Q、K、V；初始化接 query_size、key_size、value_size 与 output_size。示例后来统一为批量张量。"],
      ["3:52–10:57", "两层 Linear", "第一层用于 Q/K 打分；第二层把原 Q 与 context 拼接后投影到输出维。参数必须按实际输入最后一维定义。"],
      ["11:56–18:54", "零基础形状设定", "Q[B,Hq]；K[B,L,Hk]；V[B,L,Dv]。若 Hq≠Hk，需投影到共同匹配维。课程部分口述把 K 说成 [1,32]，但对 32 个词逐一打分时应保留序列维 [1,32,Hk]。"],
      ["18:54–23:59", "算分与概率", "将 Q 扩展到每个 key 位置，或使用点积/general attention，一次得到 scores[B,L]。Softmax(dim=-1) 得 weights[B,L]。"],
      ["23:59–28:04", "bmm 加权 V", "weights.unsqueeze(1) 变 [B,1,L]，与 V[B,L,Dv] 做 bmm，得 context[B,1,Dv]，squeeze 后 [B,Dv]。"],
      ["28:04–31:17", "增强查询", "拼接 Q[B,Hq] 与 context[B,Dv] 得 [B,Hq+Dv]，经 Linear 输出 [B,output_size]。同时返回 weights 方便可视化与检查。"],
    ]),
    points: ["K/V 必须保留序列维 L", "Softmax 沿 L", "最终投影可把拼接维降回模型隐藏维"],
    code: `import torch\nfrom attention_from_scratch.model import DotProductAttention\nq=torch.randn(2,8); k=torch.randn(2,5,8); v=torch.randn(2,5,8)\ncontext,weights=DotProductAttention()(q,k,v)\nprint(context.shape,weights.shape)`,
    output: "context=[2,8]，weights=[2,5]。",
    pit: "把 K 写成 [B,H] 会只剩一个 key，注意力失去在多个位置间选择的意义。",
    quiz: "为什么 weights 要先 unsqueeze(1)？", answer: "bmm 需要三维 [B,1,L]，才能与 [B,L,Dv] 相乘。"
  },
  {
    p: 78, slug: "attention-test", title: "注意力测试代码：检查形状、概率和与梯度",
    problem: "怎样证明模块不只“能跑”，还满足注意力的基本性质？",
    chain: ["创建尺寸参数", "造 Q/K/V", "调用模块", "检查 shape", "检查权重和与 mask"],
    extraVisuals: [shapeFlow],
    article: article([
      ["0:00–1:54", "测试数据", "老师设置查询/键维 32、32 个词、V 维 64，并创建随机 Q、K、V。随机值只用于结构测试。"],
      ["1:54–3:48", "运行与打印", "实例化自定义模块，得到 output 与 attention weights。output 是增强表示，weights 对每个词给一个概率。"],
      ["3:48–5:55", "必须验证的性质", "若 L=32，weights 形状应 [B,32]，每行和约等于 1；output 形状应与配置输出维一致。还应测试 padding mask 后无效位置权重为 0，以及 backward 能产生有限梯度。"],
    ]),
    points: ["概率和约等于 1", "mask 后 PAD 权重为 0", "测试要覆盖 batch>1"],
    code: `import torch\nfrom attention_from_scratch.model import DotProductAttention\nq=torch.randn(2,8); k=torch.randn(2,5,8); v=torch.randn(2,5,8)\nc,w=DotProductAttention()(q,k,v)\nprint(c.shape,w.shape,w.sum(-1))`,
    output: "context=[2,8]、weights=[2,5]，两行权重和都为 1。",
    pit: "只用 batch=1 很容易隐藏 squeeze 把 batch 维误删的 bug。",
    quiz: "为什么用 torch.allclose 而不是 ==1？", answer: "浮点运算有微小舍入误差，应在容差内比较。"
  },
  {
    p: 79, slug: "attention-parameters", title: "参数解释：把 1×32、32×64、1×96 全部读成语义",
    problem: "面对一串形状数字，怎样不靠背诵而从 B、L、H、Dᵥ 推导？",
    chain: ["给每一维命名", "Q/K 算 [B,L]", "weights×V 得 [B,Dv]", "拼接 Q/context", "Linear 降到输出维"],
    extraVisuals: [shapeFlow, seq2seqUml],
    article: article([
      ["0:00–5:52", "先回顾完整公式", "QK 先算匹配分；可选缩放与 mask；Softmax；再乘 V 得 context。mask 防止看到 padding 或因果未来位置，后面 Transformer 继续展开。"],
      ["5:52–13:46", "图中的 Q、K、V", "在解码器生成当前词时，上一步隐藏状态/当前查询是 Q；Encoder 的各位置状态是 K/V。每个 key 都与 Q 比较，权重再汇总 value。"],
      ["13:47–20:43", "加法式与矩阵式", "逐位置打分再相加和批量矩阵乘法都实现同一概念。矩阵实现避免 Python 循环、利于硬件并行；不能简单归因于框架品牌。"],
      ["20:43–28:39", "逐步推导示例", "设 B=1、L=32、Hq=Hk=32、Dv=64。Q=[1,32]；K=[1,32,32]；V=[1,32,64]；scores/weights=[1,32]；context=[1,64]。"],
      ["28:39–34:56", "拼接与降维", "Q[1,32] 与 context[1,64] 拼成 [1,96]，Linear(96,32) 得增强 Q[1,32]。这里的 96 不是神秘参数，而是 32+64。老师最后反复让同学把每个数字对应回批量、词数和特征维。"],
    ]),
    points: ["1=batch，32 可能是 L 或 H，必须看位置", "96=查询维32+内容维64", "Linear 改最后一维，不改 batch"],
    code: `B,L,H,Dv=1,32,32,64\nprint("Q",(B,H),"K",(B,L,H),"V",(B,L,Dv),"concat",(B,H+Dv))`,
    output: "打印每个张量的语义形状，concat 为 [1,96]。",
    pit: "同一个数字 32 可能代表词数，也可能代表特征维，不能只看数值判断。",
    quiz: "B=4、L=10、H=8、Dv=16 时 context 形状？", answer: "[4,16]；序列维已被加权求和。"
  },
];

generateTopic({
  topic: "注意力机制与 Seq2Seq 框架",
  folder: "attention",
  lessons,
  stages: [
    { title: "一、注意力直觉与 Q/K/V", start: 1, end: 3 },
    { title: "二、在 Seq2Seq 中逐步定位信息", start: 4, end: 8 },
    { title: "三、类型、公式、bmm 与完整代码", start: 9, end: 14 },
  ],
  codePackage: {
    label: "attention_from_scratch 配套实现",
    href: "../../attention_from_scratch/README.md",
    test: "python3 -m unittest attention_from_scratch.test_model -v",
  },
});
