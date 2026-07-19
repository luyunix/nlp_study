import { generateTopic } from "./course_note_helpers.mjs";

const article = (sections) => sections.map(([time, title, body]) =>
  `### ${time}　${title}\n\n${body}`
).join("\n\n");

const modelFamilies = {
  title: "Encoder-only 与 Decoder-only",
  diagram: `
flowchart LR
    T["Transformer"] --> E["Encoder-only"]
    T --> D["Decoder-only"]
    E --> B["BERT：双向理解、抽取与分类"]
    D --> G["GPT：因果注意力、自回归生成"]
`,
};

const bertOverview = {
  title: "BERT 从输入到预训练目标",
  diagram: `
flowchart LR
    A["文本或句对"] --> B["Token + Position + Segment Embedding"]
    B --> C["多层双向 Transformer Encoder"]
    C --> D["每个 token 的上下文表示 [B,L,H]"]
    D --> E["MLM：还原被遮罩 token"]
    D --> F["NSP：判断句 B 是否紧接句 A"]
`,
};

const bertUml = {
  title: "BERT 模块 UML",
  diagram: `
classDiagram
    class BertModel {
      embeddings
      encoder
      pooler
      forward(input_ids, attention_mask, token_type_ids)
    }
    class BertEmbeddings {
      word_embeddings
      position_embeddings
      token_type_embeddings
      layer_norm
    }
    class BertEncoder {
      TransformerLayer x N
    }
    class MLMHead {
      vocabulary_projection
    }
    class NSPHead {
      binary_classifier
    }
    BertModel *-- BertEmbeddings
    BertModel *-- BertEncoder
    BertModel --> MLMHead
    BertModel --> NSPHead
`,
};

const embeddingFlow = {
  title: "BERT 输入三种 Embedding",
  diagram: `
flowchart TB
    W["Token Embedding：这个 token 是谁"] --> S["逐元素相加 [B,L,H]"]
    P["Position Embedding：它在第几个位置"] --> S
    T["Segment Embedding：属于句 A 还是句 B"] --> S
    S --> N["LayerNorm + Dropout"]
    N --> E["Encoder Layers"]
`,
};

const pretrainSequence = {
  title: "BERT 一批预训练数据的时序",
  diagram: `
sequenceDiagram
    participant D as 原始句对
    participant C as MLM/NSP 数据构造
    participant B as BertForPreTraining
    participant M as MLM Loss
    participant N as NSP Loss
    D->>C: 连续文本
    C-->>B: input_ids / masks / segments / 两类 labels
    B-->>M: prediction_logits [B,L,V]
    B-->>N: seq_relationship_logits [B,2]
    M-->>B: MLM loss
    N-->>B: NSP loss
    B-->>B: 总损失反向传播
`,
};

const contextualCompare = {
  title: "静态词向量与 ELMo 动态词向量",
  diagram: `
flowchart TB
    A["静态词向量：同一词查同一向量"] --> X["bank 在所有句中相同"]
    B["ELMo：字符表示 + 双向语言模型"] --> Y["river bank：河岸语境向量"]
    B --> Z["bank account：银行语境向量"]
`,
};

const elmoUml = {
  title: "ELMo 双向语言模型",
  diagram: `
flowchart LR
    C["字符 CNN 词表示"] --> F1["前向 LSTM 第 1 层"]
    F1 --> F2["前向 LSTM 第 2 层"]
    C --> B1["后向 LSTM 第 1 层"]
    B1 --> B2["后向 LSTM 第 2 层"]
    C --> W["按任务学习各层加权和"]
    F1 --> W
    F2 --> W
    B1 --> W
    B2 --> W
    W --> O["上下文化词向量"]
`,
};

const gptFlow = {
  title: "GPT 自回归生成时序",
  diagram: `
sequenceDiagram
    participant P as Prompt tokens
    participant D as Decoder-only Transformer
    participant S as Sampling/Argmax
    P->>D: 已知前缀
    D-->>S: 最后位置 next-token logits
    S-->>P: 追加一个 token
    loop 直到 EOS 或长度上限
      P->>D: 更长前缀（可复用 KV cache）
      D-->>S: 下一个 token 分布
      S-->>P: 再追加
    end
`,
};

const lessons = [
  {
    p: 184, slug: "bert-introduction", title: "BERT 介绍：为什么它能成为通用的文本理解底座",
    problem: "Word2Vec 和单向语言模型已经能学词义，为什么还需要 BERT 的深层双向上下文表示？",
    chain: ["静态词向量的多义词问题", "双向读取上下文", "大语料预训练", "加入小任务头", "下游微调"],
    extraVisuals: [bertOverview, contextualCompare],
    article: article([
      ["0:00–4:56", "名字、时间与公开基准", "BERT 全称是 Bidirectional Encoder Representations from Transformers，可理解为“来自 Transformer 的双向编码器表示”。老师强调它由 Google 在 2018 年提出，底层依赖 2017 年 Transformer 和注意力机制，并在阅读理解、GLUE 等公开基准上取得当时非常突出的成绩。公开基准的意义是统一数据和评测，避免只在自制数据上宣布“100%”。"],
      ["4:56–10:52", "从神经语言模型到 BERT 的时间线", "课堂沿时间线回顾：神经语言模型、2013 年 Word2Vec/词表示、RNN/Seq2Seq、2015 年 Attention、2017 年 Transformer、2018 年 ELMo/GPT/BERT。老师借此说明算法、互联网积累的数据和算力共同成熟后，预训练模型才快速发展。年份用于建立脉络，不必把每个模型名称都死背。"],
      ["10:52–18:46", "BERT 的主线：深度双向", "传统单向语言模型从左到右或从右到左，BiLSTM/ELMo 可把两方向结果拼接；BERT 的关键是多层 Transformer Encoder 中的深度双向上下文。BERT Base 常见配置是 12 层、隐藏维 768、12 个注意力头；每头 64 维。模型每个 token 都能根据左右文得到动态表示。"],
      ["18:46–26:36", "老师用“撕开小说”解释只看半边", "课堂用多人把小说撕成几份、有人先看结局的故事说明：只读左半边或右半边，语义理解有限；同时掌握前后文更容易推断中间词。这个类比对应 BERT 深度双向，但技术上并不是先完整阅读再把答案泄露给模型，而是用遮罩任务防止目标 token 直接可见。"],
      ["26:36–28:56", "用 MLM 与 NSP 实现预训练", "为避免模型在双向上下文中直接看到待预测词，BERT 用 Masked Language Modeling；经典 BERT 还用 Next Sentence Prediction 学句间关系。老师把后两节的学习目标收束为：会解释深度双向、MLM 和 NSP，并与 ELMo 的较浅双向方式区分。"],
    ]),
    points: ["BERT 是 Encoder-only 双向上下文模型", "预训练参数通过任务头迁移到下游", "同一 token 的向量会随上下文变化"],
    code: `import torch
from transformers import AutoTokenizer, AutoModel
path="your-bert-checkpoint"
tok=AutoTokenizer.from_pretrained(path)
model=AutoModel.from_pretrained(path).eval()
x=tok(["他去银行办理业务","他坐在河岸边"],padding=True,return_tensors="pt")
with torch.no_grad(): h=model(**x).last_hidden_state
print(h.shape)`,
    output: "得到 `[2,L,H]` 上下文化表示；同一词在两句中的向量通常不同。",
    pit: "把 BERT 说成 Transformer 的完整编码器+解码器；标准 BERT 只使用 Encoder 堆叠。",
    quiz: "为什么 BERT 比静态词向量更能处理多义词？", answer: "每个 token 表示由当前整句上下文动态计算，而不是永久查同一张词向量表。"
  },
  {
    p: 185, slug: "bert-architecture", title: "BERT 架构：三种 Embedding、Encoder 堆叠与关键形状",
    problem: "一对句子的 token ID 怎样经过三种 Embedding 和多层 Encoder，变成可供各种任务使用的表示？",
    chain: ["拼 [CLS] A [SEP] B [SEP]", "三种 Embedding 相加", "多头自注意力", "前馈网络与残差归一化", "输出 [B,L,H]"],
    extraVisuals: [bertUml, embeddingFlow],
    article: article([
      ["0:00–1:55", "特殊 token 与句段", "句对通常拼为 `[CLS] 句A [SEP] 句B [SEP]`。课堂把 `[CLS]` 解释为分类/汇总标记、`[SEP]` 解释为分隔/结束标记。注意 `[SEP]` 不是整个模型的 EOS 生成停止符，它主要做输入边界。"],
      ["1:55–4:52", "三种 Embedding 为什么相加", "Token Embedding 表示 token 身份；Segment/Token Type Embedding 表示 A 段或 B 段；Position Embedding 表示位置。三者形状都是 `[B,L,H]`，逐元素相加，不是拼接。若 `[1,32,768]`，含义是 1 条序列 × 32 个 token 位置 × 每位置 768 维。老师还顺带提到后来模型会使用旋转位置编码等改进，但标准 BERT 用学习式绝对位置嵌入。"],
      ["4:52–7:49", "只使用 Transformer Encoder", "BERT 不使用原始 Transformer Decoder。多层 Encoder 的自注意力是双向的；每层还含前馈网络、残差和归一化。主体输出 `last_hidden_state [B,L,H]`。"],
      ["7:49–10:54", "四类下游任务头", "老师按原论文图解释：句对分类（相似、蕴含/矛盾等）、单句分类（情感）、抽取式问答（起止位置）、单句 token 标注（NER）。底部预训练 BERT 相同，顶部任务头和输出形状不同；微调时复用主体、按任务改变顶层。"],
    ]),
    points: ["三种 Embedding 相加而非拼接", "Encoder 输出每个 token 的上下文表示", "不同任务头读取同一主体的不同粒度"],
    code: `from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("your-bert-checkpoint")
batch=tok("句子A","句子B",return_tensors="pt")
for k,v in batch.items():
    print(k,tuple(v.shape),v.tolist())`,
    output: "通常看到 input_ids、attention_mask、token_type_ids，形状均为 `[1,L]`。",
    pit: "以为 token_type_ids=1 表示有效 token；有效/PAD 由 attention_mask 表示，段别由 token_type_ids 表示。",
    quiz: "为什么三种 Embedding 可以相加？", answer: "它们为同一批、同一位置都提供 H 维表示，形状一致；相加把身份、位置和段信息融合进同一向量空间。"
  },
  {
    p: 186, slug: "bert-mlm-nsp", title: "BERT 预训练任务：MLM 与 NSP 怎样共同制造监督",
    problem: "没有人工标签的大规模文本，BERT 从哪里得到可计算的训练目标？",
    chain: ["原始连续文本", "随机遮罩 token", "构造真假句对", "同时预测词与句间关系", "联合损失更新 Encoder"],
    extraVisuals: [bertOverview, pretrainSequence],
    article: article([
      ["0:00–6:30", "MLM：从原文自己造标签", "随机选取一部分 token 作为预测目标，常见 BERT 方案约 15%。选中位置中约 80% 换 `[MASK]`、10% 换随机 token、10% 保持原 token；labels 保存原 token ID，其他位置设忽略值。模型输出 `[B,L,V]`，只在选中位置算词表交叉熵。"],
      ["6:30–12:30", "为什么不能把输入全遮住", "模型需要上下文才能推断目标；全部遮罩会丢失句意。只替换 MASK 又会造成预训练看到特殊符号、下游不看到的差距，因此混入随机词和原词。经典比例是设计选择，不是自然定律。"],
      ["12:30–18:30", "NSP：句 B 是否真接句 A", "从文档取句 A；一半用真实下一句 B，一半换随机句，标签常为 IsNext/NotNext。输入 `[CLS] A [SEP] B [SEP]`，句对头输出 `[B,2]`。NSP 希望学习句间关系，但负样本过简单会让模型走主题匹配捷径。后续 RoBERTa 等研究说明去掉 NSP 也能很好，因此不要把 NSP 当成所有 BERT 类模型不可缺少的真理。"],
      ["18:30–24:00", "联合损失", "这里先说明公式解决什么问题：同一 Encoder 要同时学 token 级恢复和句对级关系。经典训练把两项损失相加：`L = L_MLM + L_NSP`（也可带权）。反向传播让共享 Encoder 同时服务两个目标。预训练完成后通常丢弃这两个头，换上下游任务头。"],
    ]),
    points: ["MLM 与 NSP 标签都可从原始文本自动构造", "MLM 只监督被选中的位置", "NSP 并非所有后续模型都保留"],
    code: `from transformers import AutoModelForPreTraining
model=AutoModelForPreTraining.from_pretrained("your-bert-pretraining-checkpoint")
out=model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    token_type_ids=batch.get("token_type_ids"),
    labels=mlm_labels,
    next_sentence_label=nsp_labels,
)
print(out.loss,out.prediction_logits.shape,out.seq_relationship_logits.shape)`,
    output: "得到总 loss、MLM `[B,L,V]` logits 和 NSP `[B,2]` logits。",
    pit: "把 MLM labels 设成遮罩后的输入 ID，而不是遮罩前的真实 ID。",
    quiz: "BERT 预训练为什么不需要人工逐条标注？", answer: "MLM 的原词和 NSP 的句子顺序都能从原始连续文本自动构造监督信号。"
  },
  {
    p: 187, slug: "bert-summary", title: "BERT 总结：MLM/NSP 复盘，以及 GLUE 与 CLUE 公共评测",
    problem: "自己数据上分数很高还不够，怎样用公开基准与标准提交证明模型具备可比较的能力？",
    chain: ["复盘 MLM", "复盘 NSP", "选择公开基准任务", "按官方划分训练/提交", "与统一榜单比较"],
    extraVisuals: [bertOverview, pretrainSequence],
    article: article([
      ["0:00–0:56", "两个预训练任务一句话复盘", "MLM 通过 15% 候选位置与 80/10/10 替换，让 BERT 学单词在上下文中的含义；NSP 用真实/随机句对，让模型学句间关系。两项结合使经典 BERT 同时具备 token 级和句对级理解。"],
      ["0:56–2:51", "GLUE 为什么比自制测试更可信", "GLUE 是由多个公开英文自然语言理解任务组成的基准。各研究者使用相同数据划分和指标，结果才可横向比较。老师强调：在自己的小数据上 100% 不足以证明模型好，公开基准结果更容易被行业认可。"],
      ["2:52–4:36", "中文对应的 CLUE", "英文 GLUE 之外，中文有 CLUE 等公开评测，覆盖分类、阅读理解、自然语言推断等任务。使用时应读取每个子任务的许可证、字段、指标和提交规则；测试标签通常不公开，需要把预测提交到官方平台获得分数。"],
      ["4:36–4:57", "数据来源和简历表达", "公开平台也可作为标准数据来源，但不要只报一个总分；应写清模型版本、子任务、验证/测试集、指标和复现实验设置。正文保留老师鼓励大家挑战榜单的意图，同时提醒公开榜单也可能受数据泄漏和反复调参影响。"],
    ]),
    points: ["GLUE 主攻英文理解任务，CLUE 面向中文", "公开基准让结果可比较", "必须写清子任务、split 和指标"],
    code: `# 示意：统一记录每个公开子任务的结果，而不是只写“效果很好”
results = {
    "task_name": "your-clue-subtask",
    "split": "validation",
    "metric": "accuracy",
    "score": 0.0,
    "checkpoint": "your-checkpoint",
}
print(results)`,
    output: "输出一条可复现实验记录；实际评测应调用对应数据集的官方 metric/提交脚本。",
    pit: "在公开 test 集上反复调参，或把 validation 分数写成官方 test 分数。",
    quiz: "为什么公开基准比自制数据上的单个高分更有说服力？", answer: "数据、划分和指标统一，别人能复现并与其他模型公平比较。"
  },
  {
    p: 188, slug: "elmo-introduction", title: "ELMo：在 Transformer 之前，双向语言模型怎样生成动态词向量",
    problem: "同一个词在不同句子含义不同，怎样不用固定查表，而是让整句上下文决定它的向量？",
    chain: ["字符 CNN 生成基础词表示", "前向语言模型读左文", "后向语言模型读右文", "取各层上下文状态", "任务学习加权组合"],
    extraVisuals: [contextualCompare, elmoUml],
    article: article([
      ["0:00–5:30", "从静态到动态", "Word2Vec/GloVe 为一个词存一个固定向量，无法让 bank 在“river bank”和“bank account”中改变。ELMo 的名字来自 Embeddings from Language Models：词表示由整句经过双向语言模型后动态产生，同一词在不同上下文得到不同向量。"],
      ["5:30–11:30", "字符层与双向 LSTM", "底层字符 CNN 帮助处理词形和未登录词；前向多层 LSTM 根据左侧预测下一个词，后向多层 LSTM 根据右侧预测前一个词。两方向独立训练/表示后合并，因此每个位置同时包含左、右上下文。"],
      ["11:30–17:30", "为什么要混合不同层", "浅层更偏词法/句法，深层更偏语义。ELMo 不只取最后一层，而让具体下游任务学习各层的 softmax 归一化权重，再做加权和并乘缩放系数。这里公式解决的问题是让任务自己决定“更需要哪一层”：`ELMo_k = γ Σ_j s_j h_{k,j}`。"],
      ["17:30–22:00", "与 BERT 的关系", "两者都生成上下文化表示，但 ELMo 主体是字符表示加双向 LSTM，前后方向不是同一层全局自注意力；BERT 用 Transformer Encoder 和 MLM 预训练，可深层联合双向交互。ELMo 仍有历史和概念价值：它清楚展示了“预训练语言模型表示迁移到下游”的转折。"],
    ]),
    points: ["ELMo 是上下文化动态词向量", "主体是字符 CNN + 双向多层 LSTM", "下游任务学习各层加权组合"],
    code: `# 概念演示：不同任务对同一位置的多层表示做加权和
import torch
layers=torch.randn(3,8,1024)  # 3 层 × 8 个 token × 1024 维
weights=torch.softmax(torch.tensor([0.2,0.5,0.3]),dim=0)
elmo=(weights[:,None,None]*layers).sum(0)
print(elmo.shape)`,
    output: "把 3 层表示加权后得到 `[8,1024] = 8 个 token × 每 token 1024 维`。",
    pit: "把 ELMo 说成 Transformer 模型；它的核心上下文编码器是双向 LSTM。",
    quiz: "为什么 ELMo 不只用最后一层？", answer: "不同层包含不同层次的语言信息，下游任务学习加权能按需求组合句法与语义特征。"
  },
  {
    p: 189, slug: "gpt-introduction", title: "GPT：因果注意力怎样把“预测下一个 token”扩展成文本生成",
    problem: "只训练一个下一个 token 预测任务，为什么模型能逐步生成段落并迁移到多种任务？",
    chain: ["Prompt token 序列", "因果 Mask 只看左侧", "预测下一个 token 分布", "选择并追加 token", "循环直到结束"],
    extraVisuals: [gptFlow, modelFamilies],
    article: article([
      ["0:00–3:53", "GPT 只保留 Decoder 的哪些部分", "GPT 是 Generative Pre-trained Transformer，经典 GPT-1 在 2018 年提出，使用 Decoder-only Transformer。因为没有 Encoder，所以原始 Decoder 中读取 Encoder 输出的交叉注意力被移除，只保留 masked self-attention、前馈网络等。老师用图比较 BERT 主理解/NLU、GPT 主生成/NLG。"],
      ["3:53–7:51", "因果遮罩不许偷看未来", "BERT Encoder 自注意力可双向看上下文；GPT 在位置 t 只能看 `<t` 的 token。标准实现通常把未来位置的 attention score 加上负无穷，使 softmax 后概率为 0。课堂口头用 0/1 矩阵相乘帮助直觉，但若直接把 score 乘 0，softmax 后并不会严格为 0，这是必须纠正的技术点。"],
      ["7:51–12:49", "从向量化到 12 层 Decoder", "文本先转 token ID 与 embedding，加位置表示后进入多层 Decoder。GPT-1 常见是 12 层，每层含 masked self-attention 与前馈网络；输出经线性层投到词表，再 softmax/采样选下一个 token，循环到 EOS 或长度上限。课堂把这条流程等价改写为“复述 Transformer Decoder 工作过程”。"],
      ["12:49–15:22", "三模型最终对比", "ELMo 的历史贡献是动态上下文化词向量，但主体是双向 LSTM；BERT 使用 Encoder 深度双向，主理解；GPT 使用 Decoder 因果单向，主生成。老师要求最终掌握三点：看懂三类结构图、会解释 BERT 的 MLM/NSP、能说出 ELMo/BERT/GPT 的优缺点。进一步补充：主理解/主生成只是架构倾向，不表示 BERT 绝不能参与生成或 GPT 不能做理解任务。"],
    ]),
    points: ["GPT 是 Decoder-only 自回归模型", "训练预测下一 token，推理逐 token 循环", "流畅生成不保证事实正确"],
    code: `from transformers import AutoTokenizer, AutoModelForCausalLM
path="your-causal-lm-checkpoint"
tok=AutoTokenizer.from_pretrained(path)
model=AutoModelForCausalLM.from_pretrained(path)
x=tok("今天学习自然语言处理，",return_tensors="pt")
ids=model.generate(**x,max_new_tokens=30,do_sample=True,top_p=0.9)
print(tok.decode(ids[0],skip_special_tokens=True))`,
    output: "在提示词后自回归生成最多 30 个新 token；内容取决于检查点与采样。",
    pit: "把高概率生成当成经过事实核验的答案。",
    quiz: "GPT 训练能并行，为什么生成仍是逐 token？", answer: "训练时整段真实目标已知，可同时监督各位置；生成时下一个 token 依赖刚生成的结果，必须按顺序产生。"
  },
];

generateTopic({
  topic: "BERT、ELMo 与 GPT",
  folder: "foundation-models",
  lessons,
  stages: [
    { title: "一、BERT 从概念到预训练", start: 1, end: 4 },
    { title: "二、上下文化表示与自回归生成", start: 5, end: 6 },
  ],
});
