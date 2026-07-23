import { generateTopic } from "./course_note_helpers.mjs";

const article = (sections) => sections.map(([time, title, body]) =>
  `### ${time}　${title}\n\n${body}`
).join("\n\n");

const modelFamilies = {
  title: "预训练模型三大家族",
  diagram: `
flowchart TB
    T["Transformer 组件"] --> E["Encoder-only：主理解"]
    T --> D["Decoder-only：主生成"]
    T --> ED["Encoder-Decoder：输入到输出"]
    E --> B["BERT 类：分类、抽取、MLM"]
    D --> G["GPT 类：自回归生成"]
    ED --> S["T5/BART 类：翻译、摘要、生成"]
`,
};

const transferFlow = {
  title: "迁移学习从源任务到目标任务",
  diagram: `
flowchart LR
    A["大规模源数据"] --> B["预训练通用参数"]
    B --> C["加载模型与 tokenizer"]
    C --> D{"目标数据规模/算力"}
    D -->|"较少"| E["冻结主体或轻量微调"]
    D -->|"足够"| F["全量微调"]
    E --> G["目标任务评估"]
    F --> G
`,
};

const threeApis = {
  title: "Transformers 三种调用层次",
  diagram: `
flowchart TB
    P["pipeline：任务名 + 模型 + 输入"] --> R["直接得到可读结果"]
    A["AutoTokenizer + AutoModelFor..."] --> T["自己做分词、前向与后处理"]
    C["BertTokenizer + BertFor... 等具体类"] --> L["明确绑定某个架构"]
`,
};

const pipelineSequence = {
  title: "pipeline 内部调用时序",
  diagram: `
sequenceDiagram
    participant U as 用户文本
    participant P as pipeline
    participant T as Tokenizer
    participant M as Task Model
    participant H as 后处理
    U->>P: 字符串或字符串列表
    P->>T: 分词、截断、补齐、转张量
    T-->>M: input_ids / attention_mask
    M-->>H: logits 或 hidden states
    H-->>U: 标签、分数、答案或生成文本
`,
};

const autoUml = {
  title: "Auto 类对象关系",
  diagram: `
classDiagram
    class AutoTokenizer {
      from_pretrained(path)
      __call__(text)
      decode(ids)
    }
    class AutoConfig {
      model_type
      label maps
      hidden_size
    }
    class AutoModelForTask {
      from_pretrained(path)
      forward(input_ids, attention_mask)
    }
    AutoTokenizer --> AutoConfig
    AutoModelForTask --> AutoConfig
`,
};

const classificationFlow = {
  title: "中文分类训练时序",
  diagram: `
sequenceDiagram
    participant D as Dataset/DataLoader
    participant T as Tokenizer
    participant M as SequenceClassifier
    participant L as CrossEntropyLoss
    participant O as Optimizer
    D->>T: 一批 text 与 label
    T-->>M: [B,L] input_ids / attention_mask
    M-->>L: logits [B,C]
    L-->>O: 标量 loss
    O->>O: zero_grad → backward → step
`,
};

const mlmFlow = {
  title: "MLM 数据与标签",
  diagram: `
flowchart LR
    A["原句 token IDs [B,L]"] --> B["选中约一部分可预测位置"]
    B --> C["输入：部分位置换成 MASK/随机词/原词"]
    B --> D["labels：只在选中位置放原 ID"]
    D --> E["未选位置设 -100，不计损失"]
    C --> F["MaskedLM → logits [B,L,V]"]
    E --> G["交叉熵只监督被选位置"]
    F --> G
`,
};

const fixedMaskFlow = {
  title: "课堂固定位置填空流程",
  diagram: `
flowchart LR
    A["完整评论"] --> B["编码/截断到 32 token"]
    B --> C["保存代码下标 16 的原 token ID 为 label"]
    C --> D["把代码下标 16 改成 MASK"]
    D --> E["BERT 输出 [B,32,768]"]
    E --> F["取下标 16 → [B,768]"]
    F --> G["Linear → [B,V]"]
`,
};

const nspFlow = {
  title: "NSP 句对构造与训练",
  diagram: `
flowchart LR
    A["句子 A"] --> C{"句子 B 的来源"}
    C -->|"真实下一句"| P["label=0 / IsNext"]
    C -->|"随机句"| N["label=1 / NotNext"]
    P --> T["[CLS] A [SEP] B [SEP]"]
    N --> T
    T --> M["token_type_ids 区分 A/B"]
    M --> O["NSP logits [B,2]"]
`,
};

const safePipelineCode = (task, model, input) =>
`from transformers import pipeline

pipe = pipeline("${task}", model="${model}")
result = pipe(${input})
print(result)`;

const lessons = [
  {
    p: 155, slug: "transfer-learning-introduction", title: "迁移学习：把别人学会的能力搬到相关任务",
    problem: "为什么不从随机参数训练，而要加载别人已经训练几天、几周甚至更久的模型？",
    chain: ["源任务预训练", "保存参数与词表", "加载到目标任务", "决定冻结或微调", "目标数据评估"],
    extraVisuals: [transferFlow],
    article: article([
      ["0:00–4:51", "大白话定义与几种迁移对象", "老师把迁移学习概括为“拿别人训练好的模型来干活”。更严谨地说，是把源任务学到的表示、参数或知识用于相关目标任务。迁移对象可以是词向量和特征抽取器，也可以是完整网络主体；例如文本生成和机器翻译都需要条件生成能力，底层表示可复用。"],
      ["4:51–10:48", "模型大小、部署和算力", "课堂用本地几个 GB 的模型与更大的参数模型对比，说明参数越多，磁盘、内存/显存、加载时间和推理时延通常越高。能在服务器训练不代表能在普通电脑或边缘设备部署，因此压缩、量化、蒸馏也是迁移后的工程问题。网络传言和夸张收益案例只作课堂调节，不应当作技术证据。"],
      ["10:48–16:42", "四种常见使用方式", "老师依次解释：冻结预训练主体，只训练新任务头；解冻全部参数进行微调；先冻结再逐层解冻；针对相似目标任务继续预训练后再微调。初学者可把它理解成“保留通用知识的程度”从高到低。数据少、算力小先尝试冻结或轻量微调；数据充分且任务差异大再考虑全量微调。"],
      ["16:42–20:42", "收益、风险与选择原则", "迁移学习能减少目标标注数据和训练时间，通常比随机初始化更稳；但源领域与目标领域差别太大可能负迁移。还要保证 tokenizer、词表和模型权重属于同一检查点。老师最后预告三类模型与 Transformers 三种调用方式，后续会用同样六个任务反复比较接口层次。"],
    ]),
    points: ["迁移的是参数、表示或知识，不只是复制文件", "源任务和目标任务越相关，迁移通常越有效", "tokenizer 与模型必须配套"],
    code: `from transformers import AutoTokenizer, AutoModel
name_or_path = "your-local-or-hub-checkpoint"
tokenizer = AutoTokenizer.from_pretrained(name_or_path)
model = AutoModel.from_pretrained(name_or_path)
print(model.config.model_type)`,
    output: "加载同一检查点的 tokenizer 和模型主体，并打印架构类型。",
    pit: "模型用 A 检查点，tokenizer 却来自 B；token ID 语义可能完全错位。",
    quiz: "什么是负迁移？", answer: "源任务学到的偏好与目标任务不匹配，迁移后反而比合适初始化或从头训练更差。"
  },
  {
    p: 156, slug: "pretrained-model-families", title: "常见预训练 NLP 模型：Encoder、Decoder 与 Encoder-Decoder",
    problem: "面对 BERT、GPT、T5、BART 等名字，怎样先按结构和训练目标归类，而不是死背模型表？",
    chain: ["看 Transformer 组件", "判断预训练目标", "判断理解或生成", "匹配下游任务", "再选具体检查点"],
    extraVisuals: [modelFamilies, transferFlow],
    article: article([
      ["0:00–5:57", "三类结构先搭骨架", "老师把模型分成三类：自回归 AR 多用 Decoder-only，依据左侧上下文预测下一个 token，代表 GPT；自编码 AE 多用 Encoder-only，通过双向上下文学习理解表示，代表 BERT；Encoder-Decoder 同时使用编码器和解码器，把输入序列变为输出序列，代表 T5、BART。课堂说“AR 只能单向”是指其因果注意力训练方式，不表示模型永远不能看到任何前置信息。"],
      ["5:57–16:56", "代表模型与预训练任务", "BERT 类常用 Masked Language Modeling（MLM），适合分类、NER、抽取式问答等理解任务；GPT 类用 next-token prediction，适合续写和生成；T5/BART 类把任务统一成文本到文本或去噪生成，适合翻译、摘要。老师逐个展示若干变体，如 RoBERTa、ALBERT、DistilBERT、XLNet 等；小白不必一次背全，应抓住“结构—目标—任务”三连线。"],
      ["16:56–25:50", "模型名字中的线索", "常见名称会包含语言、大小、是否区分大小写、训练语料或任务，例如 `base/large`、`cased/uncased`、`chinese`、`finetuned-...`。这些不是装饰：hidden size、层数和参数规模决定资源，语言与 tokenizer 决定输入覆盖，finetuned 后缀说明任务头是否已经训练。"],
      ["25:50–32:58", "如何选择而不是追最热门", "老师复盘模型家族和用途。实际选择要看任务类型、语言、许可证、最大长度、模型卡评测、资源预算和部署目标。课程后面六个任务会用适配检查点演示；不能拿一个仅预训练、没有目标任务头的 Base 模型，期待 pipeline 直接给出可靠业务标签。"],
    ]),
    points: ["Encoder-only 主理解，Decoder-only 主生成", "Encoder-Decoder 擅长有输入条件的生成", "检查点是否为目标任务微调过非常关键"],
    code: `from transformers import AutoConfig
config = AutoConfig.from_pretrained("your-checkpoint")
print("type:", config.model_type)
print("encoder-decoder:", getattr(config, "is_encoder_decoder", False))`,
    output: "查看模型类型以及是否为 Encoder-Decoder 架构。",
    pit: "仅凭模型名字“大”就认定更适合；资源、语言和任务头不匹配时，大模型也可能不可用。",
    quiz: "做抽取式中文问答时，为什么通常先找 BERT 类问答检查点？", answer: "任务需要双向理解上下文并预测答案起止位置，Encoder-only 加问答头与目标直接匹配。"
  },
  {
    p: 157, slug: "transformers-library-setup", title: "Transformers 库与环境：模型仓库、缓存和三种调用方式",
    problem: "Transformers、模型仓库和预训练权重分别是什么，离线环境又该怎样准备？",
    chain: ["选择模型仓库", "安装 transformers/datasets", "下载或准备本地检查点", "验证 tokenizer/model", "选择调用层次"],
    extraVisuals: [threeApis, pipelineSequence, autoUml],
    article: article([
      ["0:00–1:47", "先复盘模型三分法", "老师用题目复盘：AR/Decoder-only 主生成，AE/Encoder-only 主理解，Seq2Seq 混合编码和解码。随后转入“怎样把模型真正加载进代码”。"],
      ["1:48–5:32", "模型仓库与 Transformers 的关系", "Hugging Face Hub 用来托管模型、tokenizer、配置和模型卡；`transformers` Python 库负责下载/读取这些文件并提供推理与训练类。音轨多次把 Hugging Face 识别成了其他词，正文统一纠正。仓库里也可能有不安全、不适配或许可证受限的第三方模型，所以需要阅读模型卡。"],
      ["5:32–10:09", "网络受限与本地模型", "课堂展示访问海外仓库不稳定，并介绍国内镜像/社区与事先下好的本地目录。无论在线还是离线，本质上都需要一组完整文件：配置、tokenizer 资源和权重。路径要指向检查点目录，不是随便一个 `.bin` 文件；离线部署还要记录库版本和校验文件是否齐全。"],
      ["10:09–15:51", "三种接口层次", "第一层 `pipeline` 封装分词、模型、后处理，适合快速验证；第二层 Auto 类自动读取 config 选择具体 tokenizer 和任务模型，适合项目代码；第三层直接使用 `BertTokenizer`、`BertFor...` 等具体类，绑定架构更明确。三种方式底层不是三套不同模型，而是抽象程度不同。课程先做 pipeline 的六个任务，再用 Auto 重写六个，最后用具体类演示一个。"],
    ]),
    points: ["Hub 是文件仓库，Transformers 是加载与运行库", "本地检查点也必须含配置、词表和权重", "pipeline、Auto、具体类只是抽象层级不同"],
    code: `from transformers import AutoTokenizer, AutoModel
path = "your-checkpoint"
tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModel.from_pretrained(path, local_files_only=True)
print(tok.__class__.__name__, model.__class__.__name__)`,
    output: "在离线模式从本地目录加载，并打印自动选择的具体类。",
    pit: "课程旧环境的安装版本和 API 直接照抄到新环境；应锁定项目版本并先做最小加载测试。",
    quiz: "pipeline 和 AutoModel 的核心差别是什么？", answer: "pipeline 还封装了任务后处理并直接返回可读结果；AutoModel 通常返回张量，需要自己处理。"
  },
  {
    p: 158, slug: "pipeline-text-classification", title: "Pipeline 文本分类：三行代码背后的标签与概率",
    problem: "怎样快速验证一个情感/文本分类检查点，同时避免误读 LABEL_0、星级和分数？",
    chain: ["选择 text-classification", "加载任务检查点", "输入文本", "得到 label/score", "核对 id2label"],
    extraVisuals: [pipelineSequence, threeApis],
    article: article([
      ["0:00–3:49", "环境与 pipeline 参数", "老师先安装 Transformers 和 datasets，再解释 `pipeline(task, model=...)`。第一个参数是任务名，第二个是检查点路径或仓库名。课堂使用本地模型避免重复下载；只写任务不写模型时库可能选默认模型，但这会带来版本和语言不确定性，学习项目最好显式指定。"],
      ["3:49–8:35", "如何在模型仓库筛选", "先按 Text Classification 任务筛选，再按 Chinese、情感或领域搜索，阅读模型卡、标签说明和示例。网页在线试玩可快速排除不符合预期的模型，但最终仍要在自己的验证集测试。老师强调不要下载很久后才发现任务或语言不匹配。"],
      ["8:37–16:34", "运行与读结果", "把中文文本交给 classifier，返回形如 `{'label': '5 stars', 'score': 0.91}` 的结果。`score` 是模型在当前标签空间内的置信分数，不是“模型整体准确率”；`label` 的业务含义由 config 的 `id2label` 决定。若返回 `LABEL_0`，必须查模型卡或映射，不能自行猜正负。"],
      ["16:36–22:36", "批量输入、设备与警告", "pipeline 可接收字符串列表做批量推理，并可指定 CPU/GPU 设备。课堂展示模型加载与运行日志；很多 warning 不是报错，例如当前使用 CPU 或某些权重未使用，但必须读清具体内容。验证正确后再进入特征提取。"],
    ]),
    points: ["显式指定任务与检查点", "score 是单次预测置信，不是测试集准确率", "LABEL_0 的含义必须查 id2label"],
    code: safePipelineCode("text-classification", "your-classification-checkpoint", `"这家店服务很好，菜也很新鲜。"`) ,
    output: "得到标签和分数的字典列表，具体标签名取决于检查点配置。",
    pit: "把 `score=0.95` 说成模型准确率 95%；整体准确率必须在有真值的评估集上计算。",
    quiz: "模型返回 LABEL_1 时能直接解释为正面吗？", answer: "不能。要查看模型 config 的 id2label 或模型卡，LABEL_1 在不同模型中可能含义不同。"
  },
  {
    p: 159, slug: "pipeline-feature-extraction", title: "Pipeline 特征提取：没有任务头的“半成品”怎样读形状",
    problem: "特征提取为什么不直接返回类别，而会得到一大块三维浮点数？",
    chain: ["输入文本", "tokenizer 切成 L 个 token", "Encoder 输出每词 H 维", "得到 [B,L,H]", "池化或接自定义任务头"],
    extraVisuals: [pipelineSequence, autoUml],
    article: article([
      ["0:00–4:54", "带任务头与不带任务头", "老师把特征提取比作买手机零件：模型只给通用特征，还需要自己组装分类、聚类或检索模块；文本分类像直接买成品手机，任务头已把特征变成标签。这个类比对应 `AutoModel` 与 `AutoModelForSequenceClassification` 的差别。"],
      ["4:54–10:49", "创建 feature-extraction pipeline", "任务名是 `feature-extraction`，检查点应是可输出隐藏状态的基础模型。输入一句话后得到嵌套列表。设批量 B=1、分词后 L=12、隐藏维 H=768，则形状 `[1,12,768] = 1 条文本 × 12 个 token × 每个 token 768 个特征`。它不是“12 个词”这么绝对，因为 tokenizer 还会加入特殊 token，并可能把一个词拆成多个子词。"],
      ["10:49–15:46", "怎样变成句向量", "若下游需要一句一个向量，可取 `[CLS]` 位置、对非 padding token 做 masked mean pooling，或使用专门训练的句向量模型。直接把三维数组扔给传统分类器通常形状不匹配。老师强调特征提取是预训练范畴的中间结果，后面 AutoModel 版本会手工查看 `last_hidden_state`。"],
    ]),
    points: ["特征提取输出的是 token 级通用表示", "`[B,L,H]` 的 L 含特殊 token/子词", "句向量需要明确池化规则"],
    code: safePipelineCode("feature-extraction", "your-base-checkpoint", `"迁移学习可以复用知识"`) ,
    output: "返回三维嵌套列表，常见语义是 `[批量, token 数, 隐藏维度]`。",
    pit: "把第一个 token 当句向量却不知道模型是否专门训练过该用法。",
    quiz: "`[2,16,768]` 分别表示什么？", answer: "2 条文本；每条补齐/截断到 16 个 token；每个 token 有 768 维隐藏特征。"
  },
  {
    p: 160, slug: "pipeline-fill-mask", title: "Pipeline 完形填空：`[MASK]`、候选概率与多个空位",
    problem: "怎样让 BERT 根据左右文预测被遮住的 token，并正确理解它一次预测的是 token 而非汉字？",
    chain: ["选择 fill-mask", "插入 tokenizer.mask_token", "模型输出 [L,V]", "取遮罩位 top-k", "还原候选句"],
    extraVisuals: [pipelineSequence, mlmFlow],
    article: article([
      ["0:00–2:54", "任务与模型匹配", "完形填空对应 Masked Language Modeling，pipeline 任务名 `fill-mask`。必须选择带 MLM 头、且 tokenizer 语言匹配的检查点。老师展示在模型仓库按任务筛选；使用不同社区的库时，导入和接口也会不同，不能把代码混用。"],
      ["2:54–5:55", "MASK 是一个 token 槽位", "输入要使用该 tokenizer 的 `mask_token`，BERT 常见为 `[MASK]`。一次遮罩位置预测的是一个词表 token，不等于固定一个汉字：中文词表常一字一 token，但也可能有整词或子词。老师说 pipeline 示例一次只处理一个 MASK；多个 MASK 若独立一次性预测会缺少彼此条件，可循环填入上一步候选，或使用支持多空联合策略的代码。"],
      ["5:55–11:57", "返回值和设备提示", "结果通常包含 `score`、`token`、`token_str` 和 `sequence`，默认给若干 top-k 候选。分数是该遮罩位置的词表概率。课堂日志提示模型在 CPU 运行，这不是错误；若要 GPU，需要确保框架识别设备并正确指定 pipeline device。最后要检查候选句是否语义、语法都合理，而不只看最高分。"],
    ]),
    points: ["输入应使用 tokenizer.mask_token", "一个 MASK 预测一个 token，不必然是一个字", "多个 MASK 需要明确顺序或联合策略"],
    code: `from transformers import pipeline
fill = pipeline("fill-mask", model="your-chinese-mlm-checkpoint")
mask = fill.tokenizer.mask_token
print(fill(f"我明天去{mask}家吃饭", top_k=5))`,
    output: "返回遮罩位置的 5 个候选 token、分数与替换后的完整序列。",
    pit: "手写 `<mask>` 或 `[MASK]` 却与 tokenizer 的真实遮罩符不一致。",
    quiz: "为什么多个 MASK 逐个填时，顺序会影响结果？", answer: "后一个位置的预测会把前面已经填入的 token 当作上下文，不同先后顺序形成不同条件。"
  },
  {
    p: 161, slug: "pipeline-question-answering", title: "Pipeline 阅读理解：从 context 中预测答案起止位置",
    problem: "抽取式问答怎样从一段材料中找答案，而不是凭模型记忆自由发挥？",
    chain: ["准备 context", "提出 question", "拼成句对并编码", "预测 start/end", "截取 answer"],
    extraVisuals: [pipelineSequence],
    article: article([
      ["0:00–2:55", "抽取式问答的边界", "老师称阅读理解为抽取式问答：输入一段 `context` 和一个 `question`，模型输出材料中的答案片段。它是带任务头的成品任务，与只输出隐藏特征不同。抽取式意味着答案应来自 context；若材料没有答案，普通模型仍可能硬选一段，需要专门的无答案训练或阈值策略。"],
      ["2:55–6:53", "创建 question-answering pipeline", "任务名 `question-answering`，模型应是针对问答微调的检查点。课堂用本地中文 MRC 模型，准备“我叫……、职业……、喜好……”的 context，再提出名字、职业、爱好问题。调用时使用关键字参数 `question=`、`context=`，所以二者书写顺序不重要。"],
      ["6:53–12:35", "输出与同义表达", "返回包含 `answer`、`score`、`start`、`end`。start/end 是在原 context 字符串中的边界，可用切片核验。老师故意让材料写“喜好”、问题写“爱好”，观察模型能否理解近义表达。真正评估要用 Exact Match/F1 和成套问答数据，不能只凭三道自编题。"],
    ]),
    points: ["答案通常从 context 原文截取", "关键字参数避免 question/context 位置写反", "无答案问题需要专门处理"],
    code: `from transformers import pipeline
qa = pipeline("question-answering", model="your-chinese-qa-checkpoint")
context = "小林是一名教师，他喜欢编程和徒步。"
print(qa(question="小林做什么工作？", context=context))`,
    output: "返回答案文本、置信分数以及答案在 context 中的起止位置。",
    pit: "使用普通 BERT base 而不是问答微调模型，却期待 start/end 头已经学会找答案。",
    quiz: "为什么抽取式问答不适合回答 context 完全没提到的问题？", answer: "它的输出空间是 context 的起止位置；没有答案时也可能被迫选一个片段。"
  },
  {
    p: 162, slug: "pipeline-summarization", title: "Pipeline 文本摘要：生成长度、截断与事实一致性",
    problem: "怎样把长文本压缩为摘要，并避免输入被截断或模型编造原文没有的事实？",
    chain: ["选择 summarization", "tokenizer 编码长文", "Encoder 理解输入", "Decoder 生成摘要", "检查长度与事实"],
    extraVisuals: [pipelineSequence, modelFamilies],
    article: article([
      ["0:00–3:10", "摘要是生成任务", "文本摘要不是从固定标签中选择，而是生成新的 token 序列，常使用 Encoder-Decoder 检查点。pipeline 任务名 `summarization`。模型必须与输入语言、领域匹配；英文摘要模型直接处理中文通常不可靠。"],
      ["3:10–6:35", "长度参数怎么理解", "`max_length`/`min_length` 在许多版本中指生成序列的 token 长度；较新的生成接口也常用 `max_new_tokens` 指新增 token 数。输入本身还有模型最大上下文限制，过长会被截断。摘要长文应先分块或使用长上下文模型，不能无声丢掉后半篇。"],
      ["6:35–9:36", "输出与质量检查", "结果通常是 `summary_text`。老师用 pipeline 展示直接生成；学习时还要补一层人工核对：摘要是否覆盖核心事实、是否把人物数字改错、是否出现原文没有的结论。ROUGE 可比较词面重合，但不能完全判断事实一致性。"],
    ]),
    points: ["摘要属于条件生成任务", "输入长度和输出长度是两套限制", "流畅不等于事实正确"],
    code: safePipelineCode("summarization", "your-chinese-summarization-checkpoint", `"这里放入一段较长的中文文章。"`) ,
    output: "返回带 `summary_text` 的列表；具体键和长度受版本/模型影响。",
    pit: "输入超过最大长度后被静默截断，却以为摘要覆盖了整篇文章。",
    quiz: "`max_new_tokens` 与模型最大输入长度有什么区别？", answer: "前者限制新生成多少 token；后者限制模型能读入多少 token，二者互不替代。"
  },
  {
    p: 163, slug: "pipeline-ner", title: "Pipeline NER：token 标签怎样合并成人名、地点和组织",
    problem: "命名实体识别为什么会返回许多子词片段，怎样把它们合并成完整实体？",
    chain: ["选择 token-classification", "逐 token 打标签", "识别 BIO 边界", "聚合子词", "输出实体与位置"],
    extraVisuals: [pipelineSequence],
    article: article([
      ["0:00–3:50", "NER 是 token 级分类", "命名实体识别为句中每个 token 判断人名 PER、地点 LOC、组织 ORG 等。它与整句文本分类不同：输出长度随 token 数变化。模型应带 TokenClassification/NER 任务头，并使用训练时同一标签集合。"],
      ["3:50–8:20", "BIO 与子词", "标签常以 B- 开始一个实体、I- 延续实体、O 表示非实体。WordPiece/BPE 可能把一个名字拆成多个子词，因此 pipeline 原始输出可能一段实体出现多条。设置合适的 `aggregation_strategy` 可依据标签与位置合并，但中文字符级模型的聚合效果仍要抽样检查。"],
      ["8:20–12:29", "读懂返回位置", "结果常含 `entity_group`、`score`、`word`、`start`、`end`。start/end 可回原字符串切片验证。老师演示指定本地 NER 模型并打印结果；工程中还要处理空格、特殊字符和重叠实体，并用实体级 Precision/Recall/F1，而不是 token 准确率。"],
    ]),
    points: ["NER 输出是 token/实体级，不是整句单标签", "BIO 标签表达实体边界", "子词需要聚合回原文实体"],
    code: `from transformers import pipeline
ner = pipeline(
    "token-classification",
    model="your-chinese-ner-checkpoint",
    aggregation_strategy="simple",
)
print(ner("小林今天从武汉来到北京。"))`,
    output: "返回识别到的人名、地点等实体，以及分数和原文位置。",
    pit: "只看 token accuracy；大量 O 标签会让它很高，却掩盖实体识别很差。",
    quiz: "B-LOC 与 I-LOC 有什么差别？", answer: "B-LOC 表示地点实体开头，I-LOC 表示同一地点实体的后续 token。"
  },
  {
    p: 164, slug: "auto-text-classification", title: "Auto 模型文本分类：手工完成分词、前向与 argmax",
    problem: "不用 pipeline 后，怎样一步步把字符串变成 logits、概率和业务标签？",
    chain: ["AutoTokenizer 编码", "得到 [B,L] 张量", "任务模型前向", "logits [B,C]", "softmax/argmax 映射标签"],
    extraVisuals: [autoUml, classificationFlow],
    article: article([
      ["0:00–7:00", "为什么改用 Auto 类", "pipeline 方便但隐藏步骤；Auto 接口让你控制截断、补齐、批量、设备和后处理。加载时使用同一个路径创建 `AutoTokenizer` 与 `AutoModelForSequenceClassification`，后者会根据 config 自动选择 BERT、RoBERTa 等具体实现。"],
      ["7:00–19:00", "张量形状与前向", "tokenizer 用 `return_tensors='pt'` 返回字典。若 B=2、L=32，`input_ids [2,32] = 2 条文本 × 每条 32 个 token ID`，`attention_mask [2,32]` 中 1 表示有效位置、0 表示 padding。模型输出 `logits [2,C] = 每条文本对 C 个类别的未归一化分数`。推理需 `model.eval()` 与 `torch.no_grad()`。"],
      ["19:00–35:40", "概率、标签和 config", "对 logits 最后一维做 softmax 得到每类概率，argmax 得到类别 ID，再用 `model.config.id2label` 映射业务标签。老师逐步对比 pipeline 自动做掉的工作。若是多标签模型，不应 softmax/argmax，而应 sigmoid 后按阈值保留多个标签。"],
    ]),
    points: ["AutoTokenizer 与任务模型必须同检查点", "logits 不是概率", "单标签 softmax，多标签 sigmoid"],
    code: `import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
path="your-classification-checkpoint"
tok=AutoTokenizer.from_pretrained(path)
model=AutoModelForSequenceClassification.from_pretrained(path).eval()
batch=tok(["文本很好"], padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    logits=model(**batch).logits
idx=logits.argmax(-1).item()
print(model.config.id2label[idx])`,
    output: "从 `[1,L]` 输入得到 `[1,C]` logits，并输出最高分类标签。",
    pit: "模型和输入不在同一 device，或推理时忘记 eval/no_grad 造成随机 dropout 与额外显存。",
    quiz: "`logits [4,3]` 表示什么？", answer: "一批 4 条文本，每条对 3 个类别各有一个未归一化分数。"
  },
  {
    p: 165, slug: "auto-feature-extraction", title: "Auto 模型特征提取：last_hidden_state 与 masked mean pooling",
    problem: "怎样从 AutoModel 的 token 表示得到一个忽略 padding 的句向量？",
    chain: ["tokenizer 批量编码", "AutoModel 前向", "last_hidden_state [B,L,H]", "用 mask 加权求和", "除以有效 token 数"],
    extraVisuals: [autoUml],
    article: article([
      ["0:00–5:30", "基础模型没有任务头", "`AutoModel` 加载主体 Encoder，输出通用隐藏状态，不直接给分类标签。`last_hidden_state [B,L,H]`：B 是批量，L 是补齐后的 token 长度，H 是隐藏维度。"],
      ["5:30–11:30", "为什么不能普通 mean", "批量中短句会被 PAD 补到最长；若直接沿 L 平均，padding 位置也参与。把 `attention_mask [B,L]` 扩展为 `[B,L,1]`，与隐藏状态相乘，只保留有效 token，再除以每句有效 token 数。"],
      ["11:30–17:00", "特征用途与限制", "得到 `[B,H]` 句向量后可用于简单分类、聚类或相似度。但普通 MLM 预训练模型的池化未必适合语义相似度，正式检索应优先专门训练的 sentence embedding 模型，并按其模型卡执行 pooling/normalize。"],
    ]),
    points: ["last_hidden_state 是 token 级表示", "masked mean 防止 padding 污染句向量", "通用 BERT 向量不天然等于优质句向量"],
    code: `import torch
from transformers import AutoTokenizer, AutoModel
path="your-base-checkpoint"
tok=AutoTokenizer.from_pretrained(path)
model=AutoModel.from_pretrained(path).eval()
batch=tok(["短句","这是另一条更长的句子"],padding=True,return_tensors="pt")
with torch.no_grad(): h=model(**batch).last_hidden_state
mask=batch["attention_mask"].unsqueeze(-1)
sent=(h*mask).sum(1)/mask.sum(1).clamp_min(1)
print(h.shape,sent.shape)`,
    output: "例如先得到 `[2,L,H]`，再池化为 `[2,H]`。",
    pit: "把 padding token 一起平均，导致不同批次长度改变同一句的向量。",
    quiz: "attention_mask 为什么要 unsqueeze(-1)？", answer: "把 `[B,L]` 变成 `[B,L,1]`，才能沿隐藏维 H 广播并与 `[B,L,H]` 相乘。"
  },
  {
    p: 166, slug: "auto-fill-mask", title: "Auto 模型完形填空：定位 mask，再从词表 logits 取 top-k",
    problem: "不用 pipeline 时，怎样只读取 `[MASK]` 位置的 V 维词表分数？",
    chain: ["编码带 MASK 文本", "找 mask_token_id 位置", "MaskedLM 前向 [B,L,V]", "索引出 [V]", "top-k 解码"],
    extraVisuals: [autoUml, mlmFlow],
    article: article([
      ["0:00–5:00", "任务模型与输入", "使用 `AutoModelForMaskedLM`，因为它在每个 token 隐藏状态后接了词表分类头。输入字符串中的遮罩符应来自 `tokenizer.mask_token`。"],
      ["5:00–10:30", "形状一步步缩小", "模型 logits 形状 `[B,L,V] = B 条文本 × L 个位置 × V 个词表候选`。通过 `input_ids == mask_token_id` 找到遮罩坐标，索引后得到该位置 `[V]` 分数，再 `topk(k)` 得到候选 ID。不要对所有 L 个位置都 top-k 后误当成空位预测。"],
      ["10:30–16:00", "解码与多空位", "用 `convert_ids_to_tokens` 看原 token，用 `decode` 看可读文本。若有多个 MASK，索引结果会是 `[M,V]`；独立 top-k 并不保证联合最优，循环填充、beam search 或专门生成模型更合适。"],
    ]),
    points: ["MaskedLM 输出每个位置对整个词表的 logits", "先定位 MASK，再取该位置", "top-k ID 要用同一 tokenizer 解码"],
    code: `import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
path="your-mlm-checkpoint"
tok=AutoTokenizer.from_pretrained(path)
model=AutoModelForMaskedLM.from_pretrained(path).eval()
text=f"我喜欢学习{tok.mask_token}语言"
x=tok(text,return_tensors="pt")
with torch.no_grad(): logits=model(**x).logits
pos=(x["input_ids"][0]==tok.mask_token_id).nonzero().item()
ids=logits[0,pos].topk(5).indices
print(tok.convert_ids_to_tokens(ids))`,
    output: "从 `[1,L,V]` 中抽出遮罩位置 `[V]`，打印分数最高的 5 个 token。",
    pit: "用 `argmax()` 不指定维度，对整个三维 logits 取出一个扁平索引。",
    quiz: "`[2,20,30000]` 的最后一维是什么？", answer: "词表中 30000 个 token 在每个位置的候选分数。"
  },
  {
    p: 167, slug: "auto-question-answering", title: "Auto 模型阅读理解：start_logits 与 end_logits 组合答案",
    problem: "问答模型为什么输出两条长度为 L 的分数，怎样避免答案终点跑到起点前面？",
    chain: ["question/context 成对编码", "前向得到两个 [B,L]", "约束 context 范围", "搜索 start≤end", "token/字符边界还原"],
    extraVisuals: [autoUml],
    article: article([
      ["0:00–5:30", "成对编码", "`AutoModelForQuestionAnswering` 输出开始和结束位置。tokenizer 接收 question 与 context，并加入特殊 token。需要 `truncation='only_second'` 等策略优先截断 context，同时保留问题。"],
      ["5:30–12:00", "两个 logits 的形状", "`start_logits [B,L]` 给每个 token 作为答案起点的分数，`end_logits [B,L]` 给终点分数。简单分别 argmax 可能得到 end < start，或落到问题/特殊 token；更稳妥是在 context token 范围内枚举满足 `start≤end`、长度不超上限的组合，最大化 start+end 分数。"],
      ["12:00–18:00", "从 token 回到原文", "用 `offset_mapping` 将 token 位置映射回 context 字符边界，可精确保留原文；直接 decode token 片段可能改变空格或子词符号。长文超过窗口时需要滑动窗口与 `overflow_to_sample_mapping`，再跨窗口比较候选。"],
    ]),
    points: ["问答头输出 start/end 两组 token 分数", "答案必须约束在 context 且 start≤end", "offset_mapping 比直接 decode 更适合回原文"],
    code: `import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
path="your-qa-checkpoint"
tok=AutoTokenizer.from_pretrained(path)
model=AutoModelForQuestionAnswering.from_pretrained(path).eval()
x=tok("他做什么工作？","小林是一名教师。",return_tensors="pt")
with torch.no_grad(): out=model(**x)
s=out.start_logits.argmax(-1).item(); e=out.end_logits.argmax(-1).item()
print(tok.decode(x["input_ids"][0,s:e+1]))`,
    output: "教学版直接取最高起止位置并解码；正式项目还需加入合法区间约束。",
    pit: "独立 argmax 后不检查 end 是否早于 start。",
    quiz: "为什么输出不是 `[B,C]`？", answer: "答案类别不是固定 C 类，而是输入序列中 L 个可能起点和 L 个可能终点。"
  },
  {
    p: 168, slug: "auto-summarization", title: "Auto 模型文本摘要：tokenize、generate 与 decode 分工",
    problem: "不用 pipeline 时，摘要模型的前向训练输出和 `generate` 推理有什么不同？",
    chain: ["AutoTokenizer 编码", "Seq2Seq 模型读取输入", "generate 自回归搜索", "输出 token IDs", "batch_decode 成摘要"],
    extraVisuals: [autoUml, modelFamilies],
    article: article([
      ["0:00–5:00", "加载 Seq2Seq 任务模型", "摘要应使用 `AutoModelForSeq2SeqLM` 等带生成头的 Encoder-Decoder 模型。tokenizer 编码长文得到 `[B,Lsrc]`，Encoder 生成上下文表示。"],
      ["5:00–11:00", "为什么用 generate", "直接 `model(**inputs)` 只做一次前向并返回 logits；开放式推理需要上一时刻输出作为下一时刻输入，因此用 `model.generate` 循环生成、处理 EOS、beam search、长度惩罚等。输出 IDs 形状 `[B,Lout]`。"],
      ["11:00–17:00", "解码与参数", "`batch_decode(ids, skip_special_tokens=True)` 还原摘要。`num_beams`、`max_new_tokens`、`length_penalty` 等影响速度和长度，不能只追求 beam 越多。还要检查源文本截断和生成事实错误。"],
    ]),
    points: ["摘要推理应调用 generate", "generate 输出 token IDs，不是字符串", "生成参数需在验证集上选"],
    code: `from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
path="your-summarization-checkpoint"
tok=AutoTokenizer.from_pretrained(path)
model=AutoModelForSeq2SeqLM.from_pretrained(path)
x=tok("这里放入一段较长的文章。",return_tensors="pt",truncation=True)
ids=model.generate(**x,max_new_tokens=64,num_beams=4)
print(tok.batch_decode(ids,skip_special_tokens=True))`,
    output: "得到 `[B,Lout]` 生成 ID，并解码成摘要字符串列表。",
    pit: "把训练时的 logits.argmax 当作完整生成；它没有正确执行逐步解码。",
    quiz: "为什么 `generate` 比单次 forward 慢？", answer: "它要逐 token 多次调用解码器，并可能同时维护多个 beam 候选。"
  },
  {
    p: 169, slug: "auto-ner", title: "Auto 模型 NER：subword 标签对齐与实体聚合",
    problem: "自己处理 NER logits 时，怎样把 `[B,L,C]` 的子词标签重新对应到原文？",
    chain: ["tokenizer 编码与 offsets", "TokenClassification 前向", "logits [B,L,C]", "argmax→id2label", "按 BIO 与 offsets 合并"],
    extraVisuals: [autoUml],
    article: article([
      ["0:00–5:00", "任务模型输出", "`AutoModelForTokenClassification` 为每个输入 token 输出 C 类 logits，所以形状 `[B,L,C] = B 条句子 × L 个 token × C 个标签`。argmax 只沿 C 维做。"],
      ["5:00–11:00", "排除特殊与 padding", "`[CLS]`、`[SEP]`、`[PAD]` 不对应原文字实体，需结合 attention mask、special tokens mask 或 offset `(0,0)` 排除。标签 ID 用 `config.id2label` 转回 BIO 标签。"],
      ["11:00–17:00", "合并实体", "同一汉字串可能对应多个 token；按 offset 拼原文、按 B/I 规则判断新实体与延续。若 I-ORG 没有合法前导 B-ORG，应制定修复策略并记录。训练时标签也必须通过 `word_ids()` 对齐到子词，非首子词可复制 I 标签或设 -100，取决于方案。"],
    ]),
    points: ["argmax 应在标签维 C 上", "特殊 token 不参与实体", "offset 与 BIO 共同完成原文聚合"],
    code: `import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
path="your-ner-checkpoint"
tok=AutoTokenizer.from_pretrained(path)
model=AutoModelForTokenClassification.from_pretrained(path).eval()
x=tok("小林去了北京",return_tensors="pt",return_offsets_mapping=True)
offsets=x.pop("offset_mapping")[0]
with torch.no_grad(): logits=model(**x).logits
ids=logits.argmax(-1)[0]
print([(tuple(o.tolist()),model.config.id2label[i.item()]) for o,i in zip(offsets,ids)])`,
    output: "打印每个 token 的原文边界和预测 BIO 标签，供后续合并。",
    pit: "把 `[CLS]` 的高分标签也当成实体，或按 token 字符串拼接导致子词符号残留。",
    quiz: "`logits [2,30,9]` 表示什么？", answer: "2 条文本，每条 30 个 token 位置，每个位置在 9 个 NER 标签上有分数。"
  },
  {
    p: 170, slug: "specific-model-fill-mask", title: "具体模型类做完形填空：显式使用 BertTokenizer 与 BertForMaskedLM",
    problem: "具体类与 Auto 类代码几乎相同时，为什么还要知道它们的区别？",
    chain: ["确认检查点确为 BERT", "加载 BertTokenizer", "加载 BertForMaskedLM", "定位 MASK", "top-k 解码"],
    extraVisuals: [threeApis, autoUml, mlmFlow],
    article: article([
      ["0:00–4:30", "Auto 与具体类的边界", "Auto 类读取 config 后决定具体实现；`BertTokenizer` 与 `BertForMaskedLM` 则直接声明“我确定这是 BERT”。优点是类型清楚、便于访问 BERT 专属组件；缺点是换成 RoBERTa/T5 检查点时不能无改动复用。"],
      ["4:30–9:30", "代码步骤没有魔法", "仍然是 tokenizer 编码、找到 mask_token_id、模型前向得到 `[B,L,V]`、取遮罩位置 top-k、解码。老师只演示这一种具体类写法，因为其余任务与 Auto 版本结构类似。"],
      ["9:30–14:30", "何时选择哪层", "快速试玩用 pipeline；需要通用项目和多架构切换用 Auto；研究某个架构内部、静态类型或自定义模块时用具体类。三者可加载同一权重并得到等价结果，区别主要在封装和控制范围。"],
    ]),
    points: ["具体类显式绑定架构", "检查点架构不匹配会加载失败或产生大量未匹配权重", "三层 API 可按控制需求选择"],
    code: `import torch
from transformers import BertTokenizer, BertForMaskedLM
path="your-bert-mlm-checkpoint"
tok=BertTokenizer.from_pretrained(path)
model=BertForMaskedLM.from_pretrained(path).eval()
x=tok(f"我喜欢{tok.mask_token}语言",return_tensors="pt")
with torch.no_grad(): logits=model(**x).logits
pos=(x["input_ids"][0]==tok.mask_token_id).nonzero().item()
print(tok.convert_ids_to_tokens(logits[0,pos].topk(5).indices))`,
    output: "与 AutoModelForMaskedLM 相同地返回遮罩位置 top-5，但类名明确为 BERT。",
    pit: "看到文件夹名含 bert 就强行用 BertForMaskedLM；应以 config.model_type 和任务头为准。",
    quiz: "什么时候 Auto 类更合适？", answer: "代码需要兼容多种检查点架构，而不想为 BERT、RoBERTa 等分别改类名时。"
  },
  {
    p: 171, slug: "classification-data-loading", title: "中文分类案例（一）：先把原始数据加载成 text/label 样本",
    problem: "微调分类模型前，怎样确认每条文本、标签映射和训练/验证划分都没有错位？",
    chain: ["读取酒店评论 CSV", "检查 sentence/label", "分别加载 train/test/validation", "打印样本与长度", "返回三个 Dataset"],
    extraVisuals: [classificationFlow],
    article: article([
      ["0:00–6:38", "三个收尾案例与酒店评论数据", "老师说明 NLP 阶段最后做三个迁移案例：中文评论二分类、中文填空、中文句子关系。分类数据是之前用过的酒店评论，`label=1` 表示好评、`label=0` 表示差评，文本字段是 `sentence`；已经提供 train、test、validation 三个 CSV，所以不再在代码里重新切分。目标是用预训练中文 BERT 抽取 768 维文本表示，再接自己的全连接二分类层。"],
      ["6:38–18:10", "用 datasets.load_dataset 读取单个 CSV", "老师创建数据目录并复制三个文件，导入 Hugging Face `datasets`。第一种写法把文件路径放进 `data_files` 字典，再用 `load_dataset('csv', data_files=...)`，返回 DatasetDict；第二种直接指定一个 CSV 再取对应 split。课堂两种都演示，是为了看懂 API，不是要求项目重复加载两遍。加载后打印前三条、特征字段和长度，确认解析无错。"],
      ["18:10–27:04", "两种加载写法的差异", "DatasetDict 方式适合一次组织多个 split，单文件方式适合函数按路径加载。老师发现两种结果相同，后续选择更直接的单文件写法。训练集打印约 9600 条，前几条标签如 1、1、0；这些数字只对应课堂文件。正式学习还应检查 0/1 分布、空文本和重复样本。"],
      ["27:04–32:10", "测试集、验证集与统一返回", "用同样方法加载 test 与 validation，验证集约 1200 条，并打印前三条和长度。函数最后返回三个 Dataset，后续预处理、训练和评估直接复用。这里已经有独立文件，因此不要再对测试集二次切割，更不能把 test 用于调参。"],
    ]),
    points: ["课堂数据字段是 sentence 与 0/1 label", "train/test/validation 已分好，不再重复切分", "加载后必须打印样本、字段和长度核查"],
    code: `from datasets import load_dataset
train = load_dataset("csv", data_files="data/train.csv", split="train")
test = load_dataset("csv", data_files="data/test.csv", split="train")
valid = load_dataset("csv", data_files="data/validation.csv", split="train")
print(train[:3], len(train), train.features)`,
    output: "得到三个 Dataset；`split='train'` 是 CSV 加载器生成的 split 名，不表示把 test 文件当训练数据。",
    pit: "看到 `split='train'` 就误以为测试 CSV 参与训练；这里它只是单文件 DatasetDict 内部的默认 split 键。",
    quiz: "既然已有 validation 和 test，二者分别干什么？", answer: "validation 用于训练过程选模型/参数，test 留到最后做一次公正评估。"
  },
  {
    p: 172, slug: "classification-preprocessing", title: "中文分类案例（二）：批量分词、padding、truncation 与 attention_mask",
    problem: "一批句子长短不同，怎样变成规则的 `[B,L]` 张量又不让模型关注 PAD？",
    chain: ["收集一批文本", "tokenizer 批量编码", "截断到 max_length", "补齐到同一 L", "附加 labels"],
    extraVisuals: [classificationFlow],
    article: article([
      ["0:00–3:56", "整理函数一次只处理一批", "老师定义 `collate_fn1(data)`，强调传入的不是整个 Dataset，而是 DataLoader 取出的 8 条样本。先从每条字典中取 `sentence` 列表和 `label` 列表，再统一交给预训练模型配套 tokenizer。"],
      ["3:56–9:50", "batch_encode_plus 的课堂参数", "课程使用 `batch_encode_plus`：`truncation=True` 开启截断，`max_length=300` 固定最大序列长度，`padding='max_length'` 补到 300，`return_tensors='pt'` 返回 PyTorch 张量。300 是老师为了覆盖大多数评论直接选的教学值；更严谨做法是先统计 token 长度分布再决定。"],
      ["9:50–14:48", "取出四类张量", "编码结果包含 `input_ids`、`token_type_ids`、`attention_mask`，再把 labels 转成长整型张量并一起返回。若 B=8、L=300，前三者是 `[8,300] = 8 条评论 × 每条 300 个位置`，labels 是 `[8]`。token_type_ids 区分句段，attention_mask 区分有效位置与 PAD。"],
      ["14:48–20:19", "DataLoader 调用 collate_fn", "用 `DataLoader(dataset, batch_size=8, shuffle=..., drop_last=..., collate_fn=collate_fn1)`，每取一批都自动执行整理函数。老师只取第一批打印，逐项确认四个结果和 8 个 0/1 标签。固定补到 300 便于观察，但动态 padding 通常更省显存。"],
    ]),
    points: ["max_length 按 token 计数", "attention_mask 屏蔽 padding", "动态 padding 更省算力"],
    code: `import torch
def collate_fn(rows):
    encoded=tokenizer.batch_encode_plus(
        [r["sentence"] for r in rows],
        truncation=True,max_length=300,padding="max_length",
        return_tensors="pt",
    )
    return (
        encoded["input_ids"],
        encoded.get("token_type_ids"),
        encoded["attention_mask"],
        torch.tensor([r["label"] for r in rows],dtype=torch.long),
    )`,
    output: "返回含 `input_ids [B,L]`、`attention_mask [B,L]`、`labels [B]` 的批次字典。",
    pit: "把 max_length 设得远大于实际文本，导致大量 padding、显存浪费和训练变慢。",
    quiz: "为什么 labels 不需要 padding？", answer: "句子级分类每条文本只有一个标签，所以批次标签天然是固定长度 `[B]`。"
  },
  {
    p: 173, slug: "classification-model", title: "中文分类案例（三）：自定义 BERT + Linear(768→2) 网络",
    problem: "老师没有直接用现成分类模型，而是怎样把预训练 BERT 的 768 维表示接到自己的二分类层？",
    chain: ["定义 nn.Module", "保存预训练 BertModel", "取 pooled/CLS 表示 [B,768]", "Linear 输出 [B,2] logits", "交给 CrossEntropyLoss"],
    extraVisuals: [autoUml, classificationFlow],
    article: article([
      ["0:00–5:20", "自定义类的两个成员", "老师继承 `nn.Module`，在 `__init__` 中保存已经加载的中文 `BertModel`，再定义 `nn.Linear(768,2)`。768 来自该 BERT Base 的 hidden size，2 来自好评/差评两个类别；若换模型或类别数，这两个数字都要随 config/数据改变。"],
      ["5:20–10:40", "forward 的三步", "forward 接收 `input_ids`、`token_type_ids`、`attention_mask`，先传进预训练 BERT。课堂取 BERT 的 pooled output/CLS 级表示 `[B,768]`，再经线性层得到 `[B,2]`。老师随后用 softmax 转概率；训练用 CrossEntropyLoss 时更推荐直接返回原始 logits，因为该损失内部已含 log-softmax。"],
      ["10:40–16:00", "手工测试形状", "用一批全 1 或示例张量调用模型，打印 BERT 输出与最终结果形状。形状检查是为了在正式训练前发现参数名、batch/length 维度和 768→2 映射错误。模型转到 device 后，测试输入也必须同设备。"],
      ["16:00–20:00", "冻结发生在训练函数", "本节先搭网络；下一节训练时把预训练 BERT 参数 `requires_grad=False`，只更新自己的线性层。这是“固定特征提取器”迁移方式，而不是全量微调。纠正一个易错点：CrossEntropyLoss 要 logits，不需要 forward 先 softmax。"],
    ]),
    points: ["分类头的输出维必须等于类别数", "新任务头随机初始化是正常现象", "模型和输入必须同 device"],
    code: `import torch.nn as nn
class HotelClassifier(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.pre_model=bert
        self.classifier=nn.Linear(bert.config.hidden_size,2)
    def forward(self,input_ids,token_type_ids,attention_mask):
        out=self.pre_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        return self.classifier(out.pooler_output)`,
    output: "输入 `[B,L]` 三类张量，输出 `[B,2] = B 条评论 × 好/差评两个 logits`。",
    pit: "在线性层后先 softmax 再交给 CrossEntropyLoss；会重复做归一化并影响数值稳定。",
    quiz: "768 和 2 分别由什么决定？", answer: "768 由预训练 BERT 的 hidden_size 决定；2 由目标任务好评/差评两个类别决定。"
  },
  {
    p: 174, slug: "classification-training", title: "中文分类案例（四）：训练循环、梯度更新和逐轮保存",
    problem: "一个批次从 DataLoader 到参数更新完整经历哪些步骤，为什么顺序不能乱？",
    chain: ["加载训练集与模型", "冻结预训练 BERT", "CrossEntropy + Adam", "每批 forward/backward/step", "每轮保存 state_dict"],
    extraVisuals: [classificationFlow],
    article: article([
      ["0:00–5:47", "训练函数的准备阶段", "老师按流程写：加载训练 Dataset，创建自定义模型并移到 device，然后遍历 `model.pre_model.parameters()` 把 `requires_grad=False`。因此预训练 BERT 只做固定特征提取，真正学习的是 768→2 的线性层。损失用 `nn.CrossEntropyLoss()`，优化器用 Adam。"],
      ["5:47–14:30", "三轮、逐批训练", "设置 `model.train()`，外层训练 3 轮；每轮重新获取 DataLoader，记录开始时间。每批把 `input_ids/token_type_ids/attention_mask/labels` 移到 device，前向得到 `[B,2]`，算 CrossEntropy，随后 `zero_grad → backward → step`。因为 BERT 已冻结，自动微分只为分类层保留梯度。"],
      ["14:30–23:40", "每 20 批打印局部结果", "课堂每隔 20 批打印当前一段的 loss/accuracy。这里的 loss 和准确率只覆盖最近/当前 20 批，不是整轮指标，所以会明显震荡；若要整轮结果，应在循环外维护 total_loss、correct、total。老师现场解释了为什么某 20 批可从 25% 跳到 100%。"],
      ["23:40–29:20", "清理调试输出与逐轮保存", "先移除 forward 内每批打印输入的调试语句，避免日志刷屏。每轮完成后保存 `state_dict`，文件名含轮次 classification1/2/3；IDE 文件树没刷新时从磁盘 reload。课程为继续讲解可先用第一轮模型，但正式项目应依据 validation 指标选择，而不是默认第三轮最好。"],
    ]),
    points: ["train() 与 eval() 控制 dropout 等行为", "PyTorch 梯度默认累积", "最佳验证 checkpoint 通常比最后一轮更可靠"],
    code: `for p in model.pre_model.parameters():
    p.requires_grad=False
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()))
model.train()
for ids,types,mask,labels in train_loader:
    ids,types,mask,labels=[x.to(device) for x in (ids,types,mask,labels)]
    logits=model(ids,types,mask)
    loss=criterion(logits,labels)
    optimizer.zero_grad(); loss.backward(); optimizer.step()`,
    output: "每批只更新分类层；BERT 参数没有梯度。",
    pit: "以为设置 requires_grad=False 等于 model.eval()。冻结控制参数梯度，train/eval 控制 dropout 等运行行为，是两回事。",
    quiz: "为什么课堂训练快于全量微调？", answer: "BERT 主体被冻结，不计算/保存其参数梯度，只训练很小的线性分类头。"
  },
  {
    p: 175, slug: "classification-evaluation", title: "中文分类案例（五）：加载 checkpoint、全量准确率与评估边界",
    problem: "验证时怎样保证不更新参数、不受 dropout 随机性影响，并正确汇总整套数据？",
    chain: ["加载课堂第 3 轮 checkpoint", "model.eval", "torch.no_grad", "逐批 logits→预测", "累计全量准确率"],
    extraVisuals: [classificationFlow],
    article: article([
      ["0:00–3:49", "测试 DataLoader 与训练不同", "评估函数加载 test CSV 并创建 DataLoader。老师把原来写死训练路径的加载器逻辑改成可接收 Dataset；测试时 `shuffle=False`，通常也不应 `drop_last=True`，否则会漏掉最后不足一批的样本。"],
      ["3:49–6:44", "加载 state_dict", "重新创建同结构模型，选择保存的 classification checkpoint，调用 `load_state_dict(torch.load(...))` 加载参数并移到 device。课堂倾向选择第三轮，但严格做法应先用 validation 选最佳轮次，再只在 test 上评一次。"],
      ["6:44–12:44", "eval/no_grad 与全量准确率", "初始化 correct/total，调用 `model.eval()`；逐批迁移四个张量，在 `torch.no_grad()` 下前向，对 `[B,2]` 取最高类别，累计正确数和总数。最后 `correct/total` 才是整个测试集准确率。老师对比训练时每 20 批局部指标，说明两者统计范围不同。"],
      ["12:44–18:06", "课堂总结与二分类损失纠问", "老师复盘数据加载、每批 tokenizer、四个返回张量、BERT 768 维表示和自定义 2 维分类。学生问二分类为何不用 BCE：两维互斥 logits 配整数标签可以用 CrossEntropy；若改为单 logit，则配 BCEWithLogitsLoss 和浮点 0/1 标签。CrossEntropy 方案也便于把输出改为 3、4 类。"],
    ]),
    points: ["eval() 和 no_grad() 不能互相替代", "指标按样本总数加权", "模型与 tokenizer 一起 save_pretrained"],
    code: `state=torch.load("models/classification3.pt",map_location=device)
model.load_state_dict(state)
model.eval(); correct=total=0
with torch.no_grad():
    for ids,types,mask,labels in test_loader:
        ids,types,mask,labels=[x.to(device) for x in (ids,types,mask,labels)]
        pred=model(ids,types,mask).argmax(-1)
        correct+=(pred==labels).sum().item()
        total+=labels.numel()
print(correct/total)`,
    output: "输出整个验证集按样本计算的准确率。",
    pit: "只调用 no_grad 不调用 eval，dropout 仍随机工作，导致同一输入结果波动。",
    quiz: "最后一个 batch 较小时，为什么不能简单平均每批 accuracy？", answer: "那会让小批与满批权重相同；应累计正确样本数再除总样本数。"
  },
  {
    p: 176, slug: "mlm-preprocessing", title: "中文填空案例（一）：固定遮罩下标 16 的数据整理",
    problem: "课堂代码怎样把完整酒店评论改成“下标 16 被遮住、标签是原 token ID”的分类样本？",
    chain: ["取一批完整评论", "编码到长度 32", "保存下标 16 原 token ID", "把下标 16 改成 MASK ID", "返回四个输入与标签"],
    extraVisuals: [fixedMaskFlow, mlmFlow],
    article: article([
      ["0:00–4:55", "把填空看成 21128 类分类", "老师用中文 BERT 词表举例：若词表大小是 21128，那么空位就有 21128 个候选，本质是一个大规模多分类。课堂继续复用酒店评论 train/test/validation，而不是另找语料。老师先口头说可随机遮罩若干词，但为了代码简单，实际案例固定处理一个位置。"],
      ["4:55–11:50", "复制分类案例并改 collate_fn", "数据加载与上一案例相同，主要改批整理函数：每批 8 条评论，用 tokenizer 截断/补齐到长度 32，得到 `input_ids/token_type_ids/attention_mask`。这里必须以视频里的实际代码为准：代码使用 `input_ids[:, 16]`，Python 从 0 开始，因此处理的是代码下标 16、自然计数第 17 个 token。老师口头仍把它叫“第 16 个位置”，这是口述与代码下标混用，不能擅自改成下标 15。"],
      ["11:50–18:50", "先存答案，再写入 MASK", "从原始 input_ids 取目标位置 ID 作为 `labels [B]`；随后把同一位置替换为 `tokenizer.mask_token_id`。这时 `input_ids [B,32]` 是带遮罩输入，labels `[B]` 是每条评论原来被遮住的一个 token ID。模型只预测这一个位置，不是标准 BERT MLM 的全位置 `[B,L]` labels。"],
      ["18:50–27:59", "DataLoader 测试与教学简化的边界", "老师把新整理函数接进 DataLoader，打印一批确认下标 16 确实变成 MASK、labels 保存原 ID。这个固定位置方案便于复用分类训练循环，但不是经典 15% 动态遮罩/80-10-10；短文本若该位置是 PAD，预测没有意义，因此训练和评估阶段会先过滤真实长度大于 32 的文本。"],
    ]),
    points: ["课堂实现固定预测一个位置", "必须先保存原 token ID，再把输入换成 MASK", "这是教学简化，不等于完整 BERT MLM 数据构造"],
    code: `def fill_mask_collate(rows):
    enc=tokenizer(
        [r["sentence"] for r in rows],
        padding="max_length",truncation=True,max_length=32,
        return_tensors="pt",
    )
    pos=16  # 复现课堂代码：下标 16，即自然计数第 17 个 token
    labels=enc["input_ids"][:,pos].clone()
    enc["input_ids"][:,pos]=tokenizer.mask_token_id
    return enc["input_ids"],enc.get("token_type_ids"),enc["attention_mask"],labels`,
    output: "输入三类张量是 `[B,32]`，标签是 `[B]`，每条只监督一个词表 ID。",
    pit: "先覆盖为 MASK 再取 labels；这样标签会全变成 mask_token_id。",
    quiz: "课堂 labels 为什么是 `[B]` 而不是标准 MLM 常见的 `[B,L]`？", answer: "课堂只预测每条文本固定的一个位置，所以每条只需要一个目标 ID。"
  },
  {
    p: 177, slug: "mlm-model", title: "中文填空案例（二）：自定义 BERT + Linear(768→词表大小)",
    problem: "怎样只取每条文本代码下标 16 的 768 维表示，并预测整个中文词表？",
    chain: ["BERT 输出 [B,32,768]", "选固定位置 [B,768]", "Linear 到 [B,V]", "CrossEntropy 预测 token ID", "手工测试形状"],
    extraVisuals: [fixedMaskFlow, autoUml],
    article: article([
      ["0:00–1:58", "从分类网络复制后改输出维", "老师复用前面的自定义 `nn.Module`，BERT 主体不变，把 `Linear(768,2)` 改成 `Linear(768, tokenizer.vocab_size)`；课堂中文词表约 21128。用 `vocab_size` 而非把 21128 写死，换 tokenizer 时更安全，并关闭 bias 以减少少量参数。"],
      ["1:58–4:53", "只取被遮罩位置", "BERT 的 `last_hidden_state [B,32,768]` 含所有位置。课堂实际代码用 `hidden[:,16,:]` 取 `[B,768]`，再过线性层得到 `[B,21128] = B 条评论 × 每条在整个词表上的 logits`。这里必须与数据整理阶段的 `input_ids[:,16]` 完全一致；它是下标 16，也就是自然计数第 17 个 token。"],
      ["4:53–8:57", "测试模型结构", "用 DataLoader 的一批输入测试前向，检查输出第二维确实等于词表大小。训练时用 CrossEntropyLoss 直接接 logits 和 labels `[B]`；不必先 softmax。老师强调其他初始化、设备移动与分类模型几乎相同，只改任务头和抽取位置。"],
    ]),
    points: ["课堂模型只输出一个位置的 `[B,V]`", "V 由 tokenizer.vocab_size 决定", "CrossEntropyLoss 接 logits，不接 softmax 概率"],
    code: `class FixedPositionFillMask(torch.nn.Module):
    def __init__(self,bert,tokenizer):
        super().__init__()
        self.pre_model=bert
        self.linear=torch.nn.Linear(bert.config.hidden_size,tokenizer.vocab_size,bias=False)
    def forward(self,ids,types,mask):
        h=self.pre_model(input_ids=ids,token_type_ids=types,attention_mask=mask).last_hidden_state
        return self.linear(h[:,16,:])`,
    output: "若 B=8、V=21128，输出 `[8,21128]`。",
    pit: "把整个 `[B,L,H]` 直接传 Linear 后得到 `[B,L,V]`，却仍拿 `[B]` 标签计算而没有选位置。",
    quiz: "为什么线性层输入是 768？", answer: "被遮罩位置从 BERT 取出的单个 token 上下文表示维度是 hidden_size=768。"
  },
  {
    p: 178, slug: "mlm-training", title: "中文填空案例（三）：过滤长文本并复用分类训练循环",
    problem: "为什么训练前要保留真实长度大于 32 的评论，哪些地方与分类训练不同？",
    chain: ["加载训练 CSV", "filter 长度>32", "使用填空 collate_fn", "冻结 BERT/训练词表头", "逐轮保存 FillMask 模型"],
    extraVisuals: [fixedMaskFlow],
    article: article([
      ["0:00–3:55", "为什么先过滤长度大于 32", "课堂把分类训练函数复制过来，第一处修改是 `dataset.filter(lambda x: len(x['sentence']) > 32)`。如果原文很短，补到 32 后代码下标 16 可能是 PAD，拿 PAD 当真实填空标签没有意义。老师选择长文本，确保固定遮罩位置来自原文。严格说字符长度不等于 tokenizer 后 token 长度，稳妥做法应按编码后的有效 token 数过滤。"],
      ["3:55–7:30", "三处关键修改", "数据改用过滤后的训练集，DataLoader 的整理函数换成填空版本，模型换成 768→词表大小的填空网络；保存文件名换成 FillMask1/2/3。其余 device、冻结预训练 BERT、CrossEntropy、Adam、三轮循环基本复用。"],
      ["7:30–12:19", "损失和预测", "labels 是每条原文固定位置的词表 ID `[B]`，模型 logits `[B,V]`，CrossEntropy 直接计算。每 20 批可打印局部 loss 和 token top-1 准确率，但仍不是整轮指标。老师强调复用代码时要逐项改路径、函数和模型名，不能只改标题。"],
    ]),
    points: ["过滤是为避免预测 PAD", "课堂只训练自定义词表头", "字符长度过滤是简化，token 长度更严谨"],
    code: `train_ds=load_dataset("csv",data_files="data/train.csv",split="train")
train_ds=train_ds.filter(lambda x: len(x["sentence"])>32)
loader=DataLoader(train_ds,batch_size=8,shuffle=True,collate_fn=fill_mask_collate)
for ids,types,mask,labels in loader:
    logits=model(ids.to(device),types.to(device),mask.to(device))
    loss=torch.nn.functional.cross_entropy(logits,labels.to(device))
    optimizer.zero_grad(); loss.backward(); optimizer.step()`,
    output: "每批用固定位置原 token ID 监督 `[B,V]` 输出。",
    pit: "按 Python 字符数判断长度，却忽略 tokenizer 可能加入特殊 token 或拆分子词。",
    quiz: "为什么代码下标 16 若是 PAD 会破坏训练？", answer: "模型会反复学习预测 PAD，而不是根据真实上下文恢复有意义的原 token。"
  },
  {
    p: 179, slug: "mlm-evaluation", title: "中文填空案例（四）：加载 FillMask 模型并计算固定位置准确率",
    problem: "怎样让测试数据采用与训练完全相同的过滤和遮罩规则，并解释约 70% 的结果？",
    chain: ["加载 test CSV", "过滤长度>32", "使用填空整理函数", "加载 FillMask checkpoint", "比较预测 ID 与真实 ID"],
    extraVisuals: [fixedMaskFlow],
    article: article([
      ["0:00–2:52", "评估代码仍只改三处", "加载 test CSV，先过滤 sentence 长度大于 32；DataLoader 使用填空 `collate_fn2`；模型路径从 classification 改为 FillMask checkpoint。训练和评估的数据规则必须一致，否则准确率不可比较。"],
      ["2:52–4:48", "固定位置 top-1", "model.eval/no_grad 前向得到 `[B,V]`，argmax 得每条预测 token ID，与 labels `[B]` 比较并累计 correct/total。课堂跑出的准确率大约 70%，表示代码下标 16（自然计数第 17 个 token）的 top-1 命中率，不是整句完形填空准确率，也不能外推到随机位置。"],
      ["4:48–8:28", "结果为什么看似较高", "固定位置、同领域酒店评论、冻结的预训练 BERT 和词频分布都可能让任务较容易。除了 top-1，还应看 top-5、按 token 频次分组和可读候选，避免模型只会猜常见字。老师完成评估后进入 NSP 句子关系任务。"],
    ]),
    points: ["测试必须复用相同过滤与遮罩位置", "70% 是固定位置 token top-1", "应增加 top-k 与错误样例分析"],
    code: `model.eval(); hit=total=0
with torch.no_grad():
    for ids,types,mask,labels in test_loader:
        ids,types,mask,labels=[x.to(device) for x in (ids,types,mask,labels)]
        pred=model(ids,types,mask).argmax(-1)
        hit+=(pred==labels).sum().item()
        total+=labels.numel()
print(hit/total)`,
    output: "输出课堂固定遮罩位置的词表 top-1 准确率。",
    pit: "用 classification3 checkpoint 加载填空网络；层形状不同会报错或根本不是同一任务。",
    quiz: "为什么约 70% 不能说明任意位置填空都有 70%？", answer: "训练和测试都固定同一位置及同一领域，未覆盖其他位置和更一般文本。"
  },
  {
    p: 180, slug: "nsp-custom-dataset", title: "NSP 案例（一）：自定义句对数据，构造真下一句与随机下一句",
    problem: "老师怎样把一条长度至少 44 的评论切成两个 22 字片段，并随机替换后半句制造负样本？",
    chain: ["过滤长度≥44", "前 22 字作句 A", "后 22 字作真实句 B", "随机替换 B 制造负例", "返回 sentence1/2/label"],
    extraVisuals: [nspFlow],
    article: article([
      ["0:00–4:46", "任务与数据构造思路", "NSP 判断第二句是否为第一句的下半/后续。老师仍用酒店评论，先筛选真实字符长度至少约 44 的样本，取前 22 字作为 sentence1、后 22 字作为真实 sentence2，保证两段都有内容而不是 padding。这里是从同一评论硬切两段，不是按自然句号切句。"],
      ["4:46–12:30", "为什么要自定义 Dataset", "前两个案例可直接使用 datasets.Dataset；NSP 需要 `__getitem__` 每次生成句对和标签，所以老师继承 PyTorch Dataset，自行实现长度与取样。正样本保留本条后 22 字；负样本随机选另一条评论的后 22 字替换 sentence2。"],
      ["12:30–20:30", "随机标签与替换细节", "课堂通过随机数决定正/负，并在负例时随机索引其他样本。标签的 0/1 含义必须与代码统一；不要凭 BERT 文档默认值猜。还要避免随机到当前同一条，尽量让负样本真正不连续。"],
      ["20:30–26:06", "测试自定义 Dataset", "打印若干样本，检查 sentence1、sentence2 都是 22 字，label 与是否替换一致。老师强调长度过滤发生在切分前，后续 tokenizer 再负责 `[CLS]/[SEP]`、padding 和 mask。"],
    ]),
    points: ["正例是相邻句，负例是替换后的随机句", "标签 0/1 约定要核对", "困难负样本能减少只学主题的捷径"],
    code: `import random
from torch.utils.data import Dataset
class NSPDataset(Dataset):
    def __init__(self,rows):
        self.rows=[r["sentence"] for r in rows if len(r["sentence"])>=44]
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        text=self.rows[i]
        a,true_b=text[:22],text[22:44]
        if random.random()<0.5:
            return a,true_b,0
        j=random.choice([j for j in range(len(self.rows)) if j!=i])
        return a,self.rows[j][22:44],1`,
    output: "返回两个各 22 字片段和二分类标签。",
    pit: "随机负样本抽到 i+1，导致同一句对被标成 NotNext。",
    quiz: "为什么全从别的主题抽负句可能太简单？", answer: "模型只需判断主题相似度，无需真正学习句间连续关系。"
  },
  {
    p: 181, slug: "nsp-preprocessing", title: "NSP 案例（二）：句对编码、token_type_ids 与特殊 token",
    problem: "BERT 怎样在同一 `[B,L]` 序列中知道哪些 token 属于句子 A、哪些属于句子 B？",
    chain: ["输入 text A/text B", "拼 [CLS] A [SEP] B [SEP]", "生成 token_type_ids", "padding/truncation", "附加 labels [B]"],
    extraVisuals: [nspFlow],
    article: article([
      ["0:00–4:50", "理解 DataLoader 传入的数据结构", "老师先打印一批，解释自定义 Dataset 经过默认收集后可能呈现 sentence1 列表、sentence2 列表和 label 张量，而不一定还是“8 个小字典”。因此 collate 函数要按真实批结构取值，不能照搬上一个案例的 `for item in data`。"],
      ["4:50–9:40", "句对 tokenizer", "将 8 个 sentence1 和 8 个 sentence2 成对交给 tokenizer/batch_encode_plus。BERT 自动拼 `[CLS] A [SEP] B [SEP]`，生成 `input_ids`、`token_type_ids`、`attention_mask`。因为两段各 22 字，再加特殊 token，总长度可设成能容纳句对的固定值。"],
      ["9:40–15:37", "返回四项并测试", "collate 返回三类 `[B,L]` 输入和 labels `[B]`，接入 DataLoader 后只取一批打印。token_type_ids 的 0/1 区段帮助 BERT 区分 A/B；attention_mask 仍只表示有效 token/PAD。老师说明后续模型与分类案例高度相似。"],
    ]),
    points: ["句对用 tokenizer 的两个文本参数", "token_type_ids 区分 A/B 段", "不是所有 Transformer 都使用 segment embedding"],
    code: `import torch
def nsp_collate(rows):
    a,b,y=zip(*rows)
    batch=tokenizer(
        list(a),list(b),padding=True,truncation=True,
        max_length=128,return_tensors="pt",
    )
    batch["labels"]=torch.tensor(y)
    return batch`,
    output: "得到句对编码 `[B,L]` 和 NSP 标签 `[B]`。",
    pit: "手工在字符串里写 `[SEP]`，但 tokenizer 又自动加特殊 token，造成重复或普通文本化。",
    quiz: "attention_mask 与 token_type_ids 有什么不同？", answer: "前者区分有效 token/PAD；后者区分句子 A 段和句子 B 段。"
  },
  {
    p: 182, slug: "nsp-model", title: "NSP 案例（三）：复用自定义 BERT + Linear(768→2)",
    problem: "为什么课堂 NSP 模型几乎可以原样复制酒店评论二分类网络？",
    chain: ["句对输入 BERT", "取 pooled [B,768]", "Linear 到 [B,2]", "softmax/argmax 判断关系", "测试模型结构"],
    extraVisuals: [nspFlow, autoUml],
    article: article([
      ["0:00–1:57", "模型与分类案例一模一样", "老师直接复制酒店评论分类的自定义模型：预训练中文 BertModel 接收句对三类张量，取 pooled/CLS 级 `[B,768]`，再用 Linear(768,2) 输出是否连续的两个 logits。变化在数据和标签语义，不在网络形状。"],
      ["1:57–4:06", "用图解释负样本随机索引", "课堂重点反而回到 Dataset：多条 44 字评论各分 22+22；当随机标为负例时，从另一条的后 22 字覆盖当前 sentence2，于是 A/B 无真实连续关系。随后测试模型输出 `[B,2]`。"],
    ]),
    points: ["课堂没有调用 BertForNextSentencePrediction", "同一 `[B,2]` 形状可承载不同二分类语义", "真正区别来自句对构造和标签"],
    code: `class NSPClassifier(torch.nn.Module):
    def __init__(self,bert):
        super().__init__()
        self.pre_model=bert
        self.linear=torch.nn.Linear(bert.config.hidden_size,2)
    def forward(self,ids,types,mask):
        pooled=self.pre_model(
            input_ids=ids,token_type_ids=types,attention_mask=mask
        ).pooler_output
        return self.linear(pooled)`,
    output: "输出 `[B,2] = B 对句子 × 连续/不连续两个 logits`。",
    pit: "把课堂自定义二分类头说成直接使用 BERT 原生 NSP 预训练头；两者接口与权重来源不同。",
    quiz: "为什么 NSP 不是 `[B,L,2]`？", answer: "它判断整对句子的关系，每对只输出一次二分类；不是给每个 token 分类。"
  },
  {
    p: 183, slug: "nsp-training-evaluation", title: "NSP 案例（四）：训练、评估与数据捷径检查",
    problem: "即使 NSP 准确率很高，怎样判断模型学的是句间关系，而不是负样本构造漏洞？",
    chain: ["训练二分类", "验证集 eval", "累计 accuracy/F1", "检查困难负例", "保存完整 checkpoint"],
    extraVisuals: [nspFlow, classificationFlow],
    article: article([
      ["0:00–2:54", "复制训练函数并换数据整理版本", "老师复用分类训练代码，只把 DataLoader 换成 NSP Dataset/collate 版本，模型换成 NSP 自定义网络，保存名换成 NSP1/2/3。句对最大长度比分类的 300 短，因此一轮约几十秒，明显更快；日志仍每 20 批报告局部 loss/accuracy。"],
      ["2:55–4:37", "评估也只改三处", "复制分类评估函数，数据加载器换版本 3、checkpoint 路径换 NSP3、最后打印文本和预测/真实关系标签。其余 eval、no_grad、argmax、correct/total 都相同。"],
      ["4:37–10:30", "读懂预测标签", "输出中应同时展示原始 sentence1/2、模型预测关系和真实 label，才能发现随机负例是否构造正确。准确率只衡量当前人工规则；若负句总来自完全不同主题，模型可能靠主题差异走捷径。"],
      ["10:30–15:13", "三案例总复盘", "中文分类、固定位置填空、NSP 都遵循“加载/构造数据 → collate 成 BERT 输入 → 预训练 BERT 抽特征 → 自定义任务头 → 冻结主体训练 → 评估”。变化主要是标签形状、任务头输出维和 Dataset 构造。下一专题进一步解释 BERT 架构与 MLM/NSP 原理。"],
    ]),
    points: ["NSP 训练本质是句对二分类", "高准确率要排查负样本捷径", "最终保存模型、tokenizer、标签约定和数据构造版本"],
    code: `# 训练阶段与二分类相同：CrossEntropy + Adam，只训练自定义线性头
for p in model.pre_model.parameters():
    p.requires_grad=False
model.eval(); correct=total=0
with torch.no_grad():
    for ids,types,mask,labels in test_loader:
        ids,types,mask,labels=[x.to(device) for x in (ids,types,mask,labels)]
        pred=model(ids,types,mask).argmax(-1)
        correct+=(pred==labels).sum().item()
        total+=labels.numel()
print("NSP accuracy",correct/total)`,
    output: "输出验证集句对级准确率；正式评估还应分标签统计并检查困难负例。",
    pit: "负样本全部跨主题，模型只学主题不一致就达到高分。",
    quiz: "怎样构造更难的 NSP 负样本？", answer: "从同文档或同主题中选非相邻句，并尽量匹配长度/标点，让模型必须学习更细的连续关系。"
  },
];

generateTopic({
  topic: "迁移学习与 Transformers 任务",
  folder: "transfer-learning",
  lessons,
  stages: [
    { title: "一、迁移学习、模型家族与环境", start: 1, end: 3 },
    { title: "二、Pipeline 六类任务", start: 4, end: 9 },
    { title: "三、Auto 与具体类", start: 10, end: 16 },
    { title: "四、中文分类完整微调", start: 17, end: 21 },
    { title: "五、中文 MLM 完形填空", start: 22, end: 25 },
    { title: "六、NSP 句对任务", start: 26, end: 29 },
  ],
});
