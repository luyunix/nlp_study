import { generateTopic } from "./course_note_helpers.mjs";

const article = (sections) => sections.map(([time, title, body]) =>
  `### ${time}　${title}\n\n${body}`
).join("\n\n");

const fullPipeline = {
  title: "英法翻译从数据到预测的总流程",
  diagram: `
flowchart LR
    A["英法平行语料"] --> B["清洗与标准化"]
    B --> C["两套词表 + SOS/EOS"]
    C --> D["Dataset / DataLoader（batch=1）"]
    D --> E["GRU Encoder"]
    E --> F["Attention Decoder"]
    F --> G["Teacher Forcing 训练"]
    G --> H["保存检查点"]
    H --> I["逐词推理直到 EOS"]
`,
};

const architecture = {
  title: "Seq2Seq 模块 UML",
  diagram: `
classDiagram
    class EncoderGRU {
      embedding
      gru
      forward(input, hidden)
    }
    class AttnDecoderGRU {
      embedding
      dropout
      attn: Linear(2H,max_length)
      attn_combine: Linear(2H,H)
      gru
      out: Linear(H,target_vocab)
      log_softmax
      forward(input,hidden,encoder_outputs)
    }
    EncoderGRU --> AttnDecoderGRU : outputs + final hidden
`,
};

const encoderShapes = {
  title: "Encoder 的形状流",
  diagram: `
flowchart LR
    A["source_ids [B,S]"] --> B["Embedding [B,S,E]"]
    B --> C["GRU"]
    C --> D["encoder_outputs [B,S,H]"]
    C --> E["hidden [1,B,H]"]
`,
};

const decoderShapes = {
  title: "带注意力 Decoder 单步形状流",
  diagram: `
flowchart LR
    T["当前 token [1,1]"] --> E["Embedding [1,1,256]"]
    H["hidden [1,1,256]"] --> J1["拼接 [1,1,512]"]
    E --> J1
    J1 --> A["Linear 512→10 + Softmax"]
    A --> W["weights [1,1,10]"]
    O["固定 Encoder outputs [1,10,256]"] --> C["bmm → context [1,1,256]"]
    W --> C
    E --> J2["再拼接 [1,1,512]"]
    C --> J2
    J2 --> F["Linear 512→256 + ReLU"]
    F --> G["GRU → [1,1,256]"]
    G --> L["Linear + LogSoftmax → [1,4345]"]
`,
};

const trainingFlow = {
  title: "训练时一批数据的调用时序",
  diagram: `
sequenceDiagram
    participant D as DataLoader
    participant E as Encoder
    participant A as Attention Decoder
    participant L as NLLLoss
    participant O as 两个 Optimizer
    D->>E: 一条 source [1,S]
    E-->>A: 固定 [1,10,H] outputs 与 hidden
    A->>A: 初始输入 SOS
    loop 目标位置 0..T-1
      A->>A: 当前输入词 → log_probs
      A->>A: 按 Teacher Forcing 决定下一输入
      A->>L: log_probs 与真实 target_i
    end
    O->>O: zero_grad → backward → step
`,
};

const inferenceFlow = {
  title: "推理时逐词生成流程",
  diagram: `
flowchart TB
    A["编码源句并补成固定 10 步"] --> B["输入 SOS"]
    B --> C["Decoder 单步"]
    C --> D["topk(1) 选概率最大 token"]
    D --> E{"是 EOS 或到最大长度？"}
    E -->|"否"| F["把预测 token 作为下一输入"]
    F --> C
    E -->|"是"| G["还原目标文本与注意力矩阵"]
`,
};

const deviceFlow = {
  title: "设备选择与张量迁移",
  diagram: `
flowchart LR
    A["检查 CUDA/MPS/CPU 可用性"] --> B["选择唯一 device"]
    B --> C["model.to(device)"]
    B --> D["每批 x/y.to(device)"]
    C --> E["运算设备一致"]
    D --> E
`,
};

const dataUml = {
  title: "语料与加载类的职责",
  diagram: `
classDiagram
    class Vocabulary {
      token_to_id
      id_to_token
      encode(text)
      decode(ids)
    }
    class TranslationDataset {
      pairs
      en_word2index
      fr_word2index
      __getitem__()
    }
    class DataLoader {
      batch_size = 1
      shuffle = true
    }
    TranslationDataset --> Vocabulary
    TranslationDataset --> DataLoader
`,
};

const commonCode = `import torch\nfrom seq2seq_from_scratch.model import EncoderGRU, AttentionDecoderGRU, Seq2Seq\nencoder=EncoderGRU(100,16,32)\ndecoder=AttentionDecoderGRU(120,16,32)\nmodel=Seq2Seq(encoder,decoder,start_id=1,end_id=2)\nprint(type(model).__name__)`;

const lessons = [
  {
    p: 80, slug: "translation-requirements", title: "英译法需求：先看懂 800 行项目的六个模块",
    codeLabel: "配套现代化实现示例（注意力公式与课堂版不同）",
    problem: "英译法项目代码很多，怎样先建立全景，不被某一个函数带跑？",
    chain: ["平行语料", "数据与词表", "Encoder", "Attention Decoder", "训练", "预测/可视化"],
    extraVisuals: [fullPipeline, architecture],
    article: article([
      ["0:00–4:54", "任务与章节路线", "老师说明输入英文句子、输出法文句子，是 N→M 的 Seq2Seq。项目会覆盖 CUDA、清洗、Dataset/DataLoader、GRU Encoder、无/有注意力 Decoder、Teacher Forcing、训练、预测与注意力热力图。"],
      ["4:54–9:52", "为什么代码量大", "算法主体并非 800 个新知识点，代码量来自数据管道、三类模型组件、训练日志、保存加载和测试。先按模块理解接口，再串成主流程。"],
      ["9:52–14:53", "模型本质是反复做法语词分类", "老师把英文经 Encoder 压成中间表示，再由 Decoder 逐时间步生成法语。每一步都会对约 4345 个法语词给出概率，概率最高的候选作为预测，因此从输出层角度看是重复进行多分类。课程后续词表会加入 SOS 与 EOS；本案例没有定义 PAD 或 UNK。"],
    ]),
    points: ["翻译是输入输出可变长的 N→M", "源/目标需要独立词表", "先学模块接口，再学整条训练线"],
    code: commonCode,
    output: "创建一个包含 Encoder 和 Attention Decoder 的 Seq2Seq 对象。",
    pit: "不要一开始从第一行照抄到最后；先画清数据和模块边界。",
    quiz: "为什么英语和法语通常不用同一词表？", answer: "字符、词形和频率不同，独立映射更清楚；共享词表是另一种可选设计。"
  },
  {
    p: 81, slug: "cuda-concepts", title: "CUDA 环境（上）：GPU、驱动、工具包与 PyTorch 不是同一层",
    problem: "看到显卡、CUDA、cuDNN、PyTorch 版本时，怎样知道它们各负责什么？",
    chain: ["确认硬件", "确认驱动", "确认 PyTorch 构建", "检查 torch.cuda", "选择 device"],
    extraVisuals: [deviceFlow],
    article: article([
      ["0:00–6:55", "四层概念", "老师区分 NVIDIA GPU、显卡驱动、CUDA 工具链和 PyTorch。训练程序最终通过框架调用 GPU；安装某个 CUDA 工具包不等于当前 PyTorch 一定能使用。"],
      ["6:55–13:44", "先查再装", "先确认机器是否有受支持 GPU，再看 PyTorch 能否识别。课程展示命令与版本只对应当时环境；今天安装应以 PyTorch 官方选择器为准，不能盲抄旧版本号。"],
      ["13:44–20:20", "最小验证", "核心检查是 torch.cuda.is_available()、device_count、当前设备名，并做一个小张量运算。模型与数据必须在同一 device。没有 CUDA 也可先用 CPU 跑通小样例。"],
    ]),
    points: ["硬件、驱动、CUDA、框架是不同层", "旧安装命令可能过时", "先跑最小张量验证"],
    code: `import torch\nprint("cuda",torch.cuda.is_available())\ndevice=torch.device("cuda" if torch.cuda.is_available() else "cpu")\nprint(device)`,
    output: "打印当前环境是否可用 CUDA，并安全回退 CPU。",
    pit: "不要只看 nvidia-smi 就认定 PyTorch 已正确安装 CUDA 构建。",
    quiz: "模型在 GPU、数据在 CPU 会怎样？", answer: "运算会报设备不一致错误；两者必须迁移到同一 device。"
  },
  {
    p: 82, slug: "cuda-practice", title: "CUDA 环境实操：创建环境、安装、验证与排错",
    problem: "怎样用最少步骤验证环境，而不是一次装很多包后不知道哪里错？",
    chain: ["新建隔离环境", "安装匹配 PyTorch", "import torch", "小张量上 GPU", "再装项目依赖"],
    extraVisuals: [deviceFlow],
    article: article([
      ["0:00–4:51", "隔离环境", "老师现场创建独立环境，避免 NLP 项目与其他依赖冲突。先只安装 PyTorch，再做验证。"],
      ["4:51–8:53", "三层测试", "第一层能 import；第二层 is_available 为真；第三层把张量放到 cuda 并完成加法/乘法。任何一步失败都在该层排错，不急着装全部依赖。"],
      ["8:53–10:58", "常见失败", "包括驱动太旧、装到另一个 Python、CPU 版 torch、环境没激活。用当前解释器的 `python -m pip` 比裸 pip 更不易装错环境。"],
    ]),
    points: ["先装核心框架并最小验证", "确认 pip 与 python 属于同一环境", "逐层排错"],
    code: `import torch\nif torch.cuda.is_available():\n    x=torch.tensor([1.,2.],device="cuda")\n    print((x*x).device,(x*x).tolist())\nelse:\n    print("CPU 模式仍可学习流程")`,
    output: "CUDA 可用时在 GPU 完成运算，否则清楚显示 CPU 回退。",
    pit: "安装成功日志不等于运行时真的使用了目标解释器。",
    quiz: "为什么先不装项目全部包？", answer: "减少变量，能更快定位是 PyTorch/GPU 还是其他依赖问题。"
  },
  {
    p: 83, slug: "cuda-summary", title: "CUDA 配置总结：把可复现信息写进项目",
    problem: "环境能跑以后，还要记录什么才能让下次或别人复现？",
    chain: ["记录 Python", "记录 torch 构建", "记录驱动/设备", "锁定依赖", "保存最小诊断脚本"],
    extraVisuals: [deviceFlow],
    article: article([
      ["0:00–1:59", "复盘配置顺序", "老师把硬件、驱动、环境、PyTorch 安装和测试压缩为一条排错路径。"],
      ["1:59–2:50", "工程化补充", "保存 Python/torch 版本、设备名、依赖清单和随机种子。不要把某台机器的绝对路径或驱动安装包写死在训练代码里。"],
    ]),
    points: ["环境信息也是实验结果的一部分", "CPU 跑通逻辑后再优化 GPU", "安装命令应从官方当前页面获取"],
    code: `import platform,torch\nprint(platform.python_version())\nprint(torch.__version__)\nprint(torch.cuda.is_available())`,
    output: "打印最关键的可复现环境信息。",
    pit: "只保存模型权重，不记录代码与词表版本，未来仍可能无法复现。",
    quiz: "随机种子能保证所有 GPU 环境完全一致吗？", answer: "不一定；某些算子和硬件仍可能非确定，但记录种子是必要基础。"
  },
  {
    p: 84, slug: "data-cleaning", title: "数据清洗：规范文本，但不要改坏翻译含义",
    problem: "平行语料怎样同时清洗，确保英文与法文仍一一对应？",
    chain: ["读取一对句子", "Unicode 规范化", "小写/标点规则", "过滤异常与长度", "同步保留 source/target"],
    extraVisuals: [dataUml],
    article: article([
      ["0:00–5:58", "清洗函数的职责", "老师对文本做 Unicode 到 ASCII 的规范、小写、标点分隔和非法字符过滤，让词表更稳定。英文和法文要用与语言匹配的规则。"],
      ["5:58–11:53", "成对清洗", "每行是英文与法文一对；任何过滤都必须同时作用到整对，不能只删一边。空行、列数错误和无法解析字符应记录数量。"],
      ["11:53–15:40", "别把信息洗掉", "法语重音、撇号可能承载真实词形；课程简化规则服务教学。生产任务应保留原文列，抽样比较清洗前后，并用 tokenizer 代替过度正则。"],
    ]),
    points: ["平行语料必须成对保留/删除", "清洗规则与语言相关", "保留原始文本便于回溯"],
    code: `import re,unicodedata\ndef clean(s):\n    s=unicodedata.normalize("NFKC",s.strip().lower())\n    return re.sub(r"\\s+"," ",s)\nprint(clean("  Hello   world! "))`,
    output: "得到 `hello world!`，仅规范大小写和空白。",
    pit: "把所有非 ASCII 字符删除可能破坏法语重音与专名。",
    quiz: "一对句子中目标为空，能只保留源句吗？", answer: "不能用于有监督翻译；输入与目标会失去配对。"
  },
  {
    p: 85, slug: "preprocessing", title: "数据预处理：读取双语句对并建立两套词表",
    problem: "清洗函数写好后，怎样把六万多行英法语料读进内存并建立可双向查询的词表？",
    chain: ["读取 UTF-8 文件", "按制表符拆句对", "抽查样本", "建立英法词到索引", "生成反向词表"],
    extraVisuals: [dataUml, fullPipeline],
    article: article([
      ["0:00–4:46", "从文件读取全部行，但先确认读进来的数据长什么样", `老师先把本节定位成“加载数据到内存”，因为后面封装 Dataset 和 DataLoader 都依赖这里返回的结构。函数用 UTF-8 打开英法平行语料，通过 \`readlines()\` 一次取得全部行。课程数据有六万多行，直接打印会把终端淹没，因此老师只截取前五行检查。

这一步看似只是文件 I/O，却承担了第一道验收：文件路径是否正确、编码是否能解析、行数是否符合预期，以及每一行是否仍保留英法两部分。老师反复打印不是为了凑代码，而是示范“在做下一层抽象前先看真实数据”。如果读取阶段已经出现乱码或空行，后面再漂亮的词表也没有意义。`],
      ["4:46–11:28", "按制表符拆开英法句子，并对两边分别调用清洗函数", `原始语料的一行由英文、制表符和法文组成。老师用两层列表推导式处理：外层遍历每一行，内层用制表符切分，再把切出的每个句子交给前一节编写的规范化函数。最终 \`pairs\` 是一个列表，列表里的每个元素长度为 2，索引 0 是英文，索引 1 是法文。

课堂没有把整行当成一个字符串继续处理，因为翻译监督关系必须明确保留。切分后老师用前五组样本逐项说明嵌套结构：外层决定第几条句对，内层两个元素决定源语言和目标语言。这个结构约定会贯穿 Dataset 的 \`__getitem__\`，如果在这里把两列顺序弄反，后面编码器和解码器的词表也会跟着反。

清洗必须对句对两端都执行，但不能打乱配对。某行列数异常或清洗后为空时，应整对处理并记录；本节课堂主要演示正常路径，生产数据还需要额外校验。`],
      ["11:28–15:12", "抽查不同位置，而不是看完前五行就假设全体正常", `得到 \`pairs\` 后，老师专门安排“数据探查”：先看前几条，再选择第 7000 条一类中间位置，分别打印英文 \`pairs[i][0]\` 和法文 \`pairs[i][1]\`。7000 没有特殊业务含义，它只是提醒读者不要只观察文件开头。

这个动作隔离了两类问题：一是结构问题，例如某一行没有被正确切成两列；二是内容问题，例如英文和法文看起来不再互为翻译。抽样不能替代完整统计，但能在写词表前快速发现明显错误。老师的做法也说明调试输出应当少而有目的，六万行全部打印反而不利于判断。`],
      ["15:12–19:07", "分别初始化英文和法文词表，先固定 SOS 与 EOS", `课程为两种语言分别建立“单词到索引”的字典。英文和法文的词集合不同，不能共用一张映射表。老师先放入两个特殊符号：SOS 表示句子开始，索引为 0；EOS 表示句子结束，索引为 1，因此两个词表的计数器都从 2 开始。

老师同时演示了“直接写 0/1”和“引用 \`SOS_token\`、\`EOS_token\` 常量”两种写法，并推荐实际开发使用常量，避免同一个语义数字散落在代码里。这里还没有实现 PAD 或 UNK，不能把后续可能需要的特殊符号提前说成本节已经完成；本节真正落地的是 SOS、EOS 和两套独立词表。`],
      ["19:07–25:55", "遍历所有句对，只给第一次出现的词分配新索引", `构建词表时，外层遍历全部双语句对；对英文句子按空格切成词，再逐词检查是否已在英文字典中。前一节的清洗函数已把标点两侧留出空格，因此 \`you !\` 会稳定拆成两个 token，而不会一会儿得到 \`you!\`、一会儿得到 \`you\`。

若单词不存在，就用当前字典长度或计数器作为新索引，然后把计数器加一；若已经存在，则保持原索引不变。老师专门解释为什么必须先判断：同一个词会在语料中反复出现，若每次都重新编号，先前编码的句子就会失效。法语词表重复相同流程，但读取句对的索引 1。

完成后课堂打印出英文约 2803 个词、法文约 4345 个词。数字属于当前清洗和语料版本，不应当背诵；真正需要记住的是，两边独立统计且索引一经分配必须稳定。`],
      ["25:55–31:46", "建立索引到单词的反向表，并一次返回后续需要的全部结构", `模型训练使用数字 ID，但调试预测结果时还要把 ID 还原成单词，所以老师用字典推导式把“词 → ID”反转为“ID → 词”。英文和法文各有正向、反向两张表，再加上各自词表大小，组成后续 Dataset、模型输出层和预测解码都会使用的返回值。

老师最后把函数调用结果一次解包，并重新打印词表大小做收口检查。到这里完成的是“文件 → 清洗句对 → 两套双向词表”的内存准备，还没有把每句话真正编码成张量，也没有进行 padding。下一节封装 Dataset 时才会消费这些结构，因此本节的完成标准是：句对方向正确、特殊符号索引固定、正反向映射互相一致、词表大小能复现。`],
    ]),
    points: ["每条样本始终保持英文与法文配对", "src/tgt 词表独立且 SOS/EOS 索引固定", "正向表用于编码，反向表用于还原预测"],
    code: `pairs = [["i am here .", "je suis ici ."]]\nen_word2index = {"SOS": 0, "EOS": 1}\nfor word in pairs[0][0].split():\n    if word not in en_word2index:\n        en_word2index[word] = len(en_word2index)\nen_index2word = {index: word for word, index in en_word2index.items()}\nprint(en_word2index)\nprint(en_index2word)`,
    output: "同一个英文词只分配一次索引，并能通过反向表还原。",
    pit: "不要把本节说成已经完成 PAD、UNK、张量编码或过滤；课堂此处只建立双语句对和 SOS/EOS 词表。",
    quiz: "为什么英文和法文必须分别建立词表？", answer: "两种语言的词集合和词表大小不同；共用索引会混淆编码器输入与解码器输出空间。"
  },
  {
    p: 86, slug: "dataset", title: "构建 Dataset：逐词查表、追加 EOS 并转成张量",
    problem: "前一节已经得到句对和双向词表，Dataset 的 __getitem__ 怎样把其中一对英文、法文转换成模型可用的 ID 张量？",
    chain: ["初始化时取得句对与词表", "__len__ 返回句对数", "按 i 取同一对句子", "逐词查各自词表", "追加 EOS 并转 long 张量"],
    extraVisuals: [dataUml],
    article: article([
      ["0:00–4:12", "先把数据处理链条接起来，而不是重新读一遍文件", `老师先回顾前几节的顺序：定义设备和 SOS/EOS 常量，编写清洗函数，再由数据预处理函数读取句对并建立英文、法文词表。Dataset 的初始化方法直接调用这套预处理并保存返回结果，因此后面的取样逻辑始终使用同一份句对和同一套索引。

这里的关键不是 TensorDataset 这个名字，而是让 Dataset 成为“按编号提供一对训练样本”的统一入口。英文和法文必须共用同一个样本索引，不能各自随机抽取，否则监督关系会被破坏。`],
      ["4:12–8:36", "__len__ 返回句对数量，__getitem__ 先取出同一对文本", `\`__len__\` 返回双语句对列表的长度，让 DataLoader 知道数据集共有多少条。\`__getitem__(index)\` 用同一个 index 取得英文句子和法文句子，再分别按空格切词。

老师在这里沿用前一节约定：句对索引 0 是英文，索引 1 是法文。这个方向一旦交换，英文词就会去法语字典里查，轻则 KeyError，重则在错误词表上得到没有语义的编号。`],
      ["8:36–12:43", "英文查英文表、法文查法文表，得到两串 token ID", `取出的每个英文词通过英文 \`word2index\` 查编号，每个法文词通过法文 \`word2index\` 查编号。老师把列表推导式拆开解释，是为了让同学看清“词 → 字典 → 整数”的路径，而不是把字符串直接交给神经网络。

本课程词表由同一批语料预先建立，所以课堂代码直接索引字典；它此处没有定义 UNK 的回退策略。若扩展到语料外输入，需要另行增加未知词方案，不能假装本节已经实现。`],
      ["12:43–16:37", "在两句话末尾追加 EOS，再按当前设备创建 long 张量", `英文 ID 列表和法文 ID 列表都要追加 EOS。老师强调不能把字符串“EOS”混进整数列表，而应追加前面定义的 \`EOS_token\`（课堂中值为 1）。这样模型以后能够知道一句话在什么位置结束。

最后用 \`torch.tensor(..., dtype=torch.long, device=device)\` 创建张量并成对返回。long 类型是 Embedding 索引的要求，device 参数则保证样本与模型位于同一计算设备。此时每个样本仍可有不同长度；课堂下一节使用 batch_size=1，因此没有在这里做 padding。`],
    ]),
    points: ["同一个 index 必须返回一对互译句子", "英文和法文分别查各自词表", "两端都在末尾追加 EOS", "Embedding 的索引张量使用 torch.long"],
    code: `import torch\nEOS_token = 1\nen_word2index = {"i": 2, "am": 3}\nfr_word2index = {"je": 2, "suis": 3}\nx = [en_word2index[word] for word in "i am".split()] + [EOS_token]\ny = [fr_word2index[word] for word in "je suis".split()] + [EOS_token]\nprint(torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long))`,
    output: "英文和法文分别得到整数 ID 序列，末尾的 1 是 EOS。",
    pit: "不要把字符串 EOS 直接追加到整数列表；也不要用法语词表编码英文。",
    quiz: "为什么本节的英文、法文列表末尾都要追加 EOS？", answer: "让后续编码和解码过程都能识别各自句子的结束位置。"
  },
  {
    p: 87, slug: "dataloader", title: "构建 DataLoader：用 batch_size=1 检查英法 ID 序列",
    codeLabel: "课堂调用骨架（需配合上一节 Dataset）",
    problem: "Dataset 写好后，课堂怎样用最小配置随机取一条句对，并确认英文、法文张量的形状与内容？",
    chain: ["实例化 Dataset", "DataLoader 设 batch_size=1", "开启 shuffle", "遍历 X/Y", "打印形状和 ID 后 break"],
    extraVisuals: [dataUml],
    article: article([
      ["0:00–2:33", "用 Dataset 创建 DataLoader，课堂把一批固定为一条句对", `老师先实例化上一节的自定义 Dataset，再交给 PyTorch DataLoader。课堂把 \`batch_size\` 设为 1，因为英法句子长度不同，而当前案例没有编写批量补齐函数；一批只有一条时，默认 collate 不会遇到多条变长张量无法堆叠的问题。

\`shuffle=True\` 让每次遍历取得的第一条样本不一定相同，所以老师提醒“第一批”不等于数据文件中的第一行。这里只是在训练数据上演示随机取样，并未讨论验证集的 shuffle 策略。`],
      ["2:33–5:54", "遍历得到 X 和 Y：外层 batch 维为 1，内层长度随句子变化", `遍历 DataLoader 时，每次得到英文张量 X 和法文张量 Y。Dataset 已经把词换成索引并追加 EOS，因此打印出的不是原始单词，而是一串整数。

例如 X 的形状可能是 \`[1, 6]\`，Y 可能是 \`[1, 7]\`：第一个 1 是 batch_size，后面的 6、7 是两句话各自的 token 数。源语言与目标语言长度不必相同，这正是 Seq2Seq 要处理的情况。`],
      ["5:54–8:43", "只看一批完成验收，然后返回 DataLoader 给后续模型复用", `老师同时打印形状和实际 ID，确认取样方向、EOS 以及设备都没有问题。为了避免把六万多条样本全部刷到控制台，测试循环末尾立即 \`break\`。

函数最后返回 DataLoader，后续 Encoder、Decoder 测试和训练都会调用它。需要明确的是：本节课堂没有实现 PAD、mask 或自定义 collate_fn；若要把 batch_size 提高到 2 以上，才需要另加变长批处理方案。这是扩展要求，不应写成老师本节已经完成的步骤。`],
    ]),
    points: ["课堂 batch_size=1", "shuffle 后第一批不等于文件第一行", "X/Y 第一维都是 batch，第二维可以不同", "break 只用于本节检查"],
    code: `from torch.utils.data import DataLoader\nloader = DataLoader(dataset, batch_size=1, shuffle=True)\nfor x, y in loader:\n    print("English", x.shape, x)\n    print("French ", y.shape, y)\n    break`,
    output: "打印一对经过索引化的英法张量；两边序列长度可以不同。",
    pit: "当前默认组批只因 batch_size=1 才能直接处理变长样本；把 batch_size 调大却不写 collate_fn 会报堆叠尺寸不一致。",
    quiz: "为什么老师只遍历一批就 break？", answer: "这里只验收 DataLoader 返回的形状和内容，继续打印全部语料没有调试价值。"
  },
  {
    p: 88, slug: "gru-encoder", title: "GRU Encoder：Embedding 后保留每个时间步输出",
    problem: "课程版 Encoder 怎样用 Embedding、batch_first GRU 和显式初始 hidden 把英文 ID 变成所有时间步输出？",
    chain: ["source IDs [B,S]", "Embedding [B,S,E]", "GRU", "outputs [B,S,H]", "hidden [1,B,H]"],
    extraVisuals: [encoderShapes, architecture],
    article: article([
      ["0:00–5:39", "Seq2Seq 不是一个模型类就结束，而是先后构建 Encoder 与两版 Decoder", `老师先说明本章至少要写 Encoder、无 Attention Decoder、有 Attention Decoder，以及各自测试。当前只处理左侧 Encoder：它读英文序列，生成交给后续 Decoder 的状态。

图中橙色部分是张量，蓝色部分是网络层。当前输入不是原始单词，而是英文 token ID；Embedding 先把 ID 转成词向量，再由 GRU 把上一步 hidden 与当前表示结合。`],
      ["5:39–10:57", "初始化只接 input_size 与 hidden_size，但两者语义不同", `\`input_size\` 在本节表示英文词表大小，当前数据约为 2803；\`hidden_size\` 是每个词的表示和 GRU 隐藏维，课程设为 256。老师提醒“层数”和“维度”不是同一概念：本例 GRU 层数保持默认 1，隐藏维才是 256。

Embedding 因此定义为 \`Embedding(input_size, hidden_size)\`，把每个英文 ID 映射成 256 维。`],
      ["10:57–14:58", "GRU 输入维和隐藏维都是 256，并显式使用 batch_first=True", `Embedding 输出最后一维为 256，所以 GRU 的 input_size 与 hidden_size 都写 256。课程设置 \`batch_first=True\`，张量顺序统一为 \`[batch, sequence, feature]\`；若不设置，PyTorch 默认前两维是 sequence、batch，整章的形状解释都会交换。

例如一批一条、句长 6，输入 ID 是 \`[1,6]\`，Embedding 后是 \`[1,6,256]\`。`],
      ["14:58–20:45", "forward 返回所有时间步 outputs 与最终 hidden", `forward 接收英文输入和初始 hidden。输入先过 Embedding，再与 hidden 一起送 GRU，得到 outputs 与新的 hidden。若源句长度为 6，outputs 是 \`[1,6,256]\`；hidden 是 \`[1,1,256]\`，三维分别是层数、batch、隐藏维。

两项都要返回：无 Attention Decoder 可以主要使用最后 hidden；Attention Decoder 还需要 outputs 中每个英文位置的状态。`],
      ["20:45–23:36", "initHidden 创建 [1,1,256] 零张量并迁移到同一设备", `老师单独编写初始化隐藏状态的方法，按“层数 1、batch 1、hidden_size”创建零张量，并放到全局 device。虽然 PyTorch GRU 在省略 h0 时会自动使用零状态，课堂显式创建是为了让状态来源和形状可见。

到这里 Encoder 类完成；下一节必须用真实 DataLoader 样本核验源序列长度是否在 outputs 中原样保留。`],
    ]),
    points: ["input_size 是英文词表大小", "hidden_size=Embedding 维=GRU 隐藏维", "课程使用 batch_first=True", "outputs 保留每个源位置", "hidden 形状是 [层数,B,H]"],
    code: `import torch\nfrom seq2seq_from_scratch.model import EncoderGRU\nm=EncoderGRU(100,16,32)\nout,h=m(torch.randint(0,100,(4,7)))\nprint(out.shape,h.shape)`,
    output: "outputs=[4,7,32]，hidden=[1,4,32]。",
    pit: "不要把 input_size 当成词向量维，也不要漏写 batch_first=True 后仍按 [S,B,H] 解读输出。",
    quiz: "B=4、S=7、H=32 时 outputs 形状？", answer: "[4,7,32]。"
  },
  {
    p: 89, slug: "test-encoder", title: "测试 Encoder：先验形状再运行",
    problem: "如何用 DataLoader 的真实英文样本验证 EOS、outputs 和 hidden，并解释一次测试中句长看似不一致的原因？",
    chain: ["造 source IDs", "实例化 Encoder", "前向", "检查 outputs", "检查 hidden/梯度"],
    extraVisuals: [encoderShapes],
    article: article([
      ["0:00–3:48", "先取得 DataLoader，再按 2803 和 256 创建 Encoder", `老师调用上一节函数取得 DataLoader，设置英文词表大小约 2803、hidden_size=256，并实例化 Encoder。若使用 GPU，还要把模型移动到与数据相同的 device；CPU 环境不需要额外迁移。

这里不使用随机造出的 ID，而是直接取课程 Dataset 产出的真实英文张量，能同时验证数据管道与模型接口。`],
      ["3:48–7:39", "打印 X 的形状和内容，确认最后一个 ID 始终是 EOS=1", `遍历 DataLoader 取得 X/Y 后，老师先打印英文 X。不同句子长度可能是 6、7、9 等，但列表最后一个编号都应为 1，因为 Dataset 已在英文末尾追加 EOS。

shuffle=True 使每次第一条样本不同，所以不应背某一串 ID；应检查的是 batch 维为 1、dtype 与 device 正确、结尾 EOS 存在。`],
      ["7:39–10:37", "创建 h0 并执行前向，先预测两个返回形状", `调用 Encoder 的 initHidden 得到 \`[1,1,256]\`，再把 X 与 h0 送入模型。若 X 是 \`[1,S]\`，按 batch_first 约定，outputs 应为 \`[1,S,256]\`；hidden 始终为 \`[1,1,256]\`。

outputs 的 S 随句长变化，hidden 的层数、batch 与隐藏维在本例固定。`],
      ["10:37–14:25", "为何一次日志里 X 长 6，outputs 却显示长 7", `老师现场遇到看似矛盾的打印：上方 DataLoader 预览函数内部先随机取了一条并打印，测试函数随后又遍历同一个 shuffle DataLoader，实际拿到的是另一条。两次随机样本的句长不同，却被误当成同一条比较。

这不是 GRU 改变了序列长度，而是调试代码重复消费 DataLoader。老师停下来追踪调用路径，说明排错不能只看相邻日志，还要确认日志是否来自同一批数据。`],
      ["14:25–17:03", "去掉内部预览后，outputs 的序列长度必须与当前 X 完全一致", `注释掉 DataLoader 函数中额外的预览遍历后，当前测试只消费一次样本。此时 X 若为 \`[1,9]\`，outputs 就是 \`[1,9,256]\`；X 若为 \`[1,5]\`，outputs 就是 \`[1,5,256]\`。

老师以此完成 Encoder 验收：前向能运行，outputs 保留输入序列长度，hidden 为 \`[1,1,256]\`。这里只证明接口正确，不代表未训练状态已经包含可用翻译语义。`],
    ]),
    points: ["真实 X 末尾应为 EOS=1", "outputs 的 S 必须等于当前 X 的 S", "hidden 固定 [1,1,256]", "重复遍历 shuffle DataLoader 会拿到不同样本"],
    code: `import torch\nfrom seq2seq_from_scratch.model import EncoderGRU\nx=torch.randint(0,50,(2,5)); out,h=EncoderGRU(50,8,12)(x)\nassert out.shape==(2,5,12) and h.shape==(1,2,12)\nprint("ok")`,
    output: "断言通过后打印 ok。",
    pit: "不要拿 DataLoader 内部预览的 X 与测试循环下一次随机样本的 outputs 比较；它们可能不是同一条句子。",
    quiz: "当前 X 是 [1,8] 时，课程 Encoder 的 outputs 和 hidden 应是什么形状？", answer: "outputs=[1,8,256]，hidden=[1,1,256]。"
  },
  {
    p: 90, slug: "plain-decoder-plan", title: "无 Attention Decoder 思路：只靠 final hidden 生成",
    problem: "无注意力版本怎样工作，它为什么是理解有注意力版本的对照组？",
    chain: ["输入上一个目标 token", "目标 Embedding", "与 hidden 送 GRU", "Linear→目标词表", "预测下一 token"],
    extraVisuals: [architecture],
    article: article([
      ["0:00–1:40", "先沿结构图区分张量和网络层", `老师先标出当前输入 token、上一 hidden、本次 GRU output 与新 hidden。当前输入是一个法语词 ID，先经 Embedding 变成 \`[1,1,256]\`；ReLU 只改变数值，不改变形状。`],
      ["1:40–3:38", "GRU 用当前词表示和上一 hidden 得到本步状态", `上一 hidden 形状同样是 \`[1,1,256]\`，最初可直接来自 Encoder final hidden。二者送入 GRU 后，本步 output 和新 hidden 仍保持 256 维。这里就是无 Attention 版本接收源句信息的唯一通道。`],
      ["3:38–5:19", "输出层把 256 维映射到 4345 个法语候选", `GRU output 经过 Linear 映射成约 4345 个法语词的分数，再经 LogSoftmax 成为对数概率。概率最高的候选就是当前预测词。

老师本节主要讲形状流，下一节才写代码。固定一个 final hidden 承担全部源句信息，是无 Attention 版本与后续逐步重算权重版本的主要差别。`],
    ]),
    points: ["当前输入与 hidden 都整理为 [1,1,256]", "GRU output/hidden 仍是 256 维", "输出维等于法语词表 4345", "无 Attention 只通过 final hidden 接收源句信息"],
    code: `print("token + hidden → GRU → Linear(target_vocab)")`,
    output: "显示无注意力单步的最短数据流。",
    pit: "目标 Embedding 的词表大小必须用法语词表，而非英语词表。",
    quiz: "无注意力 Decoder 从哪里得到源句信息？", answer: "Encoder 的最终 hidden。"
  },
  {
    p: 91, slug: "plain-decoder-code", title: "构建无 Attention GRU Decoder：LogSoftmax 必须配 NLLLoss",
    problem: "课堂怎样把一个目标词 ID 和上一隐藏状态变成 4345 个法语词的对数概率，并正确选择损失函数？",
    chain: ["Embedding 当前词", "ReLU", "GRU 更新状态", "Linear 映射到法语词表", "LogSoftmax 后配 NLLLoss"],
    extraVisuals: [architecture],
    article: article([
      ["0:00–6:53", "从结构图确定三层尺寸：4345→256→256→4345", `老师先用图核对 Decoder 的职责。目标词来自法语词表，所以 Embedding 是 \`Embedding(4345, 256)\`；GRU 的输入维和隐藏维都是 256；Linear 再把 256 维隐藏状态映射回 4345 个法语候选。

4345 不是词向量维度，而是当前语料的法语词表大小。每一步输出 4345 个值，后面才能从中选择最可能的法语词。Encoder 不需要词表分类层，是因为它的状态还要继续交给 Decoder；Decoder 才负责目标词分类。`],
      ["6:53–13:58", "隐藏状态先更新，再由 Linear 得到当前词的候选分布", `前向传播接收当前目标词和上一步 hidden。当前词先经过 Embedding，再经 ReLU，随后与 hidden 一起送入 GRU。GRU 同时返回本步 output 和更新后的 hidden；课程代码再取合适的二维视图交给 Linear，得到 \`[1, 4345]\`。

老师借 RNN 公式再次说明顺序：当前输入和旧状态先产生新状态，再依据新状态判断当前输出词。形状变换必须保留 batch 语义，不能使用不带维度的 \`squeeze()\` 把 batch=1 一并删掉。`],
      ["13:58–20:29", "课程代码返回 LogSoftmax，训练时必须使用 NLLLoss", `本节实际在输出层之后调用 \`LogSoftmax(dim=-1)\`，因此返回的是每个法语词的对数概率，而不是未经归一化的 logits。老师明确提醒：既然模型里已经做了 LogSoftmax，后面的损失就应使用 \`NLLLoss\`；若再使用会内部执行 LogSoftmax 的 \`CrossEntropyLoss\`，就会重复处理。

课程配套的现代化示例代码采用另一种等价约定：模型直接返回 logits，损失使用 CrossEntropyLoss。两种写法都成立，但必须成对出现，不能把“LogSoftmax + CrossEntropyLoss”混搭。本节正文以老师实际实现“LogSoftmax + NLLLoss”为准。`],
    ]),
    points: ["目标 Embedding 和输出层都使用法语词表大小", "GRU output 与 hidden 含义不同", "课堂返回 log-probabilities", "LogSoftmax 配 NLLLoss；raw logits 配 CrossEntropyLoss"],
    code: `import torch\nemb = torch.nn.Embedding(120, 16)\ngru = torch.nn.GRU(16, 32, batch_first=True)\nfc = torch.nn.Linear(32, 120)\nlog_softmax = torch.nn.LogSoftmax(dim=-1)\nx = torch.relu(emb(torch.tensor([1])).unsqueeze(1))\nout, hidden = gru(x)\nlog_probs = log_softmax(fc(out.squeeze(1)))\nprint(log_probs.shape, torch.exp(log_probs).sum(-1))`,
    output: "得到 [1,120] 的对数概率；取 exp 后 120 个候选的概率和为 1。",
    pit: "模型已经输出 LogSoftmax 时不要再配 CrossEntropyLoss；若想用 CrossEntropyLoss，就删除模型内的 LogSoftmax 并返回 logits。",
    quiz: "老师本节的 Decoder 已调用 LogSoftmax，训练时应配什么损失？", answer: "NLLLoss；CrossEntropyLoss 适用于模型返回未经 LogSoftmax 的 logits。"
  },
  {
    p: 92, slug: "test-plain-decoder", title: "测试无 Attention Decoder：连接编码器并逐步验形状",
    codeLabel: "课堂接口伪代码（需配合完整 Encoder/Decoder）",
    problem: "无 Attention Decoder 单独写完后，怎样把它接到 Encoder 上，并用一条真实句对确认每一步的数据流和输出维度都正确？",
    chain: ["加载一组英法句对", "英文送入 Encoder", "取最后时间步状态", "逐个读取真实法语 token", "检查每步 4345 维输出"],
    extraVisuals: [architecture],
    article: article([
      ["0:00–5:00", "测试 Decoder 之前仍要先创建 Encoder", `老师先取得 DataLoader，再分别实例化 Encoder 和无 Attention Decoder，并把两者移动到同一设备。虽然本节标题是“测试 Decoder”，但 Decoder 需要由 Encoder 产生的中间语义状态；只创建右半边模型，测试链路并不完整。

这里也顺手纠正词表方向：英译法时，Encoder 的输入词表是英文，课堂数据大小约为 2803；Decoder 的输出词表是法文，约为 4345；两边隐藏维都设为 256。老师把 \`.to(device)\` 直接接在构造结果后面，是为了避免模型和输入张量分处 CPU、GPU。具体词表数字来自当前语料，不是通用常量。`],
      ["5:00–14:51", "从打印出的层结构读懂 2803、4345 和 256", `老师没有急着喂数据，而是先打印两套模型。Encoder 的 Embedding 把 2803 个英文词映射为 256 维词向量，GRU 接收 256 维输入并维持 256 维隐藏状态；这里没有额外分类输出层，因为编码结果还要继续传给 Decoder，而不是在源语言侧直接做类别预测。

Decoder 的 Embedding 面向 4345 个法语词，GRU 仍使用 256 维状态，最后的 Linear 再把当前隐藏状态映射为 4345 个分数。老师借 RNN 图重新解释：当前输入和上一步状态先得到当前隐藏状态，再由当前隐藏状态推目标词。因而 \`Linear(256, 4345)\` 的含义不是“输出一个词向量”，而是给法语词表中的每个候选词一个预测分数。`],
      ["14:51–21:51", "从 DataLoader 取一条配对样本，先完成英文编码", `模型结构确认后，老师从 DataLoader 中取一批数据，并用 \`break\` 只观察一组英文、法文样本。调试时先打印形状而不是整批内容：例如英文长度为 6、法文长度为 7，说明它们是一条配对样本，但源句和目标句的词数不必相同。

随后英文张量进入 Encoder。课堂实现直接返回 GRU 处理后的各时间步结果，因此一条含 6 个 token 的句子会得到近似 \`[1, 6, 256]\` 的序列表示；老师讲解时常省略 batch=1，写成 \`[6, 256]\`。每个源位置都有一个 256 维状态，越靠后的状态已经累积了更多前文信息。这一步验证的是“英文词 ID → 编码序列”能够跑通。`],
      ["21:51–25:48", "只取最后一个源时间步，作为无 Attention Decoder 的初始状态", `无 Attention 版本不会在每个目标步回看全部 Encoder outputs，所以老师从编码序列中取最后一个时间步的 256 维状态，作为 Decoder 的起始隐藏状态。索引操作的语义是：保留 batch 中的第一个样本，再取源序列最后一个位置，而不是随便从二维数组里拿最后一行。

这正是固定语义向量 C 的瓶颈：前面 6 个英文 token 的信息都要压到最后状态里。短例子可以用来验证接口，句子变长后早期信息可能被稀释，这也是后续加入 Attention 的动机。此处的完成标准只是状态形状能与 Decoder 的隐藏维匹配，并不代表翻译质量已经可靠。`],
      ["25:48–33:41", "课堂循环读取真实法语 token，而不是让模型自由生成", `老师接着遍历目标句长度 \`y.shape[1]\`，每次从真实法语张量中取出第 i 个 token ID。\`y[0, i]\` 原本是一个标量，再用 \`.view(1, -1)\` 整理成 Decoder 当前实现所要求的输入形状。老师用 123、297、126 等假想 ID 解释：循环每次拿到一个真实词编号，改好形状后送入同一个 Decoder step。

这一点必须与真正预测区分开：本节没有从 SOS 开始做 argmax，也没有把模型上一步的预测回送，更没有靠 EOS 终止。老师明确按真实目标句的词数运行，目的是让每个位置都能经过 Decoder，观察模块是否连通。这种写法更接近“带真实输入的接口测试”，不能被表述成完整推理算法。`],
      ["33:42–39:36", "每个目标位置输出一份 4345 维分数，跑通不等于会翻译", `每一步把当前真实目标 token 和前一步隐藏状态交给 Decoder，得到新的隐藏状态以及法语词表上的输出。课堂打印的每个结果形状都是 \`[1, 4345]\`：batch 中这一个样本，在当前法语位置上对 4345 个候选词各给一个分数。目标句有 8 个位置，就会打印 8 份这样的分布。

老师说明，训练完成后才可以从这些分数中选最大值、映射回法语词并拼成句子；当前模型还没有训练，直接取最大值只能得到没有意义的结果。本节真正证明的是 Encoder 与无 Attention Decoder 的尺寸、设备和调用接口能贯通。后续训练会用这些输出计算损失，而真正无真值预测则要另写以模型输出驱动下一步的循环。`],
    ]),
    points: ["测试 Decoder 仍需 Encoder 提供初始状态", "课堂按真实目标长度逐词喂入 token，只做接口验形状", "每步输出 [1,4345]，代表法语词表候选分数", "未训练时跑通不等于已经会翻译"],
    code: `# 与 P91 的 batch_first=True 形状约定保持一致\nencoder_outputs, encoder_hidden = encoder(source)  # [B,S,H], [1,B,H]\ndecoder_hidden = encoder_outputs[:, -1, :].unsqueeze(0)  # [1,B,H]\nfor i in range(target.shape[1]):\n    decoder_input = target[:, i]                    # [B]\n    output, decoder_hidden = decoder(decoder_input, decoder_hidden)\n    print(i, output.shape)                           # [B,4345]`,
    output: "目标句有多少个时间步，就打印多少个 [1,4345] 的候选词分数张量。",
    pit: "不要把这段接口测试误写成自由推理；还要注意 GRU hidden 是 [层数,B,H]，不能在 B>1 时把它写成 [B,1,H]。",
    quiz: "为什么模型还没训练，课堂仍然要逐步打印 [1,4345]？", answer: "为了确认英文编码结果能初始化 Decoder、每个真实目标 token 的输入形状正确，而且输出维确实覆盖整个法语词表；这只能验接口，不能证明翻译正确。"
  },
  {
    p: 93, slug: "attention-decoder-plan", title: "课程版 Attention Decoder：拼接查询与隐藏状态计算 10 个权重",
    problem: "老师采用的课程版 Attention Decoder 怎样把当前输入、上一隐藏状态和固定长度的 Encoder outputs 组合起来？",
    chain: ["当前词 Embedding 作查询", "与 hidden 拼成 512 维", "Linear 映射为长度 10", "Softmax 得源位置权重", "bmm 汇总 Encoder outputs"],
    extraVisuals: [decoderShapes, architecture],
    article: article([
      ["0:00–5:24", "先沿图识别三路输入，不能把这套实现误说成点积注意力", `老师从结构图出发：当前目标词先经 Embedding 得到 \`[1,1,256]\`，上一时刻 hidden 也是 \`[1,1,256]\`，Encoder 则保留源句所有时间步的状态。当前词表示在图中承担查询 Q 的角色，但课堂实现不是直接计算 QK 点积。

本课程采用 PyTorch 早期 Seq2Seq 教程常见的“拼接后过线性层”方案：把当前词表示和 hidden 拼成 512 维，再用 Linear 一次产生最大源长度个分数。它与前面讲过的注意力三步思想相通，但具体打分公式不同。`],
      ["5:24–10:03", "固定最大句长为 10，所以注意力层输出 10 个位置分数", `课程把英文句子的最大长度预设为 10。拼接后的 \`[1,1,512]\` 经过 \`Linear(512,10)\`，再用 Softmax 得到 \`[1,1,10]\` 的注意力权重。10 表示十个源位置，不是隐藏维度，也不是法语词表大小。

真实英文句子可能只有 6、7 或 8 个 token，测试阶段会先把 Encoder outputs 复制到一个 \`[1,10,256]\` 的零张量里。当前实现没有单独的 PAD mask，所以补出的零位置仍参与 Softmax；这是教学简化，不能把它描述成已经实现了 mask。`],
      ["10:03–15:46", "权重通过 bmm 汇总 Encoder outputs，得到本步上下文", `\`[1,1,10]\` 的权重与 \`[1,10,256]\` 的 Encoder outputs 做批量矩阵乘法，得到 \`[1,1,256]\` 的 attention-applied context。它表示当前生成步骤从十个源位置汇总出的信息。

老师反复让同学对照形状理解：权重的最后一维必须等于固定源长度 10，Encoder outputs 的中间维也必须是 10，二者才能相乘。隐藏维 256 在加权求和后保留下来。`],
      ["15:46–20:25", "再把当前词表示与 context 融合，交给 GRU 和法语分类层", `得到 context 后，课程又把它与当前目标词的 Embedding 拼成 512 维，通过 \`attn_combine\` 线性层降回 256 维，经过 ReLU 再送入 GRU。GRU 更新 hidden，最后 Linear 把 256 维映射到 4345 个法语词。

每个目标时间步都会重新计算一组十维权重，因此生成不同法语词时关注的英文位置可以不同。输出端还会返回这组权重，后面绘制注意力热力图。`],
    ]),
    points: ["课堂使用 concat+Linear 打分，不是点积 QK", "max_length 固定为 10", "权重 [1,1,10] 与 Encoder outputs [1,10,256] 做 bmm", "context 与当前词表示再次拼接后送 GRU"],
    code: `import torch\nB,H,L=1,256,10\nembedded=torch.randn(B,1,H); hidden=torch.randn(1,B,H)\nscore_layer=torch.nn.Linear(H*2,L)\nweights=torch.softmax(score_layer(torch.cat((embedded,hidden.transpose(0,1)),dim=-1)),dim=-1)\nencoder_outputs=torch.randn(B,L,H)\ncontext=torch.bmm(weights,encoder_outputs)\nprint(weights.shape,context.shape)`,
    output: "课程默认尺寸下，weights=[1,1,10]，context=[1,1,256]。",
    pit: "不要把课程实现改写成 hidden 与 Encoder outputs 的点积注意力，也不要声称本节已做 PAD mask。",
    quiz: "为什么注意力层 Linear 的输出维是 10？", answer: "课程把源句最大长度固定为 10，要为这十个源位置各产生一个权重。"
  },
  {
    p: 94, slug: "attention-decoder-code-part1", title: "有 Attention Decoder 代码（上）：初始化九个成员",
    problem: "output_size、hidden_size、dropout_p 和 max_length 分别决定哪些层，为什么会出现两个 512 维线性变换？",
    chain: ["保存四个参数", "Embedding 4345→256", "attn 512→10", "attn_combine 512→256", "GRU/Out/LogSoftmax"],
    extraVisuals: [decoderShapes],
    article: article([
      ["0:00–3:47", "四个构造参数分别控制词表、隐藏维、随机失活和固定句长", `老师定义 output_size、hidden_size、dropout_p、max_length。output_size 是法语词表大小约 4345；hidden_size 与 Encoder 保持 256，保证 Encoder 最后 hidden 能直接作为 Decoder 初始 hidden；dropout_p 控制 Embedding 后的随机失活；max_length 在课程里设为 10。`],
      ["3:47–7:51", "Embedding 与第一个 Linear：把查询和 hidden 映射成十个位置权重", `Embedding 把法语 token 从 4345 类映射到 256 维。当前词表示和上一 hidden 拼接后是 512 维，所以 \`attn\` 定义为 \`Linear(512,10)\`。它输出的是十个源位置的打分，之后才会经过 Softmax 成为权重。

老师特别解释 Linear 可以接三维张量：它只变换最后一维，前面的 batch 与时间维保持不变。因此 \`[1,1,512]\` 可以直接得到 \`[1,1,10]\`。`],
      ["7:51–10:03", "第二个 Linear 把查询与 context 从 512 维融合回 256 维", `注意力权重汇总 Encoder outputs 后得到 256 维 context；它还要与当前词的 256 维 Embedding 再次拼接。因此 \`attn_combine\` 是 \`Linear(512,256)\`，负责把融合结果压回 GRU 所需的隐藏维。

两个 Linear 都从 512 开始，但作用完全不同：第一个产生 10 个位置分数，第二个生成 256 维融合特征。`],
      ["10:03–11:35", "Dropout、GRU、Out 与 LogSoftmax 完成剩余成员", `课程随后创建 Dropout、\`GRU(256,256)\`、\`Linear(256,4345)\` 和 \`LogSoftmax(dim=-1)\`。最终输出是法语词表上的对数概率，因此训练阶段与 NLLLoss 配套。

本节只完成初始化，forward 的三项输入——当前词、hidden、固定长度 Encoder outputs——留到下一节逐步实现。`],
    ]),
    points: ["attn: 2H→max_length", "attn_combine: 2H→H", "GRU 输入仍是 H", "Out: H→法语词表", "LogSoftmax 配 NLLLoss"],
    code: `import torch\nH,L,V=256,10,4345\nattn=torch.nn.Linear(H*2,L)\ncombine=torch.nn.Linear(H*2,H)\nout=torch.nn.Linear(H,V)\nprint(attn,combine,out)`,
    output: "打印课程版 Attention Decoder 三个关键线性层：512→10、512→256、256→4345。",
    pit: "两个 512 维输入的 Linear 用途不同，不能把 attn_combine 误写成注意力打分层。",
    quiz: "为什么课程版 GRU 的 input_size 是 256 而不是 512？", answer: "512 维拼接结果先经 attn_combine 降回 256，再进入 GRU。"
  },
  {
    p: 95, slug: "attention-decoder-code-part2", title: "有 Attention Decoder 代码（下）：逐行完成拼接式 Attention",
    problem: "课堂版 forward 怎样从当前词、上一 hidden 和 [1,10,256] 的 Encoder outputs 得到对数概率、更新状态与权重？",
    chain: ["Embedding+Dropout", "拼 embedded/hidden 算 10 个权重", "bmm 得 context", "拼 embedded/context 并降维", "ReLU→GRU→LogSoftmax"],
    extraVisuals: [decoderShapes],
    article: article([
      ["0:00–4:17", "三个输入的课程形状先对齐", `forward 接收当前时间步 token、上一 hidden 和 Encoder 所有时间步输出。课堂 batch_size=1：token 形状为 \`[1,1]\`，hidden 为 \`[1,1,256]\`，Encoder outputs 已补进固定张量 \`[1,10,256]\`。

当前 token 经 Embedding 和 Dropout 后仍是 \`[1,1,256]\`。这份表示与 hidden 是注意力打分的两个输入，不是拿 hidden 与每个 Encoder output 做点积。`],
      ["4:17–8:24", "拼 embedded 与 hidden，经 attn 和 Softmax 得 [1,1,10]", `老师用 \`torch.cat((embedded, hidden), dim=-1)\` 得到 \`[1,1,512]\`，交给上一节的 \`attn: Linear(512,10)\`，再沿最后一维做 Softmax。结果 \`attn_weights\` 为十个固定源位置分配概率，总和为 1。

这里没有显式 mask。长度不足 10 的 Encoder outputs 后部是零向量，但相应分数仍可能分到概率；这是课程实现的简化边界。`],
      ["8:24–11:46", "bmm 用十个权重汇总 Encoder outputs", `\`attn_weights [1,1,10]\` 与 \`encoder_outputs [1,10,256]\` 做 \`torch.bmm\`，得到 \`attn_applied [1,1,256]\`。十个源位置在这一维被加权求和，只保留 256 维上下文。

老师把这一步称为“根据注意力权重计算上下文”，并用形状相乘说明为什么固定长度维必须一致。`],
      ["11:46–14:28", "再次拼接 embedded/context，再由 attn_combine 降回隐藏维", `当前词表示和 context 各为 256 维，拼起来是 \`[1,1,512]\`。\`attn_combine\` 把最后一维从 512 降到 256，再经过 ReLU。老师在这里演示 unsqueeze、squeeze 与 Linear 对末维的作用，目的是让张量符合 GRU 输入。`],
      ["14:28–16:02", "GRU 更新 hidden，Out+LogSoftmax 返回三项结果", `融合后的 256 维表示进入 GRU，得到本步 output 和新 hidden。output 经过适当的维度整理，再由 Linear 映射到 4345 个法语词，最后 LogSoftmax 得到 \`[1,4345]\` 对数概率。

函数返回 output、hidden、attn_weights。第三项并非 Decoder 计算必须的额外状态，而是为了后续可视化每个法语词对应的英语位置权重。`],
    ]),
    points: ["embedded 与 hidden 先拼接打分", "Softmax 输出固定十个源位置权重", "bmm 汇总 Encoder outputs", "embedded/context 再拼接融合", "返回 log-probabilities、hidden、weights"],
    code: `import torch\nB,H,L,V=1,256,10,4345\nembedded=torch.randn(B,1,H); hidden=torch.randn(1,B,H); enc=torch.randn(B,L,H)\nattn=torch.nn.Linear(H*2,L); combine=torch.nn.Linear(H*2,H); classifier=torch.nn.Linear(H,V)\nw=torch.softmax(attn(torch.cat((embedded,hidden.transpose(0,1)),-1)),-1)\ncontext=torch.bmm(w,enc)\nfused=torch.relu(combine(torch.cat((embedded,context),-1)))\nlog_probs=torch.log_softmax(classifier(fused.squeeze(1)),-1)\nprint(w.shape,context.shape,log_probs.shape)`,
    output: "weights=[1,1,10]，context=[1,1,256]，log_probs=[1,4345]。",
    pit: "课程代码没有 source mask，不要在整理稿中写成‘已屏蔽 PAD’；这只能作为后续改进建议。",
    quiz: "attn 和 attn_combine 都接 512 维，它们分别输出什么？", answer: "attn 输出 10 个源位置分数；attn_combine 输出 256 维融合表示。"
  },
  {
    p: 96, slug: "test-attention-decoder", title: "测试 Attention Decoder：补成固定 10 步并逐个喂真实目标词",
    problem: "怎样把变长 Encoder outputs 放进课程要求的 [1,10,256] 张量，再验证 Decoder 三个返回值的形状？",
    chain: ["取一对 X/Y", "Encoder 得 outputs/hidden", "零张量补到 10 个源位置", "遍历真实 Y token", "检查 output/hidden/weights"],
    extraVisuals: [decoderShapes],
    article: article([
      ["0:00–6:58", "实例化 Encoder 与 Attention Decoder，先核对词表和隐藏维", `老师仍先调用 DataLoader，再创建英文 Encoder 和法语 Attention Decoder。英文词表约 2803，法语词表约 4345，隐藏维都是 256，Decoder 额外接收 dropout 和 max_length=10。两套模型必须位于同一 device。

打印结构只是为了核对层尺寸；真正测试还需要从 DataLoader 取一对 X/Y 并走完整编码、解码调用。`],
      ["6:58–12:41", "Encoder 返回真实长度 outputs 和 final hidden", `英文 X 进入 Encoder，得到所有时间步 output 和最后 hidden。例如源句实际有 6 个 token，outputs 形状是 \`[1,6,256]\`，hidden 是 \`[1,1,256]\`。老师逐项解释：outputs 保留六个源位置，hidden 汇总到最后时间步。

Decoder 初始 hidden 直接使用 Encoder hidden，因为两者隐藏维一致。`],
      ["12:41–18:37", "创建 [1,10,256] 零张量，只把真实 Encoder outputs 复制到前面", `课程版注意力层固定输出十个权重，所以老师先创建 \`[1,10,256]\` 的零张量，再根据真实源长把 \`outputs[:, :source_len]\` 复制进去。若源长为 6，前六个位置保存真实编码状态，后四个位置保持零。

这叫固定长度存储，不等于已经实现带 mask 的 padding。后四个零位置仍参与注意力打分，这是课程演示方案的限制。`],
      ["18:37–23:46", "按真实法语长度逐词调用 Decoder", `测试循环遍历 Y 的真实长度，每次取 \`y[:, i]\` 作为当前 Decoder 输入。和 P92 一样，这里喂的是已知法语 token，用于验收接口；不是从 SOS 开始让模型自由生成。

每一步同时传入上一 hidden 与固定长度 Encoder outputs，得到当前输出、新 hidden 和注意力权重。`],
      ["23:46–27:17", "三项形状分别是 [1,4345]、[1,1,256]、[1,1,10]", `老师打印三项结果：output 覆盖 4345 个法语候选，hidden 保持一层、一个样本、256 维，attention weights 对十个固定源位置分配概率。目标句有几个 token，就会重复打印几组。

未训练时这些概率和权重没有翻译意义；本节只证明 Encoder、固定长度缓冲区与 Attention Decoder 的接口能够贯通。训练完成后返回 weights，才用于观察不同目标词的关注位置。`],
    ]),
    points: ["Encoder outputs 先复制进固定长度 10 的零张量", "Decoder hidden 直接接 Encoder hidden", "测试逐词喂真实 Y", "每步返回 [1,4345]、[1,1,256]、[1,1,10]"],
    code: `import torch\nmax_length,H=10,256\nencoder_outputs=torch.randn(1,6,H); hidden=torch.randn(1,1,H)\nfixed=torch.zeros(1,max_length,H)\nfixed[:,:encoder_outputs.shape[1],:]=encoder_outputs\nprint(encoder_outputs.shape,fixed.shape,hidden.shape)`,
    output: "真实 6 步 Encoder outputs 被放进 [1,10,256] 缓冲区，剩余位置为零。",
    pit: "不要把这段真实目标词循环写成推理，也不要声称零填充位置已经被 mask 为零权重。",
    quiz: "为什么源句只有 6 个 token，attention weights 仍有 10 个位置？", answer: "课程把注意力输出长度固定为 max_length=10，真实六步之外用零向量补齐。"
  },
  {
    p: 97, slug: "model-summary", title: "模型搭建总结：固定十步 Attention Decoder 的完整接口",
    codeLabel: "配套现代化实现示例（点积 Attention，与课堂版公式不同）",
    problem: "Encoder 与课程版 Attention Decoder 已写完，哪些尺寸和返回值必须保持一致？",
    chain: ["Encoder 返回 outputs/hidden", "outputs 复制进 [1,10,H]", "当前词+hidden 计算十个权重", "context 融合后更新 hidden", "返回 log_probs/hidden/weights"],
    extraVisuals: [architecture, fullPipeline],
    article: article([
      ["0:00–2:19", "Encoder 提供两份结果，Decoder 不能只拿最后状态", `老师总结 Encoder 输入英文 ID 后返回所有时间步 outputs 和 final hidden。hidden 用来初始化 Decoder；outputs 要复制到固定 \`[1,10,256]\` 缓冲区，供每个目标时间步重新计算注意力。

无 Attention 版本只依赖最后状态，课程版 Attention 则同时保留整段源序列信息。`],
      ["2:19–4:34", "Decoder 内部有两次拼接和两个不同 Linear", `当前法语输入的 Embedding 与上一 hidden 先拼成 512 维，经 \`attn\` 得十个位置权重；权重与固定 Encoder outputs 做 bmm 得 context。随后 Embedding 与 context 再拼成 512 维，经 \`attn_combine\` 降回 256，才进入 GRU。

若只记“拼接”却不区分两次目的，很容易把 512→10 和 512→256 两个 Linear 写反。`],
      ["4:34–6:22", "输出约定决定损失与可视化", `每步返回法语词表上的 LogSoftmax 对数概率、新 hidden 和十维 attention weights。对数概率与 NLLLoss 配套；weights 留给后续热力图。

Encoder 与 Decoder hidden_size 都是 256，才能直接传 hidden。课程 batch_size=1、max_length=10，也没有 source mask；这些是本案例接口的一部分，不应悄悄改写成另一套实现。`],
    ]),
    points: ["final hidden 初始化 Decoder", "全部 outputs 复制进固定十步缓冲区", "两次拼接作用不同", "LogSoftmax 配 NLLLoss", "weights 用于可视化"],
    code: commonCode,
    output: "示例展示一个可运行的现代化点积 Attention 变体；正文仍以老师的拼接式固定长度实现为准。",
    pit: "配套 Python 包采用点积 Attention，不能反过来把课堂视频说成点积实现。",
    quiz: "课程版为什么需要把真实 Encoder outputs 复制到长度 10？", answer: "attn 线性层固定输出十个权重，bmm 的源位置维必须与它一致。"
  },
  {
    p: 98, slug: "teacher-forcing", title: "Teacher Forcing：训练时有时喂真值上一词",
    problem: "为什么教师强制能加快训练，又会带来训练/推理不一致？",
    chain: ["Decoder 预测本词", "抛随机数", "真值上一词或预测词", "作为下一输入", "重复到目标末尾"],
    extraVisuals: [{
      title: "Teacher Forcing 分支时序",
      diagram: `
flowchart TB
    A["得到本步 log_probs"] --> B{"本条样本是否启用 Teacher Forcing？"}
    B -->|"是"| C["下一输入 = 真实 target_t"]
    B -->|"否"| D["下一输入 = topk(log_probs)"]
    C --> E["进入下一解码步"]
    D --> E
`,
    }],
    article: article([
      ["0:00–4:50", "真实目标是标签，Teacher Forcing 决定它是否也作为下一输入", `老师先把英文 X 称为特征、法语 Y 称为真实目标。Decoder 逐词预测时，本步输出无论对错都能与真实 Y 计算损失；Teacher Forcing 额外决定下一步是否直接喂真实词，而不是喂刚才的预测。`],
      ["4:50–8:47", "数学推导类比：中间一步错后，先纠正再让学生继续", `老师用辅导孩子做多步数学题类比。若第三步错了却把错误结果继续带到第四、第五步，后面大概率越来越偏；Teacher Forcing 相当于告诉他第三步的正确结果，再从正确状态继续推导。

它不是把后续答案全部提前泄露，而是在每个解码位置提供正确历史，减少早期随机错误的连锁影响。`],
      ["8:47–11:35", "作用与边界：收敛更快更稳，但预测时不会再有人纠正", `老师总结两点作用：纠正错误、避免越错越离谱，并加快训练收敛。下一节课程用随机数和 0.5 比例决定一条样本是否走 Teacher Forcing 分支。

预测阶段没有真实法语可喂，只能使用模型自己的输出。课程总结还提醒比例不宜盲目设得过高，否则训练时过度依赖真值历史，实际生成可能不稳定。`],
    ]),
    points: ["真实 Y 始终是监督标签", "Teacher Forcing 额外把真值作为下一输入", "避免错误逐步累积", "只用于训练", "课程下一节按样本随机选择分支"],
    code: `for ratio in (1.0,.5,0.0):\n    print(ratio,"真值概率",ratio)`,
    output: "显示三个常见教师强制比例的含义。",
    pit: "不要把 Teacher Forcing 说成‘发现哪一步错就精准纠正’；课程代码只按随机概率选择分支。",
    quiz: "Teacher Forcing 改变的是监督目标还是下一步输入？", answer: "改变下一步输入；监督损失仍然与真实目标词比较。"
  },
  {
    p: 99, slug: "train-one-batch", title: "单样本训练函数：Teacher Forcing 两条分支怎样累计 NLLLoss",
    codeLabel: "课堂训练伪代码（省略完整类与优化器定义）",
    problem: "课堂 batch_size=1 时，一对英法句子怎样完成编码、逐词解码、两种下一输入策略、反向传播和参数更新？",
    chain: ["Encoder 与固定长度缓冲区", "SOS+Encoder hidden 初始化 Decoder", "随机选择 Teacher Forcing", "逐词累计 NLLLoss", "两优化器反传更新"],
    extraVisuals: [trainingFlow],
    article: article([
      ["0:00–7:12", "为什么把单批训练单独抽成函数", `老师先解释结构选择：Seq2Seq 的单批训练包含编码、固定长度复制、逐词解码和 Teacher Forcing 分支，代码远长于普通分类任务。如果直接塞进 epoch×batch 双循环，函数会很难读，所以先把“一条句对如何训练”封装，外层训练函数只负责反复调用。

课堂 DataLoader 的 batch_size=1，因此老师口中的“一批”实际是一条英法句对。函数接收 X、Y、Encoder、Attention Decoder、两个优化器和损失函数。`],
      ["7:12–13:36", "Encoder 前向后复制到 [1,10,256]，Decoder 从 SOS 和 Encoder hidden 开始", `英文 X 先进入 Encoder，得到真实长度 outputs 和 final hidden。和测试 Attention Decoder 一样，老师创建固定 \`[1,10,256]\` 缓冲区，把真实 outputs 逐位置复制到前面。

Decoder 初始 hidden 直接取 Encoder final hidden；初始输入是形状 \`[1,1]\` 的 SOS_token，并迁移到相同 device。目标长度来自 Y 的第二维，决定最多循环多少个法语位置。`],
      ["13:36–20:41", "随机数只决定本条样本走 Teacher Forcing 还是自由回馈分支", `课程先生成一次随机数，与 teacher_forcing_ratio（示例 0.5）比较，得到本批是否使用 Teacher Forcing。它不是精准检测哪一步预测错了再纠正，而是按概率选择训练策略。

Teacher Forcing 分支中，每个时间步调用 Decoder 后，用真实 \`y[:, i]\` 作为当前监督目标并累计损失，同时把这个真实 token 作为下一步输入。这样即使本步预测错了，下一步仍从正确历史继续。`],
      ["20:41–29:42", "非 Teacher Forcing 分支用 topk 预测回馈，并允许 EOS 提前停止", `不使用 Teacher Forcing 时，每步仍用真实 \`y[:, i]\` 计算监督损失，但下一输入来自输出分布中概率最大的 token。课程用 \`topk(1)\` 同时取得最大值和对应索引，再把索引 detach 后作为下一步输入。

若预测 token 等于 EOS，解码循环立即 break；否则继续生成。这里必须区分“用于算损失的真实目标”和“作为下一输入的模型预测”，不能把二者写成同一个张量。`],
      ["29:42–34:18", "两条分支汇合后统一反传，并按目标长度返回平均损失", `循环结束后，老师对 Encoder 和 Decoder 两个优化器分别 zero_grad，累计损失执行 backward，再分别 step 更新参数。课程 Decoder 已输出 LogSoftmax，所以传入的实际损失对象是 NLLLoss；变量名即使写成 cross_entropy_loss，也不能改变它的真实类型。

函数最后返回累计损失除以目标句长度，供外层打印和画图。课程没有在这里使用 PAD ignore_index、批量展平或梯度裁剪；这些可以作为工程扩展，但不能冒充本节代码。`],
    ]),
    points: ["课堂一批就是一条句对", "Decoder 初始输入是 SOS", "真实 Y 始终用于监督损失", "下一输入由 Teacher Forcing 分支决定", "LogSoftmax 输出配 NLLLoss"],
    code: `# 伪代码保留课堂两条分支\nuse_teacher = random.random() < teacher_forcing_ratio\nfor i in range(target.shape[1]):\n    log_probs, hidden, weights = decoder(decoder_input, hidden, fixed_encoder_outputs)\n    truth = target[:, i]\n    loss = loss + nll_loss(log_probs, truth)\n    if use_teacher:\n        decoder_input = truth.view(1, -1)\n    else:\n        decoder_input = log_probs.topk(1).indices.detach()\n        if decoder_input.item() == EOS_token:\n            break`,
    output: "两条分支都用真实目标累计 NLLLoss，区别只在下一步喂真值还是模型预测。",
    pit: "不要把课程输出叫 logits 后再配 CrossEntropyLoss；也不要声称本节已实现 PAD ignore_index 或批量展平。",
    quiz: "非 Teacher Forcing 分支还需要真实 Y 吗？", answer: "需要；真实 Y 用来计算监督损失，只是下一步输入改用模型 topk 预测。"
  },
  {
    p: 100, slug: "view-function", title: "view(1,-1)：把单个 token 标量整理成 Decoder 需要的二维输入",
    problem: "P99 为什么反复对 y[0,i]、topi 调用 view(1,-1)，其中 1 和 -1 到底代表什么？",
    chain: ["先取一个 token 标量", "元素数量保持不变", "指定第一维为 1", "-1 自动推导剩余维", "得到 [1,1] 输入"],
    extraVisuals: [{
      title: "损失前的展平关系",
      diagram: `
flowchart LR
    A["y [1,T]"] --> B["y[0,i]：标量"]
    B --> C["view(1,-1)"]
    C --> D["[1,1]"]
    D --> E["作为 Decoder 当前输入"]
`,
    }],
    article: article([
      ["0:00–1:44", "问题来自 P99：索引一个目标词后只剩标量", `老师回到上一节的 \`y[0, i]\`。Y 原本形如 \`[1,T]\`，先取 batch 中第一个样本，再取第 i 个词后，只剩一个整数标量。可是课堂 Decoder 的当前输入约定为二维 \`[1,1]\`，不能直接把零维标量传进去。`],
      ["1:44–3:37", "view 只改变张量的形状解释，不会创造或删除元素", `老师用单独小脚本模拟数据，强调 view 前后元素总数必须相同。一个标量只有一个元素，所以它可以整理成一行一列，不能凭空改成两行三列。

\`view(1,-1)\` 中第一个 1 指定第一维为 1，-1 让 PyTorch 根据元素总数自动推导另一维；这里只有一个元素，最终就是 \`[1,1]\`。`],
      ["3:37–5:26", "真实 token 与 topk 预测都要整理成同一输入接口", `Teacher Forcing 分支中的真实 token，以及非 Teacher Forcing 分支取出的 topi，下一步都要成为 Decoder 输入，所以课程对它们做相同的二维整理。

本节没有讲 \`[B,T,V]→[B×T,V]\` 的损失展平，也没有讨论 transpose 后内存是否连续。那些是 view 的其他用途，不能替换老师本节围绕单 token 的解释。`],
    ]),
    points: ["索引单个 token 后是标量", "view 不改变元素数量", "1 固定第一维", "-1 自动推导剩余维", "课程最终需要 [1,1]"],
    code: `import torch\ny=torch.tensor([[119,297,465]])\ntoken=y[0,1]\nprint(token.shape,token.view(1,-1),token.view(1,-1).shape)`,
    output: "标量 token 被整理成值不变的 [1,1] 张量。",
    pit: "不要把本节误写成损失前展平 logits；老师解释的是单个 token 标量如何恢复二维输入。",
    quiz: "一个元素执行 view(1,-1) 后，-1 会被推导成多少？", answer: "1，所以形状是 [1,1]。"
  },
  {
    p: 101, slug: "full-training", title: "完整训练代码：多轮遍历、分段统计、逐轮保存与损失曲线",
    codeLabel: "课堂训练调度伪代码（需配合 P99 函数）",
    problem: "老师怎样把 P99 的单样本训练函数放进 epoch×DataLoader 循环，并观察训练是否收敛？",
    chain: ["创建两优化器与 NLLLoss", "外层 epoch/内层样本", "调用单样本训练", "按间隔打印和记录", "每轮保存两模型并画 loss"],
    extraVisuals: [trainingFlow, fullPipeline],
    article: article([
      ["0:00–8:14", "外层函数只搭训练骨架，具体前向仍调用 P99", `老师把完整训练函数定义成“多轮、多批次”的控制层。参数包括 DataLoader、Encoder、Attention Decoder、epoch 数、学习率、Teacher Forcing 比例，以及打印/画图间隔。单条句对怎样前向和反传不再重复书写，而是调用 P99 的训练函数。

这种拆分不是改变算法，而是避免把二三十行单样本逻辑嵌进双重循环后形成上百行函数。`],
      ["8:14–14:27", "分别创建 Encoder/Decoder 优化器，损失使用 NLLLoss", `课程为 Encoder 和 Decoder 各创建一个优化器，因为两套模型对象分别保存、分别 step。Decoder 内已调用 LogSoftmax，所以损失对象明确使用 \`nn.NLLLoss()\`。老师还再次提醒：如果删掉 Decoder 里的 LogSoftmax，才可改用 CrossEntropyLoss。

本课程没有 PAD token，也没有传 ignore_index。把它写进主流程会让读者误以为词表中已有 PAD。`],
      ["14:27–22:43", "epoch×DataLoader 双循环，每条样本返回一个平均损失", `外层循环控制训练轮数，内层遍历 DataLoader 中的 X/Y。每次调用单样本训练函数，传入两模型、两优化器、NLLLoss 和 Teacher Forcing 比例，得到当前法语句长度归一化后的损失。

老师的演示为了缩短录制时间，只训练每轮前约 3000 条，而完整语料有六万多条。他明确说明这会影响最终翻译效果，不能把演示结果当成充分训练后的性能。`],
      ["22:43–30:48", "两套累加器分别服务控制台打印和损失曲线", `课程设置两个间隔：例如每 1000 条打印一次平均损失和耗时，每 100 条把平均损失追加到绘图列表。每次计算完区间均值后，相应累加器必须归零，否则下一段会混入之前样本。

打印频率与绘图频率不同，所以老师维护两套总损失。曲线更密并不等于模型更好，只是记录点更频繁。`],
      ["30:48–36:53", "每轮分别保存 Encoder/Decoder，训练结束保存并显示 loss 曲线", `由于一轮可能耗时较长，老师在每个 epoch 结束时分别保存 Encoder 和 Attention Decoder 的参数文件，文件名带轮次。五轮会得到五个 Encoder 文件和五个 Decoder 文件。

训练结束后用记录的区间平均损失绘图，并先 savefig 再 show。课堂没有验证集、最佳模型选择、统一 checkpoint 或恢复优化器状态；这些属于后续工程改进，不能写成老师已经做过。`],
    ]),
    points: ["两模型各有一个优化器", "课程损失是 NLLLoss", "外层 epoch、内层 DataLoader", "打印与绘图使用不同间隔", "每轮分别保存两套 state_dict"],
    code: `criterion = torch.nn.NLLLoss()\nfor epoch in range(1, epochs + 1):\n    for index, (x, y) in enumerate(data_loader):\n        loss = train_one_pair(x, y, encoder, decoder, enc_opt, dec_opt, criterion, .5)\n        print_loss_total += loss\n        plot_loss_total += loss\n    torch.save(encoder.state_dict(), f"encoder_{epoch}.pth")\n    torch.save(decoder.state_dict(), f"decoder_{epoch}.pth")`,
    output: "每轮遍历样本、累计损失，并分别保存 Encoder 与 Decoder 参数。",
    pit: "不要声称本节包含验证集、最佳 checkpoint、PAD ignore_index 或梯度裁剪；课堂没有实现这些步骤。",
    quiz: "为什么损失用 NLLLoss 而不是 CrossEntropyLoss？", answer: "Decoder 已经输出 LogSoftmax 对数概率，NLLLoss 与它配套。"
  },
  {
    p: 102, slug: "training-summary", title: "训练结果与总结：五轮曲线下降，但 3000 条演示不代表充分训练",
    problem: "老师怎样判断本次训练正在收敛，并总结单样本与多轮训练两层结构？",
    chain: ["查看五轮耗时和模型文件", "观察 loss 曲线下降", "复述编码/解码", "区分两种下一输入", "理解演示训练边界"],
    extraVisuals: [trainingFlow],
    article: article([
      ["0:00–1:21", "五轮训练产出十个模型文件和一张损失图", `程序跑完五轮后，老师先检查控制台耗时、模型目录和 loss 图片。因为每轮分别保存 Encoder 与 Decoder，所以五轮一共出现十个参数文件。打印日志显示每隔一段样本的平均损失和耗时。`],
      ["1:21–2:33", "loss 整体下降只能说明优化在推进，不能证明翻译已经充分可靠", `老师打开损失曲线，看到整体向下，判断模型在当前训练数据上逐步收敛。曲线可以通过缩短记录间隔画得更密，但形状仍应结合实际样本判断。

本次演示每轮只取约 3000 条，远少于六万多条完整语料，因此生成效果仍有限。loss 下降不是“翻译完全正确”的证明。`],
      ["2:33–3:31", "单样本训练主线：编码、解码、Teacher Forcing、反传更新", `英文先经 Encoder 得到 outputs/hidden，再构造固定长度缓冲区；Decoder 从 SOS 和 Encoder hidden 开始逐词生成。若使用 Teacher Forcing，下一输入取真实法语词；否则取本步预测，并在 EOS 时提前停止。每步用真实目标累计 NLLLoss，最后反传并更新两模型。`],
      ["3:31–4:08", "多轮函数只负责外层调度，Teacher Forcing 比例不能误解", `外层训练函数完成 epoch×样本循环、区间损失记录、逐轮保存和绘图。老师最后用选择题强调：Teacher Forcing 可能造成训练与实际生成不一致，比例不宜简单设得过高；课程示例使用 0.5。

本节没有 PAD，也没有 target shift 的批量矩阵实现。复述时应以课堂的 batch_size=1 逐词循环为准。`],
    ]),
    points: ["五轮对应五对模型文件", "loss 下降不等于翻译已充分训练", "Teacher Forcing 只改变下一输入", "单样本训练与外层循环分开"],
    code: `print("X→Encoder→固定10步outputs→Decoder逐词→NLLLoss→两优化器更新")\nprint("下一输入：真实词（Teacher Forcing）或本步topk预测")`,
    output: "输出与课堂一致的训练主线和两种下一输入。",
    pit: "不要把只训练约 3000 条的演示模型描述成完整语料训练结果，也不要用训练 loss 下降替代翻译质量评估。",
    quiz: "Teacher Forcing 分支与非 Teacher Forcing 分支，哪一项始终不变？", answer: "两者都用真实目标词计算监督损失；变化的是下一步输入来源。"
  },
  {
    p: 103, slug: "prediction-code", title: "模型评估函数：关闭梯度后自回归生成并截取有效注意力矩阵",
    codeLabel: "课堂评估伪代码（需配合完整模型与词表）",
    problem: "训练完成后，怎样只给英文张量，在最多 10 步内生成法语词并保存每一步的十维注意力？",
    chain: ["no_grad 关闭梯度", "Encoder+固定 10 步缓冲区", "SOS 初始化", "topk 自回归直到 EOS", "返回词列表和有效 weights"],
    extraVisuals: [inferenceFlow],
    article: article([
      ["0:00–5:14", "评估开始先关闭梯度；含 Dropout 的 Decoder 还必须切到 eval", `老师把评估函数放在 \`torch.no_grad()\` 环境中。训练结束后权重已经固定，预测不需要保存反向传播图；关闭梯度能减少内存和计算。这个操作只用于评估/预测，训练阶段不能照搬。

原课堂演示重点讲 no_grad，没有单独展示 \`model.eval()\`；但 Attention Decoder 含 Dropout，正确评估还应对 Encoder 和 Decoder 调用 \`.eval()\`，否则 Dropout 仍随机丢弃特征，同一句话可能得到不稳定结果。函数接收已经数值化的英文 X、Encoder 和 Attention Decoder，加载权重和准备测试句放在下一节。`],
      ["5:14–10:37", "编码后仍要构造固定 [1,10,256] 的 Encoder outputs", `英文 X 进入 Encoder 得到真实长度 outputs 与 hidden。课程版 Attention Decoder 固定对十个源位置打分，所以评估函数也创建 \`[1,10,256]\` 零张量，把真实 outputs 复制到前面。

Decoder hidden 初始化为 Encoder hidden，当前输入初始化为 SOS_token。这里与训练一致，区别是后面没有真实法语 Y 可用。`],
      ["10:37–16:49", "准备法语词列表和 [10,10] 注意力矩阵", `老师创建列表保存生成的法语单词，并初始化 \`[10,10]\` 注意力矩阵。第一维预留最多十个目标生成步，第二维对应固定十个英文位置。

每次 Decoder 返回一行 \`[1,1,10]\` 权重，将它写入当前目标步所在的矩阵行。这样后面可以画出“目标词 × 源位置”的热力图。`],
      ["16:49–22:48", "每步用 topk 取概率最大词，EOS 时停止，否则回馈预测", `循环最多运行 max_length 次。每步调用 Decoder 得到对数概率、新 hidden 和注意力权重；用 \`topk(1)\` 取最大候选索引。若索引是 EOS，立即 break；否则通过法语 index2word 查回单词并加入列表。

当前预测索引随后 detach，并作为下一时间步输入。这正是自回归：预测阶段没有真实 Y，也绝不能使用 Teacher Forcing。`],
      ["22:48–26:22", "只返回实际生成步对应的注意力行", `如果第五步已经生成 EOS，预分配矩阵后面的行仍全是零，没有可视化意义。老师用当前 index 截取前 \`index+1\` 行，只返回实际执行过的注意力权重。

函数最终返回法语词列表和形如 \`[生成步数,10]\` 的注意力矩阵。课程词表没有 UNK/PAD 设计；下一节自定义英文句必须由现有英文词表中的词组成。`],
    ]),
    points: ["预测用 no_grad", "含 Dropout 的模型还要 eval", "源和目标最多都按 10 步处理", "从 SOS 开始", "topk 预测回馈", "EOS 终止", "注意力矩阵只截取有效生成行"],
    code: `# 课堂评估循环骨架\ndecoded_words=[]\nattention_matrix=torch.zeros(max_length,max_length)\ndecoder_input=torch.tensor([[SOS_token]],device=device)\nfor index in range(max_length):\n    log_probs,hidden,weights=decoder(decoder_input,hidden,fixed_encoder_outputs)\n    attention_matrix[index]=weights.squeeze(0).squeeze(0)\n    topi=log_probs.topk(1).indices\n    if topi.item()==EOS_token: break\n    decoded_words.append(fr_index2word[topi.item()])\n    decoder_input=topi.detach()\nreturn decoded_words,attention_matrix[:index+1]`,
    output: "得到预测法语词列表，以及 [实际生成步数,10] 的注意力矩阵。",
    pit: "预测阶段不能使用真实法语词；课程词表也没有 UNK，直接输入词表外英文会查表失败。",
    quiz: "为什么返回 attention_matrix[:index+1] 而不是完整 [10,10]？", answer: "EOS 可能提前结束，后面的预留行全是零，不属于实际生成步骤。"
  },
  {
    p: 104, slug: "prediction-test", title: "模型评估测试：加载两份权重并对照英文、真值法语和预测法语",
    codeLabel: "课堂评估调用骨架（需配合完整模型与词表）",
    problem: "怎样恢复 Encoder/Decoder 参数，对数据集样本或词表内自定义句子进行翻译，并正确理解演示结果的局限？",
    chain: ["按同结构创建两模型", "load_state_dict 恢复权重", "准备句对和英文 ID", "调用 evaluate", "打印英文/真值/预测"],
    extraVisuals: [inferenceFlow],
    article: article([
      ["0:00–5:36", "先按训练时相同结构创建 Encoder 与 Decoder", `老师指定某一轮保存的 Encoder、Attention Decoder 参数路径，再用相同英文词表大小、法语词表大小、hidden_size、dropout 和 max_length 重建模型。结构参数必须一致，否则 state_dict 不能正确加载。

两套模型都迁移到当前 device。加载时使用 map_location，使 GPU 训练的权重也能映射到 CPU 预测。课堂代码显式写了 \`weights_only=False\`；这只应加载自己信任的文件，因为完整 pickle 反序列化不适合不可信来源。对于只保存 \`state_dict\` 的现代 PyTorch 环境，兼容时优先使用 \`weights_only=True\`。加载后还应调用 \`.eval()\` 关闭 Decoder 的 Dropout。`],
      ["5:36–10:51", "数据处理函数同时提供句对与两套正反词表", `评估不仅需要模型参数，还需要训练时相同的英文 word2index 和法语 index2word。前者把英文输入编码成 ID，后者把 Decoder 预测 ID 还原成法语词。

词表与模型输出行号必须来自同一版本。只拿到权重而换了一套索引，即使形状一致，ID 的词义也会错位。`],
      ["10:51–16:47", "既可遍历数据集句对，也可输入词表覆盖范围内的自定义英文", `老师准备若干英文、法文句对，依次取 source 和真实 target。英文按空格切词，通过英文词表查 ID，追加 EOS，再创建 \`[1,S]\` long 张量并迁移设备。

由于课程没有 UNK，自定义句子中的每个英文词都必须已经存在于约 2803 词的英文词表中，否则直接索引会报错。`],
      ["16:47–20:15", "调用 evaluate 后同时打印原文、真值译文和预测译文", `评估函数返回预测法语词列表和注意力矩阵。老师用空格拼接预测词，并把英文原句、数据集中的真实法语、模型预测法语并排打印，便于直观比较。

这不是只展示一个成功例，而是让同学看到当前模型仍会出错。预测阶段不再提供真实法语给 Decoder，真值只用于人眼对照。`],
      ["20:15–22:36", "演示只训练少量数据，结果不准首先应回到训练覆盖率", `老师解释当前每轮只训练约 3000 条，而完整语料有六万多条，所以模型看到的数据比例很小。要改善效果，应取消演示用的提前 break，让更多语料、更多轮次参与训练，同时承担更长计算时间。

课堂没有实现 UNK、beam search、自动验证指标或 PAD mask；这些可以作为后续升级方向，但不能写成本节已经完成的错误诊断流程。`],
    ]),
    points: ["模型结构必须与权重一致", "词表必须与训练时一致", "预测只输入英文", "真实法语只用于对照", "少量演示训练不能代表最终质量"],
    code: `encoder.load_state_dict(torch.load(encoder_path,map_location=device,weights_only=True))\ndecoder.load_state_dict(torch.load(decoder_path,map_location=device,weights_only=True))\nencoder.eval(); decoder.eval()\nfor english,true_french in pairs:\n    x_ids=[en_word2index[word] for word in english.split()]+[EOS_token]\n    x=torch.tensor(x_ids,dtype=torch.long,device=device).view(1,-1)\n    predicted,weights=evaluate(x,encoder,decoder)\n    print(english,"|",true_french,"|"," ".join(predicted))`,
    output: "逐条打印英文原句、真实法语和模型预测法语。",
    pit: "课程没有 UNK；自定义英文含词表外单词时不会自动回退，而会查表失败。",
    quiz: "为什么只加载模型权重还不够？", answer: "还必须使用训练时相同的词表映射，否则同一个输出 ID 会被还原成错误单词。"
  },
  {
    p: 105, slug: "tensorboard-graph", title: "绘制注意力热力图：横轴英语、纵轴法语，颜色表示依赖程度",
    codeLabel: "注意力绘图调用骨架（需传入评估结果）",
    problem: "evaluate 返回的 [目标步数,10] 权重怎样画成热力图，又该如何避免把图读反？",
    chain: ["加载模型与样例", "调用 evaluate 得 weights", "imshow 绘制矩阵", "设置英法 token 刻度", "颜色越浅表示权重越大"],
    extraVisuals: [inferenceFlow],
    article: article([
      ["0:00–4:42", "本节画的是注意力矩阵，不是 TensorBoard 计算图", `老师先展示一张翻译对齐热力图：一条轴对应英语源句，另一条轴对应生成的法语词，格子的明暗代表生成某个法语词时对某个英语位置的权重。代码主要用于查看，老师明确说不要求同学逐行抄写。

因此本节目标是解释模型关注关系，不是用 SummaryWriter 或 add_graph 检查网络结构。`],
      ["4:42–8:58", "复用评估流程取得预测词和注意力矩阵", `函数先准备 DataLoader、Encoder、Attention Decoder，并加载训练好的两份权重。选定英文句子后，按训练词表编码、追加 EOS，再调用上一节 evaluate，得到预测法语词列表和裁剪后的注意力矩阵。

如果模型生成 7 个步骤，返回矩阵约为 \`[7,10]\`：7 是实际目标生成步，10 是课程固定源长度。`],
      ["8:58–12:42", "imshow 把矩阵画出来，刻度必须与真实 token 对齐", `老师将注意力张量转成 NumPy 后交给 Matplotlib \`imshow\`，再设置横纵轴刻度标签并保存图片。源句真实 token 少于 10 时，图中仍可能包含补出的固定位置；标签和矩阵列数必须对齐。

预测词列表与矩阵行数也要对应；EOS 是否显示取决于返回词列表和刻度构造，不能只凭坐标数字猜词。课堂口头解释中有一处把横纵轴说反；按 evaluate 返回的矩阵 \`[目标生成步, 源位置]\` 直接 imshow 时，列/x 轴是英语源位置，行/y 轴是法语目标步骤。应以张量维度和实际 set_xticklabels/set_yticklabels 代码为准。`],
      ["12:42–17:45", "正确读图：一个法语词对应一整行源位置权重", `阅读时选定一个生成的法语词，沿对应矩阵行观察它对各英语位置的权重。课堂配色中颜色越浅表示依赖越强；具体明暗方向取决于 colormap，换配色后必须看色标，不能把“越浅越强”当作所有热力图的通则。

老师用前后词的大致对齐说明 Attention 会随生成步变化。热力图展示模型内部权重，不保证翻译一定正确；本节最后以这张图收束 RNN、GRU、Attention 与英法翻译案例。`],
    ]),
    points: ["本节是 attention heatmap，不是 TensorBoard", "矩阵行对应目标生成步", "矩阵列对应固定十个源位置", "刻度必须与 token 对齐", "颜色意义要结合 colormap"],
    code: `import matplotlib.pyplot as plt\nplt.imshow(attention_weights.cpu().numpy(),aspect="auto",cmap="bone")\nplt.xlabel("English source positions")\nplt.ylabel("Generated French positions")\nplt.colorbar()\nplt.savefig("attention.png")\nplt.show()`,
    output: "生成目标步×源位置的注意力热力图。",
    pit: "不要把横纵轴读反，也不要把注意力高直接等同于翻译正确；换 colormap 后先查看色标方向。",
    quiz: "注意力矩阵形状是 [7,10] 时两个数字分别表示什么？", answer: "7 个实际目标生成步骤，每一步对固定 10 个英语源位置各有一个权重。"
  },
];

generateTopic({
  topic: "GRU Seq2Seq 英法翻译",
  folder: "seq2seq-translation",
  lessons,
  stages: [
    { title: "一、需求、设备与数据管道", start: 1, end: 8 },
    { title: "二、Encoder 与两类 Decoder", start: 9, end: 18 },
    { title: "三、Teacher Forcing、训练与预测", start: 19, end: 26 },
  ],
  codePackage: {
    label: "seq2seq_from_scratch 配套实现",
    href: "../../seq2seq_from_scratch/README.md",
    test: "python3 -m unittest seq2seq_from_scratch.test_model -v",
  },
});
