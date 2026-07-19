import { generateTopic } from "./course_note_helpers.mjs";

const article = (sections) => sections.map(([time, title, body]) =>
  `### ${time}　${title}\n\n${body}`
).join("\n\n");

const fullPipeline = {
  title: "英法翻译从数据到预测的总流程",
  diagram: `
flowchart LR
    A["英法平行语料"] --> B["清洗与标准化"]
    B --> C["两套词表 + SOS/EOS/PAD"]
    C --> D["Dataset / DataLoader"]
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
      source_embedding
      gru
      forward(source_ids)
    }
    class AttentionDecoderGRU {
      target_embedding
      attention
      gru
      classifier
      forward_step(token, hidden, encoder_outputs)
    }
    class Seq2Seq {
      teacher_forcing_ratio
      forward(source, target)
      greedy_decode(source)
    }
    Seq2Seq *-- EncoderGRU
    Seq2Seq *-- AttentionDecoderGRU
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
    T["当前 token [B]"] --> E["Embedding [B,1,E]"]
    H["hidden [1,B,H]"] --> Q["Q [B,H]"]
    O["Encoder outputs [B,S,H]"] --> A["Attention"]
    Q --> A
    A --> C["context [B,1,H]"]
    E --> J["拼接 [B,1,E+H]"]
    C --> J
    J --> G["GRU → [B,1,H]"]
    G --> L["Linear → logits [B,Vt]"]
`,
};

const trainingFlow = {
  title: "训练时一批数据的调用时序",
  diagram: `
sequenceDiagram
    participant D as DataLoader
    participant E as Encoder
    participant A as Attention Decoder
    participant L as CrossEntropyLoss
    participant O as Optimizer
    D->>E: source [B,S]
    E-->>A: outputs [B,S,H] 与 hidden
    loop 目标位置 1..T-1
      A->>A: 当前输入词 → logits
      A->>A: 按 Teacher Forcing 决定下一输入
    end
    A->>L: logits [B,T-1,V] 与 target[:,1:]
    O->>O: zero_grad → backward → step
`,
};

const inferenceFlow = {
  title: "推理时逐词生成流程",
  diagram: `
flowchart TB
    A["编码源句"] --> B["输入 SOS"]
    B --> C["Decoder 单步"]
    C --> D["argmax / beam 选 token"]
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
      source_sentences
      target_sentences
      __getitem__()
    }
    class CollateFunction {
      padding
      lengths
      masks
    }
    class DataLoader {
      batch_size
      shuffle
    }
    TranslationDataset --> Vocabulary
    TranslationDataset --> CollateFunction
    CollateFunction --> DataLoader
`,
};

const commonCode = `import torch\nfrom seq2seq_from_scratch.model import EncoderGRU, AttentionDecoderGRU, Seq2Seq\nencoder=EncoderGRU(100,16,32)\ndecoder=AttentionDecoderGRU(120,16,32)\nmodel=Seq2Seq(encoder,decoder,start_id=1,end_id=2)\nprint(type(model).__name__)`;

const lessons = [
  {
    p: 80, slug: "translation-requirements", title: "英译法需求：先看懂 800 行项目的六个模块",
    problem: "英译法项目代码很多，怎样先建立全景，不被某一个函数带跑？",
    chain: ["平行语料", "数据与词表", "Encoder", "Attention Decoder", "训练", "预测/可视化"],
    extraVisuals: [fullPipeline, architecture],
    article: article([
      ["0:00–4:54", "任务与章节路线", "老师说明输入英文句子、输出法文句子，是 N→M 的 Seq2Seq。项目会覆盖 CUDA、清洗、Dataset/DataLoader、GRU Encoder、无/有注意力 Decoder、Teacher Forcing、训练、预测与 TensorBoard。"],
      ["4:54–9:52", "为什么代码量大", "算法主体并非 800 个新知识点，代码量来自数据管道、三类模型组件、训练日志、保存加载和测试。先按模块理解接口，再串成主流程。"],
      ["9:52–14:53", "输入输出的边界符", "源句和目标句各自建词表，加入 SOS（开始）、EOS（结束）、PAD（补齐）等。模型训练预测的是下一个 token ID，最后再从法语词表还原文本。"],
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
    p: 85, slug: "preprocessing", title: "数据预处理：建两套词表并加入 SOS/EOS",
    problem: "干净字符串怎样变成模型训练用的 token ID 序列？",
    chain: ["分词", "统计词频", "建立 src/tgt 词表", "加 SOS/EOS/UNK/PAD", "编码为 ID"],
    extraVisuals: [dataUml, fullPipeline],
    article: article([
      ["0:00–7:52", "词表对象", "老师建立词→索引、索引→词、词频三个结构。add_sentence 遍历词，add_word 分配新 ID 并累计次数。"],
      ["7:52–15:54", "特殊 token", "SOS 作为解码起点，EOS 标识结束，PAD 用于批量补齐，UNK 接住未登录词。每个 ID 必须固定并持久化。"],
      ["15:54–23:52", "编码一对句子", "源句通常追加 EOS；目标训练序列可组织为 `[SOS, y1, ..., EOS]`。模型输入目标的前 T-1 个位置，标签是后移一位的 T-1 个位置。"],
      ["23:52–31:46", "过滤与统计", "课程按前缀/长度筛选部分英法句对降低难度。过滤规则会改变任务分布，应记录保留率，并只用训练集建词表避免泄漏。"],
    ]),
    points: ["src/tgt 词表独立", "目标输入与标签相差一位", "词表只由训练集建立"],
    code: `target=["<SOS>","je","suis","ici","<EOS>"]\nprint("input",target[:-1])\nprint("label",target[1:])`,
    output: "模型每一步输入前一个 token，预测下一个 token。",
    pit: "忘记 EOS 会让模型不知道何时停止。",
    quiz: "为什么标签从 target[1:] 开始？", answer: "输入 SOS 时应预测第一个真实词，输入最后真实词时应预测 EOS。"
  },
  {
    p: 86, slug: "dataset", title: "构建 Dataset：一条样本同时返回源 ID 与目标 ID",
    problem: "Dataset 怎样保持句对对应，并在何处做张量化最清晰？",
    chain: ["保存句对", "__len__", "__getitem__", "词表 encode", "返回 long tensors"],
    extraVisuals: [dataUml],
    article: article([
      ["0:00–5:59", "Dataset 骨架", "构造函数保存 pairs 和两套词表，__len__ 返回句对数，__getitem__(i) 读取同一对源/目标文本。"],
      ["5:59–11:56", "文本转 ID", "源句、目标句各自调用对应词表编码，得到 torch.long 张量。Embedding 接受整数 ID，不接受 float。"],
      ["11:56–16:37", "职责边界", "单样本可保持变长；padding 更适合放进 collate_fn，由一个 batch 的最大长度决定，避免整个数据集统一补到全局最大值。"],
    ]),
    points: ["__getitem__ 返回成对张量", "Embedding 输入 dtype 是 long", "padding 可延迟到 collate"],
    code: `import torch\nsample={"source":torch.tensor([4,7,2]),"target":torch.tensor([1,9,5,2])}\nprint(sample["source"].dtype)`,
    output: "ID 张量类型是 torch.int64/long。",
    pit: "把 token ID 转 float 会让 Embedding 报类型错误。",
    quiz: "为什么不在 Dataset 初始化时一次性补到全局最长？", answer: "会浪费大量内存和计算；按 batch 补齐通常更省。"
  },
  {
    p: 87, slug: "dataloader", title: "获取 DataLoader：补齐、长度和 mask 一起产出",
    problem: "不同长度句子怎样组成一个 batch，又不让 PAD 干扰注意力？",
    chain: ["抽取多条样本", "分别 pad source/target", "记录真实长度", "建立 source mask", "返回 batch"],
    extraVisuals: [dataUml],
    article: article([
      ["0:00–3:50", "加载器参数", "老师创建 Dataset 后用 DataLoader 设置 batch_size、shuffle 等。训练集可打乱，验证/测试保持稳定。"],
      ["3:50–6:49", "课程简化与批量现实", "若示例 batch_size=1，变长不会冲突；要提高效率，应自定义 collate_fn，把每批源和目标分别补齐。"],
      ["6:49–8:43", "mask", "source_mask 形状 [B,S]，真实 token 为 True、PAD 为 False。注意力算分后把 PAD 分数屏蔽，再 Softmax，否则模型会把概率分给无内容位置。"],
    ]),
    points: ["源/目标可以有不同最大长度", "mask 与 source_ids 同前两维", "只训练集 shuffle"],
    code: `import torch\nsource=torch.tensor([[4,5,2,0],[7,2,0,0]])\nmask=source.ne(0)\nprint(mask)`,
    output: "非 PAD 位置为 True，可用于注意力屏蔽。",
    pit: "先 Softmax 再把 PAD 权重清零会导致剩余概率和小于 1；应在 Softmax 前 mask 分数。",
    quiz: "第二条样本只有两个有效 token，mask 是什么？", answer: "[True, True, False, False]。"
  },
  {
    p: 88, slug: "gru-encoder", title: "GRU Encoder：Embedding 后保留每个时间步输出",
    problem: "Encoder 为什么既返回 outputs，又返回 final hidden？",
    chain: ["source IDs [B,S]", "Embedding [B,S,E]", "GRU", "outputs [B,S,H]", "hidden [1,B,H]"],
    extraVisuals: [encoderShapes, architecture],
    article: article([
      ["0:00–6:51", "类的两层", "Encoder 由 Embedding 和 GRU 组成。Embedding 把离散词 ID 变成连续 E 维向量；GRU 将序列编码成上下文状态。"],
      ["6:51–14:49", "forward 形状", "source_ids[B,S]→embedded[B,S,E]→outputs[B,S,H] 与 hidden[1,B,H]。若课程代码按 seq_first，前两维顺序会相反；本笔记统一 batch_first。"],
      ["14:49–20:45", "为什么 outputs 不能丢", "无注意力 Decoder 可能只需 final hidden；注意力 Decoder 要与每个源位置匹配，所以必须保留全部 outputs。"],
      ["20:45–23:36", "初始化与 device", "GRU 可省略 h0 使用零状态，也可显式创建。输入、初始状态和模型参数应在同一设备。"],
    ]),
    points: ["Embedding 输出维 E，GRU 隐藏维 H", "outputs 保留源位置轴 S", "attention 需要全部 outputs"],
    code: `import torch\nfrom seq2seq_from_scratch.model import EncoderGRU\nm=EncoderGRU(100,16,32)\nout,h=m(torch.randint(0,100,(4,7)))\nprint(out.shape,h.shape)`,
    output: "outputs=[4,7,32]，hidden=[1,4,32]。",
    pit: "把 outputs 只取最后一步后，注意力就失去逐源词选择能力。",
    quiz: "B=4、S=7、H=32 时 outputs 形状？", answer: "[4,7,32]。"
  },
  {
    p: 89, slug: "test-encoder", title: "测试 Encoder：先验形状再运行",
    problem: "怎样在接 Decoder 前证明编码器接口完全正确？",
    chain: ["造 source IDs", "实例化 Encoder", "前向", "检查 outputs", "检查 hidden/梯度"],
    extraVisuals: [encoderShapes],
    article: article([
      ["0:00–5:57", "准备随机 ID", "随机数范围必须小于词表大小，dtype 为 long；batch 和句长选择小数便于查看。"],
      ["5:57–11:51", "检查两类返回值", "outputs 的序列维等于输入 S，末维等于 H；hidden 第一维等于层数×方向，第二维 B，末维 H。"],
      ["11:51–17:03", "数值测试边界", "随机未训练输出没有翻译意义。应再检查 PAD embedding、device、反向传播，以及相同随机种子下可复现形状/数值。"],
    ]),
    points: ["随机 ID 必须在词表范围", "shape 测试先于业务训练", "随机输出只验证接口"],
    code: `import torch\nfrom seq2seq_from_scratch.model import EncoderGRU\nx=torch.randint(0,50,(2,5)); out,h=EncoderGRU(50,8,12)(x)\nassert out.shape==(2,5,12) and h.shape==(1,2,12)\nprint("ok")`,
    output: "断言通过后打印 ok。",
    pit: "torch.randn 产生浮点数，不能直接作为 Embedding 的 token ID。",
    quiz: "为什么用 randint 而非 randn？", answer: "Embedding 需要离散整数索引。"
  },
  {
    p: 90, slug: "plain-decoder-plan", title: "无 Attention Decoder 思路：只靠 final hidden 生成",
    problem: "无注意力版本怎样工作，它为什么是理解有注意力版本的对照组？",
    chain: ["输入上一个目标 token", "目标 Embedding", "与 hidden 送 GRU", "Linear→目标词表", "预测下一 token"],
    extraVisuals: [architecture],
    article: article([
      ["0:00–2:58", "单步 Decoder", "每步输入一个目标 token ID 和上一 hidden；Embedding 后经 GRU，Linear 输出目标词表 logits。"],
      ["2:58–5:19", "信息来源的限制", "源句只通过 Encoder final hidden 进入 Decoder。长句所有细节都压在一个状态里，这是固定 C 瓶颈。"],
    ]),
    points: ["Decoder 每步只预测一个词", "输出维等于目标词表大小", "无注意力只接 final hidden"],
    code: `print("token + hidden → GRU → Linear(target_vocab)")`,
    output: "显示无注意力单步的最短数据流。",
    pit: "目标 Embedding 的词表大小必须用法语词表，而非英语词表。",
    quiz: "无注意力 Decoder 从哪里得到源句信息？", answer: "Encoder 的最终 hidden。"
  },
  {
    p: 91, slug: "plain-decoder-code", title: "构建无 Attention GRU Decoder",
    problem: "怎样写一个可重复调用的 decoder step，并清楚区分 logits 与概率？",
    chain: ["Embedding 当前 token", "增加时间维", "GRU(input,hidden)", "Linear", "返回 logits/new hidden"],
    extraVisuals: [architecture],
    article: article([
      ["0:00–6:54", "初始化", "定义目标 Embedding、GRU、Linear。GRU input_size 等于 embedding_size，Linear 从 hidden_size 映射到 target_vocab_size。"],
      ["6:54–13:58", "forward 单步", "token[B]→Embedding[B,E]→unsqueeze 得 [B,1,E]→GRU→output[B,1,H]→squeeze→Linear 得 logits[B,Vt]。"],
      ["13:58–20:29", "返回值与损失", "返回 logits 与 new hidden。训练配 CrossEntropyLoss 时不要先 Softmax；推理需要概率可在外部 softmax。"],
    ]),
    points: ["单步输入 token 形状 [B]", "GRU 仍需要长度为 1 的时间维", "CrossEntropyLoss 接 logits"],
    code: `import torch\nemb=torch.nn.Embedding(120,16); gru=torch.nn.GRU(16,32,batch_first=True); fc=torch.nn.Linear(32,120)\nx=emb(torch.tensor([1,1])).unsqueeze(1); out,h=gru(x)\nprint(fc(out.squeeze(1)).shape)`,
    output: "两样本得到 [2,120] 目标词表 logits。",
    pit: "不要 squeeze() 不指定维度，batch=1 时可能把 batch 维也删掉。",
    quiz: "为什么 unsqueeze(1)？", answer: "把 [B,E] 变为 GRU 需要的 [B,1,E]。"
  },
  {
    p: 92, slug: "test-plain-decoder", title: "测试无 Attention Decoder：逐词循环与 EOS",
    problem: "一个单步 Decoder 怎样循环成完整目标句？",
    chain: ["以 SOS 开始", "decoder step", "选 top1", "作为下一输入", "遇 EOS 停止"],
    extraVisuals: [inferenceFlow],
    article: article([
      ["0:00–8:56", "单步测试", "用 Encoder hidden 初始化 Decoder，输入 SOS，检查 logits[B,Vt] 与 hidden[1,B,H]。"],
      ["8:56–18:52", "循环生成", "argmax 选下一 token，重新送入 Decoder，直到 EOS 或 max_length。若不设最大长度，未训练或异常模型可能无限生成。"],
      ["18:52–29:46", "训练与推理差别", "测试循环用自身预测作为下一输入；训练可用真实上一词。不要在推理时偷用目标答案。"],
      ["29:46–39:36", "接口验收", "检查每步词 ID 范围、hidden 形状不变、EOS 能终止。无注意力输出作为后续注意力版本对照。"],
    ]),
    points: ["SOS 启动，EOS 终止", "最大长度是安全上限", "推理下一输入来自模型预测"],
    code: `tokens=["<SOS>"]\nfor predicted in ["je","suis","ici","<EOS>"]:\n    tokens.append(predicted)\n    if predicted=="<EOS>": break\nprint(tokens)`,
    output: "生成序列在 EOS 处结束。",
    pit: "忘记 break 或最大长度会造成死循环。",
    quiz: "推理时能把真实法语上一词喂进去吗？", answer: "不能，真实目标在实际使用中不存在。"
  },
  {
    p: 93, slug: "attention-decoder-plan", title: "有 Attention Decoder 思路：每步重新查询源句",
    problem: "相比无注意力版本，单步 Decoder 新增了哪些输入和计算？",
    chain: ["hidden 作 Q", "Encoder outputs 作 K/V", "得到 weights/context", "拼目标 embedding", "GRU 预测"],
    extraVisuals: [decoderShapes, architecture],
    article: article([
      ["0:00–5:59", "新增 Encoder outputs", "除了当前 token 与 hidden，还要传入 Encoder 全部 outputs[B,S,H]。hidden[-1] 作为当前 Q。"],
      ["5:59–11:57", "计算 context", "Q 与 S 个 key 打分，mask PAD，Softmax 得 [B,S] 权重，再加权 values 得 [B,1,H] context。"],
      ["11:57–16:46", "融合再送 GRU", "当前目标 embedding[B,1,E] 与 context[B,1,H] 拼成 [B,1,E+H]，因此 Decoder GRU 的 input_size 要设 E+H。"],
      ["16:46–20:25", "返回注意力矩阵", "每步除 logits/hidden 外再返回 weights；所有目标步堆叠后形成 [B,T,S]，可画翻译对齐热力图。"],
    ]),
    points: ["attention 每个目标步重算", "Decoder GRU 输入维变 E+H", "返回 weights 便于解释和调试"],
    code: `import torch\nfrom seq2seq_from_scratch.model import AttentionDecoderGRU\nm=AttentionDecoderGRU(120,16,32)\nlogits,h,w=m.forward_step(torch.tensor([1]),torch.zeros(1,1,32),torch.randn(1,5,32))\nprint(logits.shape,h.shape,w.shape)`,
    output: "logits=[1,120]、hidden=[1,1,32]、weights=[1,5]。",
    pit: "忘记把 context 拼进 GRU 输入，模型就没有真正利用注意力结果。",
    quiz: "源长 S=5 时每个目标步有几个权重？", answer: "5 个。"
  },
  {
    p: 94, slug: "attention-decoder-code-part1", title: "有 Attention Decoder 代码（上）：定义层与接口",
    problem: "初始化参数怎样与 Embedding、GRU、分类层的输入输出一一对应？",
    chain: ["目标 Embedding Vt→E", "Attention H/H", "GRU 输入 E+H", "GRU 隐藏 H", "Linear H→Vt"],
    extraVisuals: [decoderShapes],
    article: article([
      ["0:00–4:52", "构造参数", "target_vocab_size 决定 Embedding 行数与最终类别数；embedding_size=E；hidden_size=H。"],
      ["4:52–8:48", "层的尺寸", "Embedding(Vt,E)，GRU(E+H,H)，Linear(H,Vt)。只要把这三行尺寸推出来，后面大部分形状不会乱。"],
      ["8:48–11:35", "forward_step 接口", "输入 current_ids[B]、hidden[1,B,H]、encoder_outputs[B,S,H]、可选 source_mask[B,S]；输出 logits、new_hidden、weights。"],
    ]),
    points: ["GRU input_size=E+H", "Linear out_features=目标词表大小", "mask 形状与源序列一致"],
    code: `from seq2seq_from_scratch.model import AttentionDecoderGRU\nm=AttentionDecoderGRU(vocabulary_size=120,embedding_size=16,hidden_size=32)\nprint(m.gru.input_size,m.classifier.out_features)`,
    output: "GRU 输入维 48，分类输出 120。",
    pit: "把 source_vocab_size 用到 Decoder Embedding 会产生错误词义映射。",
    quiz: "E=16、H=32 时 GRU input_size 是多少？", answer: "48。"
  },
  {
    p: 95, slug: "attention-decoder-code-part2", title: "有 Attention Decoder 代码（下）：逐行完成 forward_step",
    problem: "如何从隐藏状态一步步得到 logits，并确保每个变形都不误删维度？",
    chain: ["query=hidden[-1]", "bmm 得 scores", "mask+Softmax", "bmm 得 context", "拼 embedding→GRU→Linear"],
    extraVisuals: [decoderShapes],
    article: article([
      ["0:00–5:52", "Q/K 打分", "query=hidden[-1] 得 [B,H]；query.unsqueeze(-1) 得 [B,H,1]；encoder_outputs[B,S,H] bmm 后得到 [B,S,1]，squeeze(-1) 成 [B,S]。"],
      ["5:52–10:50", "mask 与权重", "PAD 位置分数填很小负数，再 Softmax(dim=-1)。这样权重非负、每样本和为 1、PAD 接近 0。"],
      ["10:50–13:50", "加权 V", "weights.unsqueeze(1)[B,1,S] bmm encoder_outputs[B,S,H] 得 context[B,1,H]。"],
      ["13:50–16:02", "GRU 与分类", "目标 embedding[B,1,E] 与 context 拼接，GRU 更新 hidden，Linear 输出 [B,Vt] logits。"],
    ]),
    points: ["squeeze 只删明确为 1 的维", "mask 在 Softmax 前", "context 保留时间维便于拼接"],
    code: `import torch\nB,S,H=2,5,8\nenc=torch.randn(B,S,H); q=torch.randn(B,H)\nscores=torch.bmm(enc,q.unsqueeze(-1)).squeeze(-1)\nw=torch.softmax(scores,-1); c=torch.bmm(w.unsqueeze(1),enc)\nprint(scores.shape,w.shape,c.shape)`,
    output: "scores/weights=[2,5]，context=[2,1,8]。",
    pit: "使用 squeeze() 可能在 B=1 时误删 batch 维；写 squeeze(-1)。",
    quiz: "为什么 context 是 [B,1,H] 而不是 [B,S,H]？", answer: "权重已把 S 个源位置加权求和成一个本步上下文。"
  },
  {
    p: 96, slug: "test-attention-decoder", title: "测试 Attention Decoder：权重、mask 与单步输出",
    problem: "怎样证明注意力 Decoder 真正在看源序列，而不只是形状能跑？",
    chain: ["编码源句", "准备 SOS/hidden", "调用单步", "检查 weights", "检查 mask/梯度"],
    extraVisuals: [decoderShapes],
    article: article([
      ["0:00–7:55", "测试输入", "创建小词表和短源句，先跑 Encoder 得 outputs/hidden，再用 SOS 调 Decoder。"],
      ["7:55–15:50", "形状断言", "logits[B,Vt]、new_hidden[1,B,H]、weights[B,S]。每行 weights 和约等于 1。"],
      ["15:50–22:42", "mask 与 batch", "把源尾部设为 PAD 并传 mask，确认对应权重为 0；batch>1 可发现 squeeze/广播错误。"],
      ["22:42–27:17", "数值意义", "未训练权重没有语言解释；测试关注概率性质、梯度可回传和有限值。训练后才可观察对齐。"],
    ]),
    points: ["权重和为 1", "PAD 权重为 0", "测试 batch>1"],
    code: `import torch\nfrom seq2seq_from_scratch.model import EncoderGRU,AttentionDecoderGRU\nenc=EncoderGRU(20,6,8); dec=AttentionDecoderGRU(25,7,8)\no,h=enc(torch.randint(3,20,(2,5))); logits,h,w=dec.forward_step(torch.ones(2,dtype=torch.long),h,o)\nprint(logits.shape,w.sum(-1))`,
    output: "logits=[2,25]，每条权重和为 1。",
    pit: "只打印一大串随机数不如写 shape 和性质断言。",
    quiz: "未训练时某词权重最高能说明翻译对齐吗？", answer: "不能，只是随机初始化结果。"
  },
  {
    p: 97, slug: "model-summary", title: "模型搭建总结：三个模块如何对接",
    problem: "Encoder、Decoder、Attention 的输出输入怎样闭环，哪里最容易尺寸不一致？",
    chain: ["Encoder outputs/hidden", "Decoder 用 hidden 作 Q", "Attention 读 outputs", "Decoder 更新 hidden", "循环到 EOS"],
    extraVisuals: [architecture, fullPipeline],
    article: article([
      ["0:00–3:47", "三类接口", "Encoder 输入源 ID，输出全部状态和 final hidden；Decoder 单步输入 token、hidden、Encoder outputs，输出 logits/new hidden/weights。"],
      ["3:47–6:22", "统一隐藏维", "最简实现让 Encoder H 与 Decoder H 相同，final hidden 直接初始化 Decoder。若不同，需桥接 Linear；双向 Encoder 还需合并方向。"],
    ]),
    points: ["接口先写 shape 再写代码", "Encoder/Decoder hidden 不同需桥接", "attention 连接全部 Encoder outputs"],
    code: commonCode,
    output: "统一 hidden_size=32，接口可直接连接。",
    pit: "双向 Encoder hidden 第一维含两个方向，不能直接当单向 Decoder 初始状态。",
    quiz: "为什么本例 Encoder/Decoder hidden 都设 32？", answer: "简化对接，使 final hidden 可直接传给 Decoder。"
  },
  {
    p: 98, slug: "teacher-forcing", title: "Teacher Forcing：训练时有时喂真值上一词",
    problem: "为什么教师强制能加快训练，又会带来训练/推理不一致？",
    chain: ["Decoder 预测本词", "抛随机数", "真值上一词或预测词", "作为下一输入", "重复到目标末尾"],
    extraVisuals: [{
      title: "Teacher Forcing 分支时序",
      diagram: `
flowchart TB
    A["得到本步 logits"] --> B{"随机数 < ratio？"}
    B -->|"是"| C["下一输入 = 真实 target_t"]
    B -->|"否"| D["下一输入 = argmax(logits)"]
    C --> E["进入下一解码步"]
    D --> E
`,
    }],
    article: article([
      ["0:00–4:50", "两种下一输入", "Teacher Forcing 使用真实目标上一词；自由运行使用模型自己的预测。比例 1 表示总用真值，0 表示从不使用。"],
      ["4:50–8:47", "为什么训练更稳", "早期模型预测很差，若错误不断传递，后面输入全乱；真值能让每个位置先学会局部下一词预测。"],
      ["8:47–11:35", "暴露偏差", "推理时没有真值，模型没充分见过自身错误会产生 exposure bias。可逐步降低比例（scheduled sampling），但策略也需验证。"],
    ]),
    points: ["Teacher Forcing 只用于训练", "ratio 是概率，不是固定前半句", "推理永远使用模型输出"],
    code: `for ratio in (1.0,.5,0.0):\n    print(ratio,"真值概率",ratio)`,
    output: "显示三个常见教师强制比例的含义。",
    pit: "不要在验证/推理指标中偷偷使用真实上一词。",
    quiz: "ratio=0.5 是否一定每句恰好一半真值？", answer: "不一定；通常每步随机决定，长期期望约一半。"
  },
  {
    p: 99, slug: "train-one-batch", title: "模型训练（单批次）：先把一次前向和损失走通",
    problem: "完整 epoch 前，怎样只用一批数据验证解码循环和损失对齐？",
    chain: ["Encoder 前向", "SOS 启动 Decoder", "循环 T-1 步", "累计/展平 logits", "CrossEntropy+反传"],
    extraVisuals: [trainingFlow],
    article: article([
      ["0:00–7:49", "一批数据先验形状", "source[B,S]，target[B,T]；Encoder outputs[B,S,H]；最终 logits 计划堆成 [B,T-1,Vt]。"],
      ["7:49–17:54", "解码循环", "decoder_input=target[:,0]（SOS），每步输出 logits，保存到列表；按 Teacher Forcing 决定下一输入。循环 step=1..T-1。"],
      ["17:54–27:42", "标签对齐", "预测第 1 个真实词对应 target[:,1]，所以标签是 target[:,1:]。PAD 目标位置应在损失中用 ignore_index。"],
      ["27:42–34:18", "单批反向", "把 logits 与标签展平为 [B(T-1),V] 和 [B(T-1)]，计算 CrossEntropyLoss，zero_grad、backward、梯度裁剪、step。先确认一批能稳定下降，再开完整训练。"],
    ]),
    points: ["目标第 0 位 SOS 不作为标签", "PAD 用 ignore_index", "先单批过拟合是强力调试方法"],
    code: `import torch\nlogits=torch.randn(2,4,10); labels=torch.randint(0,10,(2,4))\nloss=torch.nn.CrossEntropyLoss()(logits.reshape(-1,10),labels.reshape(-1))\nprint(loss.item())`,
    output: "把 batch×time 合并后得到一个标量交叉熵。",
    pit: "logits 与 target 错一位会让模型学习“预测当前输入”而非下一个词。",
    quiz: "target 长 5 时为什么只产生 4 个训练 logits？", answer: "第 0 位 SOS 是初始输入，预测位置是 1..4。"
  },
  {
    p: 100, slug: "view-function", title: "view()：只改观察形状，不改元素顺序",
    problem: "为什么训练损失前要把 [B,T,V] 变 [B×T,V]，view 有什么限制？",
    chain: ["原张量 [B,T,V]", "确认元素数不变", "展平前两维", "得到 [B*T,V]", "与标签 [B*T] 对齐"],
    extraVisuals: [{
      title: "损失前的展平关系",
      diagram: `
flowchart LR
    A["logits [B,T,V]"] --> B["reshape(-1,V)"]
    B --> C["[B×T,V]"]
    D["labels [B,T]"] --> E["reshape(-1)"]
    E --> F["[B×T]"]
    C --> G["CrossEntropyLoss"]
    F --> G
`,
    }],
    article: article([
      ["0:00–2:48", "元素总数守恒", "view/reshape 只改变维度解释，不改变数据。新形状各维乘积必须等于原元素数。"],
      ["2:48–5:26", "连续内存", "view 通常要求内存连续；transpose/permute 后可能报错，可先 contiguous().view 或直接 reshape。课程用 view 展平 logits 和标签。"],
    ]),
    points: ["V 类别维必须保留", "前两维展平为样本位置", "reshape 对非连续张量更方便"],
    code: `import torch\nx=torch.arange(24).reshape(2,3,4)\nprint(x.reshape(-1,4).shape)`,
    output: "从 [2,3,4] 变为 [6,4]，24 个元素不变。",
    pit: "不能把类别维 V 也压平后交给 CrossEntropyLoss。",
    quiz: "[2,5,10] 展平前两维后是什么？", answer: "[10,10]：10 个 token 位置，每个 10 类 logits。"
  },
  {
    p: 101, slug: "full-training", title: "完整训练代码：epoch、验证、保存与日志",
    problem: "单批训练正确后，怎样扩成可恢复、可比较的完整实验？",
    chain: ["epoch 循环", "train batches", "验证 no_grad", "记录损失/指标", "保存最佳 checkpoint"],
    extraVisuals: [trainingFlow, fullPipeline],
    article: article([
      ["0:00–8:55", "准备对象", "创建 Encoder/Decoder 或统一 Seq2Seq、优化器、CrossEntropyLoss(ignore_index=PAD)、DataLoader、device 与训练配置。"],
      ["8:55–18:51", "训练循环", "model.train；每批迁移设备；前向；展平损失；zero_grad/backward；clip_grad_norm_ 缓解循环网络梯度爆炸；step；累计按有效 token 归一化的损失。"],
      ["18:51–27:54", "验证循环", "model.eval 与 no_grad；Teacher Forcing 比例应明确，翻译质量最好用真实推理方式评估。不能在验证集反传。"],
      ["27:54–34:49", "检查点", "保存 model/optimizer state_dict、epoch、最佳指标、两套词表、模型超参数和随机状态，才能恢复训练/推理。"],
      ["34:49–36:53", "日志与结束", "记录每 epoch 的训练/验证损失和耗时。只保存最后一轮可能错过最佳泛化点，应按验证指标保存最佳。"],
    ]),
    points: ["验证不反传", "PAD 不计损失", "checkpoint 要包含词表与配置"],
    code: `# 核心骨架\nmodel.train()\noptimizer.zero_grad()\nlogits=model(source,target,teacher_forcing_ratio=.5)\nloss=criterion(logits.reshape(-1,logits.size(-1)),target[:,1:].reshape(-1))\nloss.backward()\ntorch.nn.utils.clip_grad_norm_(model.parameters(),1.0)\noptimizer.step()`,
    output: "完成一批可训练的 Seq2Seq 更新。",
    pit: "训练 loss 的分母若包含大量 PAD，不同 batch 的平均值不可比。",
    quiz: "为什么保存词表？", answer: "权重输出的是 ID；没有同一 id_to_token 就无法正确还原法语词。"
  },
  {
    p: 102, slug: "training-summary", title: "训练总结：把 800 行压缩成一条可复述主线",
    problem: "怎样用几分钟完整复述英译法训练数据流？",
    chain: ["source/target batch", "Encoder", "SOS+循环 Decoder", "Teacher Forcing", "loss/backward/checkpoint"],
    extraVisuals: [trainingFlow],
    article: article([
      ["0:00–3:43", "口头复述", "源句进 Encoder 得 outputs/hidden；Decoder 从 SOS 开始，每步对 outputs 做注意力并预测下一法语词；按 Teacher Forcing 选择下一输入；所有 logits 与 target 后移标签算损失；反传更新。"],
      ["3:43–4:08", "验收清单", "会说出每个张量形状、SOS/EOS/PAD 作用、训练/推理差异、注意力 weights 形状和保存内容，才算真正跑通。"],
    ]),
    points: ["先复述再看代码", "每个模块都能写输入输出 shape", "训练闭环最终由 loss 连接"],
    code: `print("source→Encoder→Attention Decoder→logits→shifted target loss→backward")`,
    output: "输出训练主线。",
    pit: "只记函数名但说不出张量语义，换数据就会崩。",
    quiz: "哪一个张量同时连接 Encoder 与每一步 Attention？", answer: "encoder_outputs，包含每个源位置的隐藏状态。"
  },
  {
    p: 103, slug: "prediction-code", title: "模型预测代码：无真值时逐词生成",
    problem: "推理阶段没有法语答案，怎样从一条英文句子生成 ID 序列和注意力矩阵？",
    chain: ["清洗/编码英文", "加载词表与权重", "Encoder", "SOS 循环 Decoder", "EOS 停止并 decode"],
    extraVisuals: [inferenceFlow],
    article: article([
      ["0:00–6:51", "准备输入与模型", "对英文使用训练时完全相同的清洗和 src 词表；创建同结构模型，加载 state_dict，eval/no_grad。"],
      ["6:51–13:48", "编码与启动", "source_ids 加 EOS、增加 batch 维、迁移 device；Encoder 得状态；decoder_input 初始化为 SOS。"],
      ["13:48–20:47", "逐步生成", "每步 logits.argmax 得 token ID，保存 ID 与 weights；若为 EOS 结束，否则作为下一输入。最大长度防止不结束。"],
      ["20:47–26:22", "还原文本", "用目标 id_to_token 解码，移除 SOS/EOS/PAD。未知输入映射 UNK。返回翻译与 [T,S] 注意力矩阵。"],
    ]),
    points: ["预测绝不使用目标真值", "eval+no_grad", "保存每步 attention"],
    code: `import torch\nfrom seq2seq_from_scratch.model import EncoderGRU,AttentionDecoderGRU,Seq2Seq\nm=Seq2Seq(EncoderGRU(20,6,8),AttentionDecoderGRU(25,7,8),start_id=1,end_id=2)\nids,w=m.greedy_decode(torch.randint(3,20,(1,5)),maximum_length=6)\nprint(ids.shape,w.shape)`,
    output: "预测 ID 为 [1,T]，注意力为 [1,T,5]。",
    pit: "加载权重前模型结构/词表大小不一致会报错或语义错位。",
    quiz: "为什么要返回注意力矩阵？", answer: "可检查每个目标词主要查看哪些源位置，并画热力图调试。"
  },
  {
    p: 104, slug: "prediction-test", title: "预测代码测试：从样例翻译发现数据与模型问题",
    problem: "翻译出一句话后，怎样判断问题来自模型、词表还是预处理？",
    chain: ["固定测试样例", "打印 source IDs", "打印预测 IDs", "还原文本", "查看 attention/错误类型"],
    extraVisuals: [inferenceFlow],
    article: article([
      ["0:00–5:55", "固定样例", "老师加载训练权重，对若干英文句子预测法语。测试句应同时包含训练分布内样例和未见组合。"],
      ["5:55–12:54", "逐层排错", "若输出全 UNK，先查词表；立即 EOS，查训练与结束标签；重复词，查解码循环/训练不足；注意力全落 PAD，查 mask。"],
      ["12:54–18:50", "不要只看一个成功例", "至少报告验证集损失和翻译指标，并人工抽样。贪心解码不是唯一策略，beam search 可保留多个候选但更慢。"],
      ["18:50–22:36", "注意力图", "目标步×源位置的矩阵应与两句 token 标签对齐。热力图只是模型内部对齐线索，不是翻译正确性的证明。"],
    ]),
    points: ["固定回归样例", "按数据→词表→模型→解码逐层排错", "单例成功不代表泛化"],
    code: `checks=["是否立即EOS","是否重复","是否全UNK","是否关注PAD"]\nfor x in checks: print("-",x)`,
    output: "打印预测调试清单。",
    pit: "拿训练句预测很漂亮，可能只是记忆，不代表新句有效。",
    quiz: "注意力集中正确但翻译错，可能吗？", answer: "可能；对齐只是中间过程，目标词分类层或语言建模仍可能出错。"
  },
  {
    p: 105, slug: "tensorboard-graph", title: "绘制张量图：用图检查模块连接，不把图当性能证明",
    problem: "TensorBoard 计算图能帮助看什么，又不能说明什么？",
    chain: ["准备样例输入", "记录模型图/标量", "启动 TensorBoard", "检查模块连接", "结合指标与样例调试"],
    extraVisuals: [architecture],
    article: article([
      ["0:00–5:52", "SummaryWriter", "老师创建 writer，准备符合 forward 签名的样例张量，并尝试 add_graph。复杂循环、随机 Teacher Forcing 或多返回值可能让 tracing 失败，可单独包装确定性前向。"],
      ["5:52–11:48", "图怎样看", "检查 Embedding→Encoder GRU→Attention Decoder→Linear 的模块连接和张量流；同时记录 loss、学习率、梯度范数。"],
      ["11:48–17:45", "边界", "计算图能发现错接、维度和模块复用，但不能证明翻译好。性能仍需验证集、翻译指标、人工检查和错误分析。日志目录要分实验，避免多次运行混在一起。"],
    ]),
    points: ["图用于结构调试", "随机控制流可能不易 tracing", "不同实验用不同日志目录"],
    code: `# 示例：writer.add_scalar("train/loss", loss.item(), step)\nprint("记录结构 + 标量 + 配置，三者一起看")`,
    output: "提示 TensorBoard 的正确用途。",
    pit: "看到漂亮计算图不等于模型学会翻译。",
    quiz: "计算图最适合发现哪类问题？", answer: "模块是否正确连接、张量是否流经预期层，以及部分形状/复用问题。"
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
