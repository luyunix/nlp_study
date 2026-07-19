import { generateTopic } from "./course_note_helpers.mjs";

const article = (sections) => sections.map(([time, title, body]) =>
  `### ${time}　${title}\n\n${body}`
).join("\n\n");

const overview = {
  title: "FastText 文本分类总流程",
  diagram: `
flowchart LR
    A["带标签文本"] --> B["清洗与统一格式"]
    B --> C["词与 word N-gram 特征"]
    C --> D["嵌入查表并求平均"]
    D --> E["线性分类层"]
    E --> F["softmax / hs / ova"]
    F --> G["标签与概率"]
`,
};

const architecture = {
  title: "三层结构与数据形状",
  diagram: `
flowchart LR
    A["一条文本：若干 token / N-gram ID"] --> B["Embedding 查表：每项得到 [D]"]
    B --> C["平均池化：整句得到 [D]"]
    C --> D["线性层：得到 [C]"]
    D --> E["损失函数：输出标签概率"]
`,
};

const huffman = {
  title: "层次 Softmax 的哈夫曼路径",
  diagram: `
flowchart TB
    R["根节点：做一次二分类"] -->|"0"| L["左子树"]
    R -->|"1"| H["右子树"]
    H -->|"0"| C["继续二分类"]
    H -->|"1"| A["高频标签 A"]
    C -->|"0"| D["低频标签 D"]
    C -->|"1"| B["标签 B"]
`,
};

const negativeSampling = {
  title: "全量更新与负采样对比",
  diagram: `
flowchart TB
    X["正样本：目标词/标签"] --> ALL["全量方法：正样本 + 所有负样本"]
    X --> NS["负采样：正样本 + 少量抽中的负样本"]
    ALL --> W1["更新几乎整张输出权重表"]
    NS --> W2["只更新本次相关的少量行"]
`,
};

const tuning = {
  title: "调优实验闭环",
  diagram: `
flowchart LR
    A["固定训练/验证划分"] --> B["只改一个参数"]
    B --> C["重新训练"]
    C --> D["记录 Precision / Recall / 耗时"]
    D --> E{"验证集更好吗？"}
    E -->|"是"| F["保留候选配置"]
    E -->|"否"| G["回退并试下一项"]
    F --> B
    G --> B
`,
};

const multilabel = {
  title: "多标签预测时序",
  diagram: `
sequenceDiagram
    participant U as 一条文本
    participant M as OVA 分类器
    participant T as threshold 过滤
    participant K as top-k 截断
    U->>M: 文本特征
    M-->>T: 每个标签各自的概率
    T-->>K: 仅保留超过阈值的标签
    K-->>U: 概率最高的至多 k 个标签
`,
};

const apiUml = {
  title: "训练、预测、保存 API 关系",
  diagram: `
classDiagram
    class fasttext {
      train_supervised(input, ...)
      load_model(path)
    }
    class FastTextModel {
      predict(text, k, threshold)
      test(path)
      save_model(path)
      get_nearest_neighbors(word)
    }
    fasttext ..> FastTextModel : 创建或加载
`,
};

const basicFasttextCode = `try:
    import fasttext
except ImportError:
    raise SystemExit("请先在独立环境安装 fasttext；安装方式以当前官方说明为准")

model = fasttext.train_supervised(input="data/train.txt")
print(model.test("data/valid.txt"))`;

const lessons = [
  {
    p: 144, slug: "fasttext-introduction", title: "FastText 简介与环境：为什么一行 API 也值得学原理",
    problem: "已经会写 RNN 和 Transformer，为什么还要学一个调用很短的文本分类工具？",
    chain: ["回顾复杂训练代码", "认识 FastText", "比较速度与能力", "拆出三层结构", "准备运行环境"],
    extraVisuals: [overview, architecture, apiUml],
    article: article([
      ["0:00–2:47", "从“代码很长”引出 FastText", "老师先把本章放进迁移学习专题：FastText 工具与原理、文本分类、词向量迁移。此前用 GRU、RNN、LSTM 或 Transformer 自己搭模型，数据、网络、损失和训练循环会写很多代码；FastText 把常用流程封装成了 `train_supervised` 等 API。调用短不表示模型没有原理，而是工程细节已经由库实现。老师也回顾了 one-hot、Word2Vec 的 CBOW/Skip-gram 与 embedding，提醒大家 FastText 还能训练词向量。"],
      ["2:47–5:30", "三条加速线索", "老师预告三块：简单的线性结构、层次 Softmax、负采样。为了让初学者先有直觉，他把负采样比作考试前优先复习错题与重点，而不是从第一本书第一页重新看完。这个类比只说明“只处理一小部分负例”；真正的抽样分布和损失计算要到 P146 再说。"],
      ["5:30–10:55", "用途、优点与局限", "FastText 主要可用于文本分类和词向量学习。文本分类包括新闻类别和情感倾向。它的优势是训练、预测快，API 简单，在许多基线任务上能保持有竞争力的精度。老师强调“较高”是相对说法：数据没清洗、参数用默认值时，精度可能很低，后面仍需调优。局限是把局部特征汇总后做线性分类，难以细致表示长距离词序；加入 word N-gram 可以缓解，却不能等同于 RNN 或 Transformer 的上下文建模。"],
      ["10:55–16:35", "安装演示与必要纠错", "老师现场展示不同 Python 环境中的安装，并提醒二进制构建可能受系统、编译器和 Python 版本影响。学习时应新建隔离环境，先验证 `import fasttext`，再跑最小数据。还要纠正一句口误：FastText 分类器的底层不是 Transformer；它更接近“嵌入查表 → 特征平均 → 线性分类”。另外，层次 Softmax 与负采样通常是不同的损失/加速选项，不是必须同时启用的同一过程。"],
    ]),
    points: ["FastText 是强基线，不是 Transformer 的简写", "API 简单与原理简单是两回事", "先跑默认基线，再用验证集调优"],
    code: basicFasttextCode,
    output: "`train_supervised` 返回模型；`test` 通常返回测试样本数、Precision@1、Recall@1。",
    pit: "直接复制旧安装命令和版本号；环境变化后可能无法构建。隔离环境并按当前官方说明安装。",
    quiz: "FastText 为什么快，但不擅长长距离词序？", answer: "它主要对词与 N-gram 嵌入做求和/平均，再接线性分类；计算并行而简单，但细致的顺序结构在汇总时会丢失。"
  },
  {
    p: 145, slug: "hierarchical-softmax-huffman", title: "层次 Softmax 与哈夫曼树：从全部类别改成走一条路径",
    problem: "类别多到几千、几万时，怎样避免每次都为所有类别计算完整 Softmax？",
    chain: ["文本拆成 N-gram", "嵌入求和/平均", "按频次建哈夫曼树", "沿目标路径做二分类", "路径概率相乘"],
    extraVisuals: [architecture, huffman],
    article: article([
      ["0:00–1:53", "先看 FastText 三层结构", "输入层把文本表示成词和 word N-gram；每个离散特征查到一个 D 维向量。隐藏层把这些向量求和或平均，得到整段文本的 D 维特征；输出层再根据这个特征预测标签。若一条文本有 6 个特征、嵌入维度 D=100，查表结果可理解为 `[6,100] = 6 个特征 × 每个 100 维`，平均后是 `[100]`，分类到 C 个标签后是 `[C]`。"],
      ["1:53–5:43", "老师的猜数字类比", "普通 Softmax 会为 C 个类别都计算一个 logit 并统一归一化；C 很大时成本高。老师用“猜 1 到 100 的数字”类比：逐个猜像遍历全部类别，不断判断前半/后半则只沿一条分支走。这个类比帮助理解树形决策，但哈夫曼树不是普通的平衡二分查找树：它按类别频率安排路径，高频类别更靠近根，路径更短。"],
      ["5:45–8:40", "什么是带权路径长度", "叶子结点表示标签，权重可以看作标签出现频次，深度是从根走到该叶子的边数。这里先说明公式解决什么问题：我们要衡量“常见标签走多远”的总成本。带权路径长度是 `WPL = Σ(叶子权重 × 叶子深度)`。哈夫曼算法让 WPL 尽量小，因此高频标签通常更靠近根。老师用权重 9、7、3、5 算出一个示例 WPL，并强调学习重点是会构树、会读路径。"],
      ["8:45–21:41", "手工构造哈夫曼树", "老师按 3、5、7、9 演示：每轮取当前最小的两个权重合并；3+5=8，集合变成 7、8、9；7+8=15，集合变成 9、15；9+15=24 成为根。较小节点放左、较大放右是课堂约定，用来稳定编码；左记 0、右记 1 也只是约定，反过来同样可以，只要训练与推理一致。于是每个标签得到一串 0/1 路径。"],
      ["21:41–30:09", "路径概率怎样代替全量 Softmax", "要预测某个叶子，只计算根到它路径上的几个二分类概率。这里先说明公式解决什么问题：我们要把“到达目标标签”的概率拆成沿途每次选左/右的概率。若路径编码为 `1,1,0`，则 `P(label|x)=P(1|x)×P(1|x)×P(0|x)`；每个内部节点用 sigmoid 做二分类。训练只更新这条路径相关节点。高频标签路径短，所以平均计算量下降。老师最后复盘：模型结构、层次 Softmax、负采样是理解 FastText 快的三条线，下一节继续负采样。"],
    ]),
    points: ["哈夫曼树按频率缩短高频标签路径", "0/1 是编码约定，不是正负的天然含义", "层次 Softmax 只计算目标路径上的内部节点"],
    code: `from fasttext_from_scratch.huffman import huffman_codes\nprint(huffman_codes({"A": 5, "B": 9, "C": 7, "D": 3}))`,
    output: "打印每个标签的 0/1 编码；权重最大的标签通常拥有更短编码。具体左右编码可能不同，但码长性质一致。",
    pit: "把哈夫曼树当成必然平衡的二分搜索树。它优化的是带权路径长度，不保证左右子树节点数相等。",
    quiz: "1000 个标签时，层次 Softmax 为什么不必计算 1000 个输出？", answer: "目标标签对应树上的一片叶子，只需计算根到该叶子的若干次二分类，理想量级接近树高。"
  },
  {
    p: 146, slug: "negative-sampling", title: "负采样：正样本之外，只挑少量“陪练”负样本",
    problem: "词表有一万个词时，为什么每次只更新目标词和少量负词就能大幅省计算？",
    chain: ["确定正样本", "排除正样本", "按分布抽负样本", "计算二分类损失", "只更新相关权重"],
    extraVisuals: [negativeSampling],
    article: article([
      ["0:00–2:58", "全量 Softmax 的负担", "老师从 Skip-gram 的输出层切入：若词表 V=10000，一次训练要产生一万个概率，还可能更新一大块输出权重。类比考试复习，如果时间有限，不会从第一天所有知识点重学，而会盯住考点和错题。负采样的核心也是不处理所有“这次不是答案”的词。"],
      ["2:58–6:56", "正样本与负样本", "例子中输入 hello、目标 world（音轨把它识别成了其他词）。目标词是正样本，期望二分类输出为 1；其余 V-1 个词都是候选负样本，期望输出为 0。负采样从候选中抽 K 个，只计算 1 个正样本加 K 个负样本。若隐藏维度 D=300、V=10000，全量输出矩阵有 `300×10000=3,000,000` 个参数位置；K=5 时本步涉及 `300×(1+5)=1,800` 个位置。这里比较的是本步需要访问和更新的权重规模，不表示模型总参数被删除。"],
      ["6:56–13:53", "老师的错题本类比", "传统方法像每次考试都复习整本书；负采样像复习本次考点，再加几道容易混淆的错题。抽入的负例像噪声或“陪练”，迫使模型把真正目标与其他词分开。负样本不是随便永久删掉的数据：每个训练步会重新抽样，长期看不同词仍可能被抽到。"],
      ["13:53–19:39", "完整训练步骤", "先把正样本标 1；从排除正样本后的词表按采样分布抽 K 个负词并标 0；对这 K+1 个候选计算 sigmoid 二分类损失；反向传播只更新输入相关向量和这些输出行。小数据常可多抽一些负例，大数据通常每个正例配少量负例，具体 K 需要验证。"],
      ["19:39–26:32", "与层次 Softmax 的关系和课堂复盘", "层次 Softmax 把类别概率改写成路径概率乘积；负采样则把全量多分类近似成若干个正/负二分类。两者都减少单步计算，但机制不同，在 FastText 配置里通常是不同 `loss` 选项，不应理解成“选 hs 后又自动叠加 ns”。老师用选择题复盘：负采样主要解决计算成本，负例应从正样本之外的候选集合按规定分布抽取。"],
    ]),
    points: ["正样本一定保留，负样本只抽一小批", "未抽中的权重本步不更新，不等于永远不训练", "负采样与层次 Softmax 是两种不同加速思路"],
    code: `from fasttext_from_scratch.sampling import sample_negatives\nvocab={"world":100, "book":60, "apple":30, "river":10}\nprint(sample_negatives(vocab, positive="world", k=2, seed=7))`,
    output: "得到两个不含 `world` 的负样本；固定 seed 时结果可复现。",
    pit: "从候选集中又抽到正样本，或把 `hs` 与 `ns` 当成同时必开的两层功能。",
    quiz: "未被抽中的 9994 个词在这一步发生什么？", answer: "它们不参与本步损失，相关输出权重不更新；后续训练步仍可能被抽中。"
  },
  {
    p: 147, slug: "direct-training", title: "直接训练：数据格式、预测、测试与第一版基线",
    problem: "怎样用最少代码建立一个可测量的文本分类基线，并正确读懂返回结果？",
    chain: ["识别分类类型", "准备 __label__ 数据", "train_supervised", "predict 单条", "test 验证集"],
    extraVisuals: [overview, apiUml, tuning],
    article: article([
      ["0:00–4:51", "先分清三类任务", "老师先区分二分类、单标签多分类、多标签多分类。单标签多分类表示一条文本只属于 C 类中的一个，常用 Softmax 多类交叉熵；多标签表示多个标签可同时为真，可看成 C 个独立二分类，常用 sigmoid/BCE 思路。在 FastText 中后者可用 OVA 损失。这里的“拆成多个二分类”主要是目标函数层面的理解，不一定要把一行文本真的复制 C 份。"],
      ["4:51–14:43", "数据文件与标签格式", "案例数据一行包含标签与文本，标签必须带前缀 `__label__`。例如 `__label__sport 球队 赢得 决赛`。多标签样本可在一行放多个 `__label__...`。老师展示了未清洗与已清洗的训练/验证文件，并解释原数据约一万多行、标签数很多；分类训练属于监督学习，所以必须有标签。"],
      ["14:43–19:42", "一行训练与默认参数", "`fasttext.train_supervised(input=...)` 读取整个训练文件并返回模型。老师进入源码查看默认参数，训练轮数、学习率、维度和损失等都可覆盖。课堂重点不是背下所有默认值，而是知道如何查看当前安装版本的函数帮助；旧视频里的默认值可能随版本变化。"],
      ["19:42–28:34", "预测与 test 输出", "`model.predict(text)` 返回标签和概率；`model.test(valid_path)` 返回 `(N, P@1, R@1)`。N 是评估样本数量，P@1 是预测的第一个标签有多可靠，R@1 是真实标签被第一个预测覆盖的比例。在单标签任务里两者常接近；多标签任务里含义会不同。训练日志中的 words、labels、progress、words/sec/thread、lr、loss、eta 也被老师逐项讲解。"],
      ["28:34–31:59", "低精度不是失败，而是基线", "默认参数加未清洗数据得到的指标很低，但速度确实快。老师由此列出优化路线：清洗数据、增加 epoch、调整学习率、加入 word N-gram、替换损失、自动调参、处理多标签、保存加载。正确实验方法是固定验证集、每次只改一项并记录结果，不能只挑一条训练样本看预测是否“像对的”。"],
    ]),
    points: ["每个标签以 `__label__` 开头", "`predict` 看个例，`test` 才看整体", "第一版默认结果是后续比较的基线"],
    code: basicFasttextCode,
    output: "`test` 的三元组分别是样本数、P@1、R@1；不要把第一个整数误认为准确率。",
    pit: "拿训练文件当验证集，或只凭三条单例预测宣称模型已经很好。",
    quiz: "为什么建立低分基线仍有价值？", answer: "它提供固定参照；后续每项清洗或调参是否有效，都能用同一验证集与基线比较。"
  },
  {
    p: 148, slug: "data-preprocessing", title: "数据预处理：统一大小写、分开标点，并保持训练预测一致",
    problem: "为什么人眼觉得相同的 `Apple`、`apple`、`apple?` 会被模型当成三个特征？",
    chain: ["检查原始样本", "统一大小写", "标点与单词分离", "写出新数据文件", "预测使用同一规则"],
    extraVisuals: [overview, tuning],
    article: article([
      ["0:00–1:55", "原始数据的问题", "老师展示大小写和标点混杂的数据。程序按 token/字符串识别特征，因此 `Apple`、`apple`、`apple?` 默认是不同项，会分散本来属于同一词的统计量。课堂处理是全部转小写，并在词与标点间加入空格，让标点成为独立 token。"],
      ["1:55–4:52", "更换清洗后的训练与验证文件", "老师复制上一节代码，只替换为处理后的 train/valid 文件；用于单条预测的文本也必须做相同清洗。如果训练时小写、预测时仍保留大写和粘连标点，模型会看到大量未登录或低频特征。工程上应该把清洗封装成同一个函数，训练、验证、推理共同调用，而不是手工按快捷键修改。"],
      ["4:52–6:55", "比较结果与继续调优", "清洗后课堂指标从约 0.14 提升到约 0.17，说明方向有效，但具体数字只对该次数据划分与配置成立。老师强调 FastText 虽快，但默认精度不一定高，预处理只是第一步；下一节继续 epoch 和学习率。还要注意：不是标点越少越好，情感任务中的 `!`、`?` 可能有信息，应通过验证实验决定保留还是独立分词。"],
    ]),
    points: ["模型按离散特征区分大小写与标点", "训练和推理必须复用同一清洗函数", "清洗规则要服务任务，不应盲目删除信息"],
    code: `from fasttext_from_scratch.data import normalize_text, format_labeled_line\nprint(normalize_text("Apple,  APPLE?"))\nprint(format_labeled_line(["fruit"], "Apple,  APPLE?"))`,
    output: "文本被统一为小写，标点与词分开；第二行生成合法的 `__label__fruit ...` 格式。",
    pit: "只清洗训练集，不清洗验证集和线上输入；这会造成训练—推理分布不一致。",
    quiz: "为什么不建议把所有标点直接删除？", answer: "标点有时携带语气和边界信息；先分离，再用验证集决定是否保留更稳妥。"
  },
  {
    p: 149, slug: "epochs-learning-rate", title: "训练轮数与学习率：一个决定看几遍，一个决定每步走多远",
    problem: "epoch 和学习率都能改变训练效果，它们分别控制什么，为什么不能机械地越大或越小？",
    chain: ["固定清洗数据", "增加 epoch", "观察耗时与指标", "调整 lr", "组合验证"],
    extraVisuals: [tuning],
    article: article([
      ["0:00–4:48", "把 epoch 从默认值增大", "老师复制清洗后的基线，只加入 `epoch=25`。epoch 表示把训练集完整学习多少轮；轮数增加通常让模型有更多机会拟合模式，也会线性增加训练时间。课堂结果明显提升，但这不证明 25 永远最佳：继续增大可能收益变小甚至过拟合。"],
      ["4:48–7:45", "如何确认默认参数", "老师再次进入 `train_supervised` 的定义，提醒监督训练可能在通用参数基础上覆盖默认值，应以当前函数签名和帮助为准。这一段的学习目的不是背 `0.05` 或 `0.1`，而是学会分清“通用默认”和“监督任务覆盖值”。"],
      ["7:45–9:15", "学习率像下山步长", "课堂把 `lr` 调大到 1 并观察指标。学习率控制一次参数更新走多远：太小像每走两步就重新看方向，稳但慢；太大像跨很大步，收敛快却可能越过低谷、震荡或发散。老师要求自己尝试大、小值并计时，而不是背“越小越好”。正确做法是同时记录训练损失、验证指标与耗时，再选组合。"],
    ]),
    points: ["epoch 是遍历数据的轮数", "lr 是每次更新的步幅", "二者都要看验证集，不能单调追大或追小"],
    code: `import fasttext\nmodel=fasttext.train_supervised(\n    input="data/train.clean.txt", epoch=25, lr=0.2\n)\nprint(model.test("data/valid.clean.txt"))`,
    output: "在同一清洗数据上以 25 轮、0.2 学习率训练并输出验证指标。",
    pit: "调高 epoch 后又换数据集、换划分，却把指标变化都归因于 epoch。",
    quiz: "为什么学习率很小不一定更好？", answer: "有限训练轮数内可能尚未收敛；太小会让每步改动过弱，模型还没学到合适参数就停止。"
  },
  {
    p: 150, slug: "ngrams-loss", title: "word N-gram 与损失函数：补一点局部词序，再选合适输出方式",
    problem: "平均词向量会忽略词序，怎样用局部短语特征缓解，并在 softmax、hs 等损失间选择？",
    chain: ["加入 unigram", "生成相邻 bigram", "共同嵌入平均", "选择 loss", "比较验证指标"],
    extraVisuals: [architecture, tuning, huffman],
    article: article([
      ["0:00–3:52", "从 unigram 到 bigram", "`wordNgrams=2` 表示除单词本身外，再加入相邻两个词组成的特征。例如“我 爱 武汉”包含 `我`、`爱`、`武汉`，还加入 `我_爱`、`爱_武汉`。这样“我爱”和“爱我”不再完全一样。课堂建议通常从 2 开始，3-gram 特征更多、训练更慢且更稀疏，需要自己验证。"],
      ["3:52–4:51", "调参没有脑内最优解", "老师强调参数效果必须实测：加特征有可能提升，也可能因数据少、稀疏或过拟合下降。可靠做法是固定训练/验证划分，一次改一项，记录指标和运行成本，再尝试少量有依据的组合。"],
      ["4:51–8:49", "从普通 Softmax 切到层次 Softmax", "课堂把 `loss` 从默认 Softmax 改为 `hs`，一行参数即可启用前面讲的层次 Softmax。普通 Softmax 适合类别量不太大且需要完整归一化的场景；`hs` 在类别很多时通过树路径降低计算。这里纠正老师口头混在一起的一点：`hs` 是 hierarchical softmax，`ns` 才是 negative sampling；二者是不同 loss 选项。"],
      ["8:49–9:32", "参数要组合验证", "切到 `hs` 后还可以重新搜索学习率、epoch 和 N-gram。某个损失在一组参数上更好，不代表所有数据都更好。原理帮助你缩小搜索方向，验证集决定最终选择。"],
    ]),
    points: ["wordNgrams=2 会保留相邻两个词的局部顺序", "`hs` 与 `ns` 是不同损失选项", "更多 N-gram 会增加特征稀疏度和计算"],
    code: `import fasttext\nmodel=fasttext.train_supervised(\n    input="data/train.clean.txt", epoch=25, lr=0.2,\n    wordNgrams=2, loss="hs"\n)\nprint(model.test("data/valid.clean.txt"))`,
    output: "训练一个包含 unigram/bigram 且使用层次 Softmax 的分类器。",
    pit: "把 `wordNgrams=2` 理解成只保留 bigram。FastText 通常同时使用 1 到 N 阶 word N-gram。",
    quiz: "加入 bigram 后，“我爱你”和“你爱我”为什么更容易区分？", answer: "两句的单词集合相同，但相邻二元组分别包含 `我_爱/爱_你` 与 `你_爱/爱_我`。"
  },
  {
    p: 151, slug: "autotune", title: "自动调参：给验证集和时间预算，让程序寻找更好组合",
    problem: "参数很多时，怎样减少手工复制代码，同时避免只在训练集上挑出“看起来最好”的配置？",
    chain: ["准备训练集", "准备独立验证集", "设置时间预算", "自动搜索", "在未参与调参的测试集复核"],
    extraVisuals: [tuning],
    article: article([
      ["0:00–2:57", "为什么需要自动搜索", "手动组合 epoch、lr、N-gram、loss 很累，也容易漏掉更优区域；网格搜索仍需要人为给范围。FastText 提供自动调参接口：传入验证集 `autotuneValidationFile`，再用 `autotuneDuration` 指定搜索时间。课堂用“老板只说三天后交结果，不规定每一步”来类比时间预算。"],
      ["2:57–5:43", "三个关键参数", "`input` 是训练文件；`autotuneValidationFile` 是用于比较候选模型的验证文件；`autotuneDuration` 以秒计，是搜索预算。时间变长通常能探索更多候选，但不是无限越长越好：验证集可能被反复选择而间接过拟合，计算成本也持续增加。"],
      ["5:43–10:18", "等待、保存与迁移学习预告", "自动调参会持续到预算结束，课堂短预算结果甚至比手调略低，这很正常。真正实验应在自动搜索后，用从未参与搜索的测试集做一次最终评估，并立即保存模型，避免下次重复搜索。老师借等待时间预告迁移学习：后面会直接加载别人预训练的模型完成完形填空、阅读理解、摘要、分类等任务。"],
    ]),
    points: ["训练集拟合参数，验证集挑配置，测试集只做最终报告", "时间预算不是质量保证", "自动搜索完成后要保存模型和参数记录"],
    code: `import fasttext\nmodel=fasttext.train_supervised(\n    input="data/train.clean.txt",\n    autotuneValidationFile="data/valid.clean.txt",\n    autotuneDuration=180,\n)\nmodel.save_model("models/autotuned.bin")`,
    output: "最多搜索约 180 秒，返回自动选择配置后的模型并保存。",
    pit: "用测试集作为 `autotuneValidationFile`；这样测试结果已经参与选参，不再是公正的最终评估。",
    quiz: "自动调参完成后为什么还需要独立测试集？", answer: "验证集被反复用于挑选配置，可能产生选择偏差；独立测试集才用于估计未知数据表现。"
  },
  {
    p: 152, slug: "multilabel-ova", title: "多分类多标签：OVA、k 与 threshold 的完整含义",
    problem: "一段文本能同时属于多个标签时，怎样训练并控制最终返回哪些标签？",
    chain: ["每个标签独立判断", "OVA 训练", "得到全部标签概率", "threshold 过滤", "top-k 截断"],
    extraVisuals: [multilabel, tuning],
    article: article([
      ["0:00–3:52", "把多标签看成多个二分类", "一条文本可同时有“体育”“武汉”“人物”等标签。OVA 是 one-vs-all：针对每个标签都判断“是/不是”。训练文件一行可以带多个 `__label__...`；使用 `loss='ova'` 后，模型为各标签学习相对独立的二分类分数。课堂说“拆成多条二分类”是理解目标的方式，FastText 内部不要求你手工复制成 C 行。"],
      ["3:52–6:48", "`k` 和 `threshold` 先后怎样作用", "`model.predict(text, k=3, threshold=0.5)` 可按两个条件筛选。先理解 threshold：概率未达到阈值的标签直接排除；再理解 k：在剩余候选中至多返回概率最高的 k 个。所以写 k=3 并不保证一定返回 3 个，只有一个标签过阈值就只返回一个。`k=-1` 表示不设数量上限，但仍应配合理性阈值。"],
      ["6:48–11:33", "老师现场改阈值验证", "课堂先看到 k=3 只返回一个标签，因为另外两个概率没过 0.5；把阈值降到 0.04 后返回两个，第三个约 0.03 仍被过滤；若再把 k 改 1，则只保留这两个中的最高者。这个实验准确展示了“阈值负责资格，k 负责名额”。阈值不应凭感觉固定 0.5，应依据每类 Precision/Recall 或业务代价在验证集上选。"],
    ]),
    points: ["OVA 为每个标签做独立的是/否判断", "threshold 决定能不能入围，k 决定最多留几个", "多标签评估不应只看 Accuracy"],
    code: `import fasttext\nmodel=fasttext.train_supervised(\n    input="data/train.multilabel.txt", epoch=25, lr=0.2,\n    wordNgrams=2, loss="ova"\n)\nlabels, probs = model.predict("sample text", k=3, threshold=0.5)\nprint(list(zip(labels, probs)))`,
    output: "返回概率达到 0.5 的候选中至多 3 个标签，可能是 0、1、2 或 3 个。",
    pit: "认为 `k=3` 会强制凑满三个标签；真正数量还受 threshold 限制。",
    quiz: "候选概率为 0.93、0.40、0.03，k=3、threshold=0.5 时返回几个？", answer: "只返回 1 个，因为只有 0.93 达到阈值；k 只是上限。"
  },
  {
    p: 153, slug: "save-load-model", title: "保存与加载模型：训练一次，部署和复现实验反复使用",
    problem: "调参跑了一小时，如果不想下次重新训练，模型、词表和配置应怎样保存与验证？",
    chain: ["训练或自动调参", "save_model 写入磁盘", "load_model 恢复", "同样输入再预测", "记录版本与指标"],
    extraVisuals: [apiUml],
    article: article([
      ["0:00–2:58", "为什么必须保存", "老师从自动调参可能耗时一小时切入：训练结束不保存，下次使用就得全部重跑。先创建模型目录，调用 `model.save_model('models/fasttext.bin')`。扩展名并不是 PyTorch 固定的 `.pth`；FastText 常用 `.bin`，关键是使用库对应的保存/加载接口。"],
      ["2:58–5:42", "加载后必须用新对象验证", "用 `fasttext.load_model(path)` 得到新对象 `model2`，随后应调用 `model2.predict(...)`，而不是误用内存里的旧 `model`。老师现场纠正了这个变量细节，并检查保存目录确实出现文件。更可靠的验收是：同一输入在保存前后标签与概率一致或在容许误差内一致。"],
      ["5:46–9:30", "专题总结与课堂题", "老师复盘 `predict`、`test`、`save_model`、`load_model`，以及 epoch、lr、wordNgrams、loss、autotune。文本分类包括二分类、单标签多分类、多标签多分类；多标签可用 OVA。课堂题强调：N-gram 通过 `wordNgrams` 配置，保存用 `save_model`，多标签用 `ova`。工程上还应同时保存清洗代码、标签说明、训练数据版本和指标，否则只有二进制文件仍无法解释模型。"],
    ]),
    points: ["保存的是训练成果，不只是 Python 对象变量", "加载后用新对象做回归预测", "模型文件必须和预处理、标签表、版本记录一起管理"],
    code: `import fasttext\nmodel=fasttext.train_supervised(input="data/train.clean.txt")\nbefore=model.predict("sample text")\nmodel.save_model("models/classifier.bin")\nloaded=fasttext.load_model("models/classifier.bin")\nafter=loaded.predict("sample text")\nprint(before, after)`,
    output: "打印保存前和加载后的预测，二者应一致。",
    pit: "加载到 `loaded` 后仍调用旧变量 `model`，导致误以为加载验证成功。",
    quiz: "为什么只保存 `.bin` 还不够？", answer: "模型依赖同一套清洗、标签语义和库版本；缺少这些信息，线上输入可能不一致，也难以复现实验。"
  },
  {
    p: 154, slug: "pretrained-vectors-transfer", title: "词向量迁移：把别人训练好的语义坐标系拿来起步",
    problem: "自己的标注数据很少时，怎样复用大语料上已经学到的词向量，而不是从随机数开始？",
    chain: ["选择同语言预训练向量", "下载并校验", "解压/加载", "检查近邻", "用于下游任务或继续训练"],
    extraVisuals: [apiUml, overview],
    article: article([
      ["0:00–1:47", "回顾三类词表示", "老师回顾 one-hot、Word2Vec 与 FastText 词向量，并介绍官方提供多语言预训练向量。迁移的核心是：别人先在大规模语料上学出“词在语义空间中的位置”，我们下载模型或向量后直接使用，或作为自己任务的初始化。大文件下载前要同时考虑压缩包、解压文件和最终模型占用，不能只看下载大小。"],
      ["1:52–3:39", "用最近邻做直观检查", "加载后可用 `get_nearest_neighbors(word)` 查看最相似的 K 个词。老师回忆此前用某个词找到同类犬种的例子，并因前面 Word2Vec 专题已详细演示而不重复。近邻只能做直观 sanity check，不是下游任务质量的正式评估；错别字、领域词和语言不匹配都可能让结果失真。"],
      ["3:39–4:24", "下载、加载与继续训练", "课堂把流程压缩为下载、解压、`load_model`、查看近邻或继续训练。迁移并不保证更好：预训练语料与当前领域差异太大时会产生负迁移。还要区分两种文件：可直接 `load_model` 的 FastText 二进制模型，与文本格式 `.vec` 词向量不是同一种加载方式。老师说明这一节是复习，随后正式进入迁移学习。"],
    ]),
    points: ["迁移学习复用的是已经学到的参数/表示", "二进制模型与 `.vec` 文本向量加载方式不同", "同语言、同领域通常更容易迁移"],
    code: `import fasttext\nmodel=fasttext.load_model("models/pretrained.bin")\nfor score, word in model.get_nearest_neighbors("语言", k=5):\n    print(f"{word}: {score:.3f}")`,
    output: "打印“语言”的 5 个近邻及余弦相似度；实际结果取决于预训练模型。",
    pit: "把任意 `.vec` 路径传给 `load_model`，或认为预训练模型在所有领域都必然提升。",
    quiz: "词向量迁移为什么可能失败？", answer: "源语料与目标语料的词义、语言或领域差异过大时，预训练表示会带来不合适的先验，即负迁移。"
  },
];

generateTopic({
  topic: "FastText 文本分类",
  folder: "fasttext-classification",
  lessons,
  stages: [
    { title: "一、原理：模型为什么快", start: 1, end: 3 },
    { title: "二、基线与数据", start: 4, end: 5 },
    { title: "三、调优与多标签", start: 6, end: 9 },
    { title: "四、保存与迁移", start: 10, end: 11 },
  ],
  codePackage: {
    label: "FastText 原理配套练习包",
    href: "../../fasttext_from_scratch/README.md",
    test: "python -m unittest discover -s fasttext_from_scratch -p 'test_*.py'",
  },
});
