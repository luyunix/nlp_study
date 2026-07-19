import fs from "node:fs";
import path from "node:path";

const root = "/Users/lyn/Documents/nlp_study";
const topics = [
  ["课程导学", "course-introduction", 1, 4],
  ["文本预处理", "text-preprocessing", 5, 37],
  ["RNN", "rnn", 38, 65],
  ["Attention", "attention", 66, 79],
  ["Seq2Seq 英法翻译", "seq2seq-translation", 80, 105],
  ["Transformer", "transformer", 106, 143],
  ["FastText 文本分类", "fasttext-classification", 144, 154],
  ["迁移学习与 Transformers 任务", "transfer-learning", 155, 183],
  ["BERT、ELMo 与 GPT", "foundation-models", 184, 189],
];

const sections = [];
let articleCount = 0;

for (const [topic, folder, start, end] of topics) {
  const dir = path.join(root, "notes", folder);
  const files = fs.readdirSync(dir)
    .filter(name => /^\d{2}-.*\.md$/.test(name))
    .sort();
  if (files.length !== end - start + 1) {
    throw new Error(`${folder}: expected ${end - start + 1}, got ${files.length}`);
  }
  const links = files.map((file, offset) => {
    const text = fs.readFileSync(path.join(dir, file), "utf8");
    const title = text.match(/^# 第 \d+ 节：(.+)$/m)?.[1];
    const p = text.match(/对应原视频 P(\d+)/)?.[1];
    if (!title || Number(p) !== start + offset) {
      throw new Error(`${folder}/${file}: title or P mismatch`);
    }
    articleCount += 1;
    const global = start + offset;
    return `${global}. [${title}](./notes/${folder}/${file})`;
  }).join("\n");
  sections.push(
    `## ${start}–${end}　${topic}\n\n` +
    `[打开专题学习说明](./notes/${folder}/README.md)\n\n${links}`
  );
}

const readme = `# NLP 零基础学习笔记：1–189 节总目录

这是按老师原声讲解顺序整理的完整学习入口，共 ${articleCount} 篇。每篇都包含：

- 老师讲解改写成的连贯文章；
- 可核对遗漏的按时间戳原声音轨记录；
- 原创概念图，以及流程图、UML 或时序图；
- 最小代码、输入输出解释、易错点和自测。

## 小白学习顺序

1. 先读每节“这节解决什么问题”，不要急着背术语。
2. 顺着原创图把数据流说一遍。
3. 阅读“老师原声整理稿”，遇到形状就翻译成“几条样本 × 几个 token × 每个多少维”。
4. 运行最小代码，先猜输出，再核对结果。
5. 完成自测；答不出来就回到对应时间段，而不是整节重看。

> 原声转写用于完整性核查，可能仍有同音字；正文已对技术术语、口误和关键概念作校正。

${sections.join("\n\n")}

## 配套可运行代码

- [文本预处理练习](./text_preprocessing_from_scratch/README.md)
- [RNN 练习](./rnn_from_scratch/README.md)
- [Attention 练习](./attention_from_scratch/README.md)
- [Seq2Seq 练习](./seq2seq_from_scratch/README.md)
- [Transformer 练习](./transformer_from_scratch/README.md)
- [FastText 原理练习](./fasttext_from_scratch/README.md)
`;

fs.writeFileSync(path.join(root, "README.md"), readme);
console.log(`generated master index with ${articleCount} articles`);
