import fs from "node:fs";
import path from "node:path";

const workspaceRoot = "/Users/lyn/Documents/nlp_study";

function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function wrap(text, max = 12) {
  const value = String(text);
  // Chinese can wrap character by character, but Latin terms such as
  // "token", "Softmax" and "N-gram" should stay intact whenever possible.
  const parts = value.match(/[A-Za-z0-9_./+-]+|[\u3400-\u9FFF]|[^\u3400-\u9FFFA-Za-z0-9_./+-]/gu) ?? [];
  const lines = [];
  let line = "";
  for (const part of parts) {
    if (part === " " && line.length === 0) continue;
    if (line.length + part.length <= max) {
      line += part;
      continue;
    }
    if (line.trim()) lines.push(line.trimEnd());
    if (part.length <= max) {
      line = part === " " ? "" : part;
      continue;
    }
    const chars = [...part];
    while (chars.length > max) {
      lines.push(chars.splice(0, max).join(""));
    }
    line = chars.join("");
  }
  if (line.trim()) lines.push(line.trimEnd());
  return lines.length ? lines : [""];
}

function svgText(text, x, y, options = {}) {
  const {
    size = 23,
    color = "#16324F",
    weight = 600,
    anchor = "middle",
    max = 12,
    gap = 34,
  } = options;
  return wrap(text, max).map((line, index) =>
    `<text x="${x}" y="${y + index * gap}" text-anchor="${anchor}" font-family="-apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif" font-size="${size}" font-weight="${weight}" fill="${color}">${escapeXml(line)}</text>`
  ).join("\n");
}

function makeSvg(lesson, index) {
  const nodes = lesson.chain;
  const diagramTitle = lesson.diagramTitle ?? lesson.title;
  const colors = ["#E7F3FF", "#EAF8F0", "#FFF4D9", "#F6EAFF", "#FFECE7"];
  const strokes = ["#3B82C4", "#31966F", "#D69424", "#8C62C7", "#D6674F"];
  const gap = 36;
  const margin = 70;
  const usable = 1200 - margin * 2 - gap * (nodes.length - 1);
  const width = usable / nodes.length;
  const y = 245;
  const height = 190;

  const cards = nodes.map((node, cardIndex) => {
    const x = margin + cardIndex * (width + gap);
    const center = x + width / 2;
    const lineCount = wrap(node, 10).length;
    const textY = y + height / 2 - ((lineCount - 1) * 34) / 2 + 32;
    return `<g>
      <rect x="${x}" y="${y}" width="${width}" height="${height}" rx="24" fill="${colors[cardIndex]}" stroke="${strokes[cardIndex]}" stroke-width="3"/>
      <circle cx="${center}" cy="${y + 40}" r="19" fill="${strokes[cardIndex]}"/>
      <text x="${center}" y="${y + 47}" text-anchor="middle" font-family="sans-serif" font-size="19" font-weight="700" fill="white">${cardIndex + 1}</text>
      ${svgText(node, center, textY, { max: 10 })}
    </g>`;
  }).join("\n");

  const arrows = nodes.slice(0, -1).map((_, arrowIndex) => {
    const x1 = margin + arrowIndex * (width + gap) + width + 7;
    const x2 = margin + (arrowIndex + 1) * (width + gap) - 9;
    return `<line x1="${x1}" y1="${y + height / 2}" x2="${x2}" y2="${y + height / 2}" stroke="#60788E" stroke-width="4" marker-end="url(#arrow)"/>`;
  }).join("\n");

  return `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="680" viewBox="0 0 1200 680" role="img" aria-labelledby="title desc">
  <title id="title">${escapeXml(diagramTitle)}</title>
  <desc id="desc">${escapeXml(nodes.join("，然后"))}</desc>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#F8FBFF"/><stop offset="1" stop-color="#F7F4FF"/></linearGradient>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#60788E"/></marker>
  </defs>
  <rect width="1200" height="680" rx="34" fill="url(#bg)"/>
  <text x="70" y="62" font-family="sans-serif" font-size="18" font-weight="700" fill="#3B6A91">第 ${String(index).padStart(2, "0")} 节 · 概念图</text>
  ${svgText(diagramTitle, 70, 112, { size: 34, color: "#102A43", weight: 760, anchor: "start", max: 29, gap: 42 })}
  ${arrows}
  ${cards}
  <rect x="70" y="512" width="1060" height="105" rx="20" fill="#102A43"/>
  ${svgText(lesson.problem, 600, 553, { size: 21, color: "#FFFFFF", weight: 500, max: 42, gap: 30 })}
  <text x="1130" y="650" text-anchor="end" font-family="sans-serif" font-size="16" fill="#6C8194">原创示意图</text>
</svg>`;
}

function noteFilename(index, lesson) {
  return `${String(index).padStart(2, "0")}-${lesson.slug}.md`;
}

function makeMermaidFlow(items) {
  const labels = items.map((item, index) =>
    `    N${index}["${String(item).replaceAll('"', "'")}"]`
  );
  const edges = items.slice(0, -1).map(
    (_, index) => `    N${index} --> N${index + 1}`
  );
  return `\`\`\`mermaid
flowchart LR
${labels.join("\n")}
${edges.join("\n")}
\`\`\``;
}

function makeNote(topic, lessons, lesson, index, codePackage) {
  const previous = index > 1 ? lessons[index - 2] : null;
  const next = index < lessons.length ? lessons[index] : null;
  const nav = [
    previous ? `[← 上一节：${index - 1} ${previous.title}](./${noteFilename(index - 1, previous)})` : "← 已是第一节",
    "[返回总目录](./README.md)",
    next ? `[下一节：${index + 1} ${next.title} →](./${noteFilename(index + 1, next)})` : "已是最后一节 →",
  ].join(" · ");
  const codeIntro = codePackage
    ? `下面代码默认从项目根目录运行；专题配套实现见 [${codePackage.label}](${codePackage.href})。`
    : "下面代码是帮助理解本节概念的最小示例，默认从项目根目录运行。";
  const extraVisuals = lesson.extraVisuals?.length
    ? `\n\n${lesson.extraVisuals.map(({ title, diagram }) =>
        `### ${title}\n\n\`\`\`mermaid\n${diagram.trim()}\n\`\`\``
      ).join("\n\n")}`
    : "";
  const beginnerLesson = lesson.beginner
    ? `\n\n## 零基础精讲：先把这一节真正弄懂\n\n${lesson.beginner.trim()}`
    : "";

  return `# 第 ${index} 节：${lesson.title}

> 笔记编号 ${index}/${lessons.length} · 对应原视频 P${lesson.p} · [打开这一集](https://www.bilibili.com/video/BV14mdfBDE4Q?p=${lesson.p})

${nav}

## 这节解决什么问题

${lesson.problem}

![第 ${index} 节原创概念图](./diagrams/${String(index).padStart(2, "0")}-concept.svg)

图从左向右读。先跟着数据或推理过程走一遍，再学习下面的术语。

## 辅助流程图

${makeMermaidFlow(lesson.chain)}${extraVisuals}${beginnerLesson}

## 老师原声整理稿（按讲解顺序）

${lesson.article}

## 完整原声逐段记录

[查看本节按时间戳整理的完整音轨转写](./transcripts/p${String(lesson.p).padStart(3, "0")}.md)

逐段记录用于核查老师讲解是否遗漏；正文会进一步纠正口误和语音识别中的技术术语。

## 零基础先记住

${lesson.points.map(point => `- ${point}`).join("\n")}

## ${lesson.codeLabel ?? "最小可运行代码"}

${codeIntro}

\`\`\`python
${lesson.code}
\`\`\`

### 输入和输出怎么看

${lesson.output}

## 最容易踩的坑

${lesson.pit}

## 本节知识链

\`${lesson.chain.join(" → ")}\`

## 自测

**问题：${lesson.quiz}**

<details>
<summary>点开核对答案</summary>

${lesson.answer}

</details>

## 学完检查

- [ ] 我能用自己的话复述老师的讲解顺序
- [ ] 我能在运行前预测关键输出或张量形状
- [ ] 我知道这节方法最容易用错的地方
- [ ] 我能独立回答自测题

${nav}
`;
}

export function generateTopic({
  topic,
  folder,
  lessons,
  stages,
  codePackage = null,
}) {
  const outDir = path.join(workspaceRoot, "notes", folder);
  const diagramDir = path.join(outDir, "diagrams");
  fs.mkdirSync(diagramDir, { recursive: true });

  lessons.forEach((lesson, offset) => {
    const index = offset + 1;
    fs.writeFileSync(
      path.join(diagramDir, `${String(index).padStart(2, "0")}-concept.svg`),
      makeSvg(lesson, index),
    );
    fs.writeFileSync(
      path.join(outDir, noteFilename(index, lesson)),
      makeNote(topic, lessons, lesson, index, codePackage),
    );
  });

  const stageMarkdown = stages.map(({ title, start, end }) => {
    const links = lessons.slice(start - 1, end).map((lesson, offset) => {
      const index = start + offset;
      return `${index}. [${lesson.title}](./${noteFilename(index, lesson)})（原视频 P${lesson.p}）`;
    }).join("\n");
    return `## ${title}\n\n${links}`;
  }).join("\n\n");

  const readme = `# ${topic}：${lessons.length} 节零基础文章笔记

本专题按老师原声的实际讲解顺序整理。口头重复、停顿和现场操作被改写为连贯文章，但保留重要类比、代码解释、易错点和纠错过程。

每节包含原创 SVG 概念图；图片中不含视频分集、课程来源或平台标识。

## 学习方法

1. 先读“老师讲解整理成文章”，理解为什么要学。
2. 再沿原创图复述数据流。
3. 不看输出先运行或手算最小代码。
4. 完成自测，再进入下一节。

${stageMarkdown}

${codePackage ? `## 配套代码\n\n- [${codePackage.label}](${codePackage.href})\n- 测试命令：\`${codePackage.test}\`\n` : ""}
`;
  fs.writeFileSync(path.join(outDir, "README.md"), readme);
  console.log(`generated ${lessons.length} notes and SVG diagrams for ${topic}`);
}
