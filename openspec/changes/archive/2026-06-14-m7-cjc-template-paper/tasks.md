# Tasks: M7 CJC 模板论文排版

> 基于 `paper/latex-/` 的 CJC 模板，将 M6 论文重新排版为期刊投稿格式。
> 对应 `tasks.md` M7: 7.1–7.7

---

## 1. 模板准备与目录设置

- [ ] 1.1 复制 `cjc.cls` 和 `cjc.bst` 到 `paper/` 目录
- [ ] 1.2 复制 `latexmkrc` 到 `paper/` 目录（覆盖现有或合并）
- [ ] 1.3 更新 `paper/Makefile` 新增 cjc 编译目标（`cjc-main`）
- [ ] 1.4 验证模板文件可用：`cd paper && latexmk -xelatex -pdf example.tex` 通过

**验收**: `paper/cjc.cls`, `paper/cjc.bst`, `paper/latexmkrc` 存在，example.tex 可编译

---

## 2. 创建 cjc-main.tex 主文件

- [ ] 2.1 创建 `paper/cjc-main.tex`，配置 `\documentclass{cjc}` 和基础包（booktabs, algorithm, algorithmic, siunitx, hyperref）
- [ ] 2.2 编写 `\classsetup{title, title*, authors, affiliations}`，中文标题使用原论文标题，英文标题直译，3 位作者使用 `[TODO]` 占位符
- [ ] 2.3 编写 `\classsetup{abstract, abstract*, keywords, keywords*}`，中英文摘要与关键词从原论文提取
- [ ] 2.4 配置 `\classsetup{grants}` 占位符，`\graphicspath{{../outputs/figures/}}`
- [ ] 2.5 配置 `\begin{document}`, `\maketitle`, `\input{chapters/...}` 6 章引用，`\bibliographystyle{cjc}`, `\bibliography{cjc-refs}`, `\end{document}`

**验收**: cjc-main.tex 文件完整，`\classsetup{}` 包含所有必填字段，`\input{}` 引用 6 章

---

## 3. 迁移 6 章正文（ch1–ch6）

- [ ] 3.1 ch1-intro.tex：`figure*` → `figure`，`\columnwidth` → `\textwidth`，3 张图表尺寸适配单栏
- [ ] 3.2 ch2-preprocess.tex：同上，5 张图表尺寸适配
- [ ] 3.3 ch3-ml.tex：`figure*`/`table*` → `figure`/`table`，3 张图 + 对比表
- [ ] 3.4 ch4-dl.tex：同上，图表环境迁移
- [ ] 3.5 ch5-comparison.tex：`figure*`/`table*` → `figure`/`table`，7 张图 + 2 个表
- [ ] 3.6 ch6-conclusion.tex：纯文字章节，检查无图表环境残留

**验收**: 6 章文件无 `figure*`/`table*`/`\columnwidth` 残留，`\textwidth` 替换完成

---

## 4. 创建补充模块

- [ ] 4.1 创建 `paper/background.tex`：英文背景介绍（约 400 词），涵盖研究领域、课题意义、本文贡献
- [ ] 4.2 创建 `paper/appendix.tex`：包含模型超参数完整列表表、SMOTE 详细实验结果表、特征完整排名（Top-20）
- [ ] 4.3 创建 `paper/algorithm.tex`：至少 1 个伪代码块——NSL-KDD 入侵检测实验总流程（训练-评估循环）
- [ ] 4.4 在 `cjc-main.tex` 尾部引用：`\acknowledgments`, `\bibliography`, `\input{background}`, `\appendix`, `\input{appendix}`, `\input{algorithm}`, `\makebiographies`

**验收**: background/appendix/algorithm 三个文件存在且内容非空，主文件正确引用

---

## 5. 参考文献迁移

- [ ] 5.1 复制 `paper/refs.bib` → `paper/cjc-refs.bib`
- [ ] 5.2 验证字段兼容性：`number`/`volume` 字段方向适配 `cjc.bst`（期刊用 `number={卷}, volume={期}`），中文条目 `language={chinese}` 已存在
- [ ] 5.3 检查 `doi` 字段：cjc.bst 如不支持则移除或转为 `note` 字段
- [ ] 5.4 确保引用的 bib key 与正文 `\cite{}` 一致（tavallaee2009, kddcup1999, nslkdd_unb, scikit-learn, pytorch, zh_renshi2023, zh_wang2022, zh_liuzong2024, smote）

**验收**: cjc-refs.bib 存在，9 条参考文献字段完整，bib key 与正文一致

---

## 6. 编译验证

- [ ] 6.1 执行首轮编译：`cd paper && latexmk -xelatex cjc-main.tex`，记录错误
- [ ] 6.2 修复编译错误（包冲突、缺失引用、格式问题）
- [ ] 6.3 验证 PDF 输出：封面信息完整、目录正确、6 章内容完整、图表显示正常
- [ ] 6.4 验证交叉引用：`\ref{fig:*}`/`\ref{tab:*}` 编号正确，`\cite{}` 引用对应参考文献列表
- [ ] 6.5 验证特殊元素：`\texttt{}` 代码块、`\%` 百分号、`$...$` 数学公式渲染正常

**验收**: `latexmk -xelatex cjc-main.tex` 退出码 0，PDF 可打开且内容完整

---

## 7. 终稿检查与文档更新

- [ ] 7.1 检查字数与篇幅：确认各章内容完整无截断
- [ ] 7.2 检查图表编号连续性：图 1–15 按章节顺序，表 1–3 连续
- [ ] 7.3 检查参考文献引用顺序：[1]–[9] 按正文出现顺序
- [ ] 7.4 更新 `tasks.md` M7 任务状态（7.1–7.7 全部完成）
- [ ] 7.5 更新 `SNAPSHOT.md` 项目状态（M7 完成）
- [ ] 7.6 记录优化点：PNG → 矢量图（O-M7-01 到 `optimization/spec.md`）

**验收**: tasks.md/SNAPSHOT.md 已更新，M7 标记完成，优化点已记录
