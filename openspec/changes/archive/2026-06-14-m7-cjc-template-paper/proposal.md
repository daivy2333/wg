## Why

M6 阶段已完成 6 章论文内容撰写（ctexart + 双栏排版），但采用自定义排版格式，无法直接用于期刊投稿。现需套用基于《计算机学报》规范的 CJC LaTeX 模板（由 CTeX-org 维护的 chinesejournal 框架），将论文重新排版为符合学术期刊投稿规范的终稿 PDF。该模板提供标准化的双语元数据、作者简介、参考文献格式、算法伪代码、英文背景介绍等期刊必需模块，是论文从"课程作业格式"到"学术投稿格式"的关键一步。

## What Changes

- **迁移文档类**：从 `ctexart` + 自定义 `geometry`/`twocolumn` 更换为 `cjc` class，模板自带排版规范
- **重构元数据**：将 `\title`/`\author`/`abstract` 迁移到 `\classsetup{}` 集中配置，补充英文对应字段 (title*/abstract*/keywords*)
- **适配正文**：将 6 章（ch1-ch6）从双栏 `figure*`/`table*` 环境改为单栏 `figure`/`table` 环境，调整图表尺寸
- **新增英文背景**：添加 `background` 环境（约 400 词），介绍研究领域、课题意义、团队成果
- **新增算法伪代码**：使用 `algorithm`/`procedure` 环境展示关键训练/评估流程
- **新增附录**：使用 `\appendix` 放置补充表格与推导
- **更换参考文献系统**：从 `unsrt` + `cite` 更换为 `cjc.bst` + `hyperref`，按正文引用顺序排列
- **配置编译流程**：适配 `latexmk -xelatex` 编译，确保目录/交叉引用/图表编号正确

## Capabilities

### New Capabilities

- `paper-cjc-template`: 基于 CJC 模板的论文主文件，含 `\classsetup` 元数据配置、双语标题摘要关键词、编译流程
- `paper-cjc-chapters`: 6 章正文（绪论/数据预处理/传统ML/神经网络/对比分析/总结）适配 cjc 单栏排版，图表环境从 figure*/table* 迁移
- `paper-cjc-supplementary`: 英文背景 (background)、算法伪代码 (algorithm/procedure)、附录 (appendix) 三个补充模块

### Modified Capabilities

<!-- 本次为纯排版迁移，不涉及现有 spec 的能力变更 -->

## Impact

- **新增文件**：`paper/cjc-main.tex`（新主文件）、`paper/cjc-refs.bib`（CJC 格式参考文献）、`paper/background.tex`、`paper/appendix.tex`、`paper/algorithm.tex`
- **修改文件**：`paper/chapters/ch1-*.tex` ~ `ch6-*.tex`（图表环境适配），`paper/Makefile`（新增 cjc 编译目标）
- **依赖**：TeX Live 2024+（含 xelatex、latexmk、cjk 支持）、`cjc.cls`、`cjc.bst`（从 `paper/latex-/` 模板目录复制）
- **不涉及**：Python 代码、数据集、测试套件、模型文件、实验图表内容（仅调整引用方式）
- **回滚方案**：原 `paper/main.tex` 与 `refs.bib` 保留不动，新文件使用 `cjc-` 前缀命名，失败时切回旧版本即可
