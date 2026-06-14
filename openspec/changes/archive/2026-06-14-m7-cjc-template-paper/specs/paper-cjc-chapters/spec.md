# paper-cjc-chapters

> 6 章正文 cjc 单栏排版适配
> Version: 1.0.0 | Last Updated: 2026-06-14

## ADDED Requirements

### Requirement: 章节文件保持独立引用
六章正文 SHALL 保持为独立 `.tex` 文件（`paper/chapters/ch1-intro.tex` ~ `ch6-conclusion.tex`），通过 `\input{}` 在主文件中引用，每章内容不重复。

#### Scenario: 主文件引用全部章节
- **WHEN** 读取 `cjc-main.tex` 正文部分
- **THEN** 包含 6 条 `\input{chapters/chX-*.tex}` 语句
- **AND** 章节按 ch1 → ch6 顺序排列

### Requirement: 图表环境从通栏迁移到单栏
各章中的 `figure*`/`table*` 环境 SHALL 替换为 `figure`/`table`，`\columnwidth` 替换为 `\textwidth`，图表在单栏版心内正确显示。

#### Scenario: 图表环境使用单栏模式
- **WHEN** 在 cjc 模板下编译
- **THEN** 所有 `figure*` 已替换为 `figure`
- **AND** 所有 `table*` 已替换为 `table`
- **AND** 图表宽度参数从 `\columnwidth` 替换为 `\textwidth`

#### Scenario: 原跨栏大图尺寸适配
- **WHEN** 编译包含原 `width=\textwidth`（跨双栏）的图表
- **THEN** 图表宽度调整为 `width=0.85\textwidth` 或适当比例
- **AND** 图表在单栏页面内不超出右边界

### Requirement: 图表引用格式不变
正文中 `\ref{fig:*}` / `\ref{tab:*}` 的交叉引用 SHALL 保持有效，编译后图表编号与引用一致。

#### Scenario: 图表交叉引用正确
- **WHEN** 编译完成后查看 PDF
- **THEN** 所有 `\ref{fig:chX-*}` 引用显示正确的图号
- **AND** 所有 `\ref{tab:*}` 引用显示正确的表号
- **AND** 图表编号按出现顺序自动递增

### Requirement: 数学公式与特殊字符兼容
各章中的 `$...$` / `\[...\]` 数学公式、`\cite{}` 引用、`\texttt{}` 代码标记、`\%` 百分号 SHALL 在 cjc 模板下正确渲染。

#### Scenario: 所有符号正确显示
- **WHEN** 编译后查看 PDF 各章
- **THEN** 数学公式渲染正常（特征名如 `flag\_SF`）
- **AND** `\cite{tavallaee2009}` 显示为 `[1]` 格式
- **AND** `\texttt{src/evaluation/metrics.py}` 显示为等宽字体
- **AND** `\%` 正确显示为百分号

### Requirement: 图形路径配置
`\graphicspath{}` SHALL 指向 `outputs/figures/` 目录，使各章 `\includegraphics` 可正确找到图片文件。

#### Scenario: 图片文件可被找到
- **WHEN** 编译过程加载图片
- **THEN** 无 "file not found" 错误
- **AND** 所有 15 张图表 PNG 正确嵌入
