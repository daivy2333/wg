## Context

M6 论文使用 `ctexart` 文档类 + 自定义双栏排版生成 6 章完整内容。M7 需将其迁移到基于《计算机学报》(CJC) 规范的 LaTeX 模板（`cjc.cls`），模板源码位于 `paper/latex-/`（CTeX-org/chinesejournal），通过 `\classsetup{}` 集中管理双语元数据，使用 `cjc.bst` 参考文献格式，支持 algorithm/procedure/background/appendix 等期刊必备环境。

## Goals / Non-Goals

**Goals:**
- 创建基于 `cjc.cls` 的新主文件，配置中英文标题/摘要/关键词/作者信息
- 将 6 章正文从双栏 `figure*`/`table*` 改为单栏 `figure`/`table` 环境
- 添加 background（约 400 词英文背景）、algorithm（关键流程伪代码）、appendix（补充表格/推导）
- 将参考文献从 `unsrt` 格式迁移到 `cjc.bst`，确保按引用顺序排列
- 使用 `latexmk -xelatex` 成功编译，目录/交叉引用/图表编号/参考文献正确

**Non-Goals:**
- 不修改实验图表内容（仅调整引用方式）
- 不修改正文文字内容（仅排版适配）
- 不转换为矢量图（留作后续优化 O-M7-01）
- 不填写真实作者信息（使用占位符）

## Decisions

### D1: 新主文件命名策略

**选择**：新建 `paper/cjc-main.tex`，保留原 `paper/main.tex` 不动

**原因**：
- 原主文件是 M6 的完整工作产出，需保留作为对比基准
- 新旧两套体系（ctexart vs cjc）差异大，合并在一个文件会增加出错风险
- 新文件独立存在，便于 A/B 对比编译结果

**替代方案**：直接修改 `main.tex` → 拒绝，因为需要保留 M6 双栏版本作为参考

### D2: 模板文件管理

**选择**：将 `cjc.cls`、`cjc.bst` 从 `paper/latex-/` 复制到 `paper/` 目录

**原因**：
- `latex-/` 目录是上游模板仓库（含 git 子仓库），不能直接修改
- `paper/` 是项目论文工作目录，编译时 `.cls`/`.bst` 需在同一目录或 `TEXMF` 路径下
- 复制而非软链：避免跨仓库依赖，确保自包含可复现

**替代方案**：设置 `TEXINPUTS` 环境变量 → 拒绝，增加编译复杂度，不便于团队协作

### D3: 图表环境迁移

**选择**：`figure*`/`table*` → `figure`/`table`，`width=\columnwidth` → `width=\textwidth`

**原因**：
- CJC 模板默认单栏，`figure*`（跨双栏）在单栏下等价于 `figure`
- `\columnwidth` 在双栏模式下是单栏宽度，单栏下 `\textwidth` 等于版心宽度
- 图表尺寸需适当缩放，单栏版心较窄（约 14cm vs 双栏 7.5cm）

**图表尺寸规则**：
| 原尺寸 | 新尺寸 | 适用图表 |
|--------|--------|----------|
| `width=\columnwidth` | `width=\textwidth` | 全宽图表 |
| `width=\textwidth` (跨栏) | `width=0.85\textwidth` | 原跨栏大图 |
| `width=0.9\columnwidth` | `width=0.9\textwidth` | 一般图表 |

### D4: 参考文献迁移

**选择**：复制 `refs.bib` → `cjc-refs.bib`，调整字段以适配 `cjc.bst`

**原因**：
- `cjc.bst` 使用 `number`/`volume` 字段（而非某些 BST 的 `volume`/`number` 反序）
- 中文参考文献需要 `language = {chinese}` 字段
- 引用需按正文出现顺序排列（`cjc.bst` 默认行为）
- M6 参考文献本身已基本满足 cjc 格式，需验证 DOI/URL 字段兼容性

**替代方案**：直接使用 `refs.bib` → 部分条目可能需要微调字段名

### D5: 编译流程

**选择**：使用 `latexmk -xelatex` 替代原 `xelatex → bibtex → xelatex × 2` 手动流程

**原因**：
- 模板提供的 `latexmkrc` 已配置 `pdf_mode=5`（xelatex）和 bibtex 自动检测
- `make` 命令：`make MAIN=cjc-main` 即可编译
- 自动处理编译次数，确保交叉引用/目录/参考文献正确

## Risks / Trade-offs

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| `cjc.cls` 与某些 LaTeX 包冲突 | 中 | 编译失败 | 从最小包集开始，逐个添加并编译验证 |
| 中文参考文献 BibTeX 兼容性 | 中 | 引用显示乱码或缺失 | 验证 `language={chinese}` 字段，必要时用 `bibtexu` |
| 表格过长导致单栏溢出 | 低 | 排版难看 | 使用 `small`/`footnotesize` 缩小字体，或拆分表格 |
| 图表 PNG 不满足矢量要求 | 高 | 投稿被拒 | 记录为优化点 O-M7-01，后期用 matplotlib `savefig(format='pdf')` 重新生成 |
| 作者信息占位符遗漏 | 低 | 编译警告 | 在 classsetup 中填写明显标记 `[TODO: 请填写]` |

## Open Questions

- Q1: 是否需要在正文中增加算法伪代码？如果有具体算法流程（如训练循环、评估流程），需要确定描述哪些
- Q2: 附录需要包含哪些内容？已知可选：SMOTE 详细实验结果表、特征选择完整排名、模型超参数完整列表
- Q3: 英文标题翻译需要确认（目前暂用直译，可由作者审定）
