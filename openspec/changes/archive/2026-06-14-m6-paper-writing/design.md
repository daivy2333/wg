## Context

M6 是将 M1-M5 实验成果整合为课程论文的阶段。论文使用 LaTeX + ctex 中文支持，目标 ≥5000 字、图文并茂。三人分工撰写 6 个章节。

**关键约束**：
- LaTeX 编译环境：需要 xelatex + ctex 宏包（conda 环境可能没有，需系统安装）
- 图表引用：使用相对路径链接 `outputs/figures/` 下的 15 张 PNG
- 参考文献：BibTeX 格式，引用 NSL-KDD 论文 + scikit-learn/PyTorch 文档
- 封面：含课程名称、题目、三人姓名学号

## Goals / Non-Goals

**Goals:**
- 完整 LaTeX 项目，`xelatex main.tex` 一键编译为 PDF
- 6 章正文 ≥5000 字，引用 15 张图表
- README.md 包含完整复现步骤
- 三人分工明确，各自撰写对应章节

**Non-Goals:**
- 不翻译为英文版
- 不投稿期刊（课程报告级别，非学术投稿）
- 不新增实验、不修改已有模型
- 不制作交互式 PPT（静态 PPT 提纲）

## Decisions

### D1: LaTeX 项目结构

```
paper/
├── main.tex          # 主文件（封面 + 摘要 + 引入各章 + 参考文献）
├── chapters/
│   ├── ch1-intro.tex      # 绪论 + 数据集介绍（A同学）
│   ├── ch2-preprocess.tex # 数据预处理（A同学）
│   ├── ch3-ml.tex         # 传统ML：DT + RF（B同学）
│   ├── ch4-dl.tex         # 神经网络：MLP/CNN/LSTM（C同学）
│   ├── ch5-comparison.tex # 对比分析（C同学）
│   └── ch6-conclusion.tex # 总结与展望（三人协作）
├── figures/          # 符号链接 → ../../outputs/figures/
├── refs.bib          # BibTeX 参考文献
└── main.pdf          # 编译产物（gitignore 可选）
```

**理由**: 章节独立文件便于三人并行撰写，`\input{}` 合并到主文件。

### D2: 编译工具链

**选择**: XeLaTeX + ctex + BibTeX

**理由**: 
- XeLaTeX 原生支持 UTF-8 中文，无需 CJK 编码转换
- ctex 宏包处理中文排版（字体、行距、缩进）
- BibTeX 管理参考文献

### D3: 图表引用方式

**选择**: 符号链接 `paper/figures/` → `../../outputs/figures/`

**理由**: 避免复制 15 张 PNG 造成重复，保持单一数据源。

### D4: 论文分工

| 章节 | 负责人 | 字数分配 | 内容要点 |
|------|--------|----------|----------|
| 第1章 绪论 | A同学 | ~800字 | 入侵检测背景、NSL-KDD 数据集介绍、41 特征表 |
| 第2章 数据预处理 | A同学 | ~800字 | EDA、特征选择 Top-20、数据划分 |
| 第3章 传统ML模型 | B同学 | ~1200字 | DT/RF 原理、网格搜索、二/多分类结果 |
| 第4章 神经网络模型 | C同学 | ~1200字 | MLP/CNN/LSTM 架构、训练策略、SMOTE 实验 |
| 第5章 对比分析 | C同学 | ~1000字 | 统一测试集评估、图表对比、关键发现 |
| 第6章 总结展望 | 三人 | ~600字 | 核心结论、局限、未来工作 |

总计 ≥5000 字（正文不含封面/摘要/参考文献）。

### D5: 并行写作策略

章节 1-5 可并行撰写（无内容依赖），第 6 章依赖前 5 章完成后提炼。

## Risks / Trade-offs

| 风险 | 缓解措施 |
|------|----------|
| LaTeX 编译环境未安装 | 先检查 `xelatex --version`；若缺失，在用户终端安装 texlive-xetex + ctex |
| 中文引用图表路径问题 | 使用相对路径 `../outputs/figures/` 并用 `\graphicspath` 统一配置 |
| BibTeX 兼容性 | 使用标准 BibTeX 格式，避免 biblatex 复杂依赖 |
| 三人内容风格不统一 | 在 main.tex 中统一章节标题格式、图表编号、引用风格 |
