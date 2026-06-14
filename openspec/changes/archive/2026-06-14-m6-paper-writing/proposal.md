## Why

M1-M5 完成了 NSL-KDD 网络入侵检测系统的完整实验流程（数据预处理、传统 ML、神经网络、对比分析），积累了丰富的实验结果（144 测试、10 模型、15 张图表、4 份报告）。M6 需要将这些分散的成果整合为一篇符合中文核心期刊格式的课程论文（≥5000 字，图文并茂），完成代码整理和答辩准备。

## What Changes

- 新建 `paper/` 目录，包含 LaTeX 项目（ctex 中文支持，编译为 PDF）
- 撰写论文正文：绪论+数据集、数据预处理、传统ML模型、神经网络模型、对比分析、总结展望（6 章）
- 论文引用 M1-M5 产出的 15 张图表和 10 模型指标
- 完善 README.md 项目说明（运行指南、复现步骤）
- 准备答辩 PPT / 演示材料
- **不修改** M1-M5 任何已有代码、模型、数据文件

## Capabilities

### New Capabilities

- `paper-latex`: LaTeX 项目骨架 + 中文论文编译（ctex + PDF 输出）
- `paper-content`: 6 章论文正文（三人分工撰写，≥5000 字）
- `project-delivery`: README 完善 + 答辩 PPT 准备

### Modified Capabilities

<!-- M6 仅新增，不修改已有 spec -->

## Impact

- **新增目录**: `paper/`（LaTeX 源码 + 编译产物）
- **新增文件**: `paper/main.tex`, `paper/chapters/*.tex`, `paper/figures/`（符号链接到 outputs/figures/）
- **修改文件**: `README.md`（运行指南更新）
- **新增产出**: `paper/main.pdf`（最终论文 PDF）
- **不影响**: M1-M5 所有已有文件
