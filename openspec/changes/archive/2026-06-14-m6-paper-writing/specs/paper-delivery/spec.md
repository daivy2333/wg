## ADDED Requirements

### Requirement: LaTeX 项目骨架
系统 SHALL 提供完整的 LaTeX 项目，使用 ctex 文档类支持中文，通过 `xelatex` 编译为 PDF。

#### Scenario: 编译 PDF
- **WHEN** 用户在 `paper/` 目录执行 `xelatex main.tex`
- **THEN** 生成 `main.pdf` 包含封面、目录、摘要、6 章正文、参考文献

### Requirement: 论文内容完整性
论文正文 SHALL 包含 6 个章节，总字数 ≥5000，引用 M1-M5 产出的 15 张图表。

#### Scenario: 章节完整性
- **WHEN** 论文编译完成
- **THEN** 封面含课程名/题目/三人姓名学号，6 章覆盖数据预处理→模型训练→对比分析→总结

### Requirement: 项目交付完整性
项目 SHALL 提供可复现的运行指南和完善的代码说明。

#### Scenario: 新用户复现
- **WHEN** 新用户按 README.md 步骤操作
- **THEN** 能够从零运行 `train_m3.py` 和 `evaluate_m5.py` 复现全部实验
