## ADDED Requirements

### Requirement: 对比分析报告完整结构
系统 SHALL 自动生成结构完整的 Markdown 对比分析报告，覆盖全部 7 个对比维度。

#### Scenario: 报告章节完整性
- **WHEN** 编排脚本完成所有评估和图表生成后
- **THEN** `docs/comparison_report.md` 包含以下章节：
  1. 概述（目标、数据、模型列表）
  2. 模型指标汇总表（二分类 + 多分类）
  3. 混淆矩阵分析（DT/RF/MLP 对比）
  4. 攻击大类性能对比（DoS/Probe/R2L/U2R 的 F1）
  5. ROC 曲线分析
  6. 特征重要度对比（DT vs RF）
  7. 深度学习 vs 传统 ML 总结
  8. SMOTE 失败实验分析（独立小节）
  9. 总结与局限

### Requirement: 数值引用一致性
报告中的指标数值 SHALL 与 `metrics_m5.json` 中的数值一致，所有图表引用使用正确的文件路径。

#### Scenario: 指标一致性
- **WHEN** 报告引用 MLP tuned 的 f1 时
- **THEN** 数值与 `outputs/metrics_m5.json` 中 `mlp_binary_tuned.f1` 完全一致

#### Scenario: 图表引用
- **WHEN** 报告引用混淆矩阵图时
- **THEN** 使用正确的相对路径 `outputs/figures/09_confusion_matrix_*.png`

### Requirement: 诚实记录局限
报告 SHALL 诚实记录所有已知局限，包括多分类 unseen attack 问题、SMOTE 副作用、DT/RF 多分类接近随机等。

#### Scenario: 多分类局限说明
- **WHEN** 报告讨论多分类结果时
- **THEN** 明确说明"测试集含 15 种训练未见攻击类型，模型仅有 23 个输出节点，无法预测这些类别"，并同时报告 full_accuracy 和 known_class_accuracy

#### Scenario: SMOTE 副作用说明
- **WHEN** 报告提及 SMOTE 时
- **THEN** 引用 O-NN-01 优化记录，注明"副作用：多分类 full_accuracy 从 0.101 降至 0.028（-72%）"

### Requirement: SMOTE 失败实验独立处理
SMOTE 模型的评估结果 SHALL 在独立小节中呈现，不混入主对比图表和表格。

#### Scenario: SMOTE 独立呈现
- **WHEN** 生成对比报告时
- **THEN** SMOTE 模型出现在"§8 SMOTE 失败实验分析"章节，不出现在 §2 指标汇总表、§3 混淆矩阵、§4 攻击大类对比、§5 ROC 曲线中
