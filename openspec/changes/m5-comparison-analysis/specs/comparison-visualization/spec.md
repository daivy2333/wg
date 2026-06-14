## ADDED Requirements

### Requirement: 混淆矩阵热力图
系统 SHALL 为每个二分类模型生成 2×2 混淆矩阵热力图，包含真实数值标注和配色方案。

#### Scenario: DT 混淆矩阵生成
- **WHEN** 对 DT tuned 模型的二分类预测结果调用 `plot_confusion_matrix_heatmap()`
- **THEN** 生成 `outputs/figures/09_confusion_matrix_dt.png`，2×2 矩阵含 (TN, FP, FN, TP) 数值标注，使用 Blues 配色

#### Scenario: RF 混淆矩阵生成
- **WHEN** 对 RF tuned 模型的二分类预测结果调用 `plot_confusion_matrix_heatmap()`
- **THEN** 生成 `outputs/figures/09_confusion_matrix_rf.png`

#### Scenario: MLP 混淆矩阵生成
- **WHEN** 对 MLP tuned 模型的二分类预测结果调用 `plot_confusion_matrix_heatmap()`
- **THEN** 生成 `outputs/figures/09_confusion_matrix_mlp.png`

### Requirement: 性能对比柱状图
系统 SHALL 生成多模型在四大攻击类别上的 F1 分组柱状图。

#### Scenario: 攻击大类 F1 柱状图
- **WHEN** 传入 DT/RF/MLP/CNN/LSTM 在 DoS、Probe、R2L、U2R 四个类别上的 F1 数值
- **THEN** 生成 `outputs/figures/10_f1_by_category.png`，X 轴=攻击类别，分组=模型，Y 轴=F1，每个柱子顶部标注数值，含图例

### Requirement: ROC 曲线叠加图
系统 SHALL 生成多模型 ROC 曲线叠加对比图。

#### Scenario: 二分类 ROC 曲线叠加
- **WHEN** 传入 DT/RF/MLP/CNN/LSTM 的 `(fpr, tpr, auc)` 元组
- **THEN** 生成 `outputs/figures/11_roc_curves.png`，每条曲线不同颜色，图例标注"模型名 (AUC=0.xxx)"，含 y=x 参考虚线

### Requirement: 特征重要度对比图
系统 SHALL 生成 DT 与 RF 的 Top-15 特征重要度并排对比图。

#### Scenario: 特征重要度并排对比
- **WHEN** 传入 DT 和 RF 的 Top-15 特征重要度 Series
- **THEN** 生成 `outputs/figures/12_feature_importance_comparison.png`，左右并排水平柱状图，共享特征名标签，Y 轴=特征名，X 轴=重要度

### Requirement: 深度学习 vs 传统 ML 对比图
系统 SHALL 生成深度学习模型与传统 ML 模型的性能差距对比图。

#### Scenario: 二分类关键指标对比
- **WHEN** 传入 DT/RF/MLP/CNN/LSTM 的 accuracy/f1/auc 三个关键指标
- **THEN** 生成 `outputs/figures/13_dl_vs_ml_comparison.png`，分组柱状图 X 轴=指标，分组=模型，标注具体数值

### Requirement: 图表样式一致性
所有图表 SHALL 使用统一的 matplotlib 样式配置。

#### Scenario: 样式配置
- **WHEN** 生成任意图表时
- **THEN** 使用 Agg 后端、dpi=80、统一的配色方案（DT=蓝、RF=绿、MLP=红、CNN=橙、LSTM=紫）、12pt 字体、`plt.tight_layout()` 和 `plt.close()` 后保存
