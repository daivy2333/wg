## ADDED Requirements

### Requirement: 统一的模型评估框架
系统 SHALL 提供模型无关的评估函数，支持 sklearn 和 PyTorch 双后端，在测试集上计算完整的二分类和多分类指标。

#### Scenario: 二分类评估
- **WHEN** 传入模型预测结果 `y_pred`、预测概率 `y_prob` 和真实标签 `y_true`
- **THEN** 返回包含 accuracy、precision、recall、f1、auc 五个指标的 dict

#### Scenario: 多分类评估（含 unseen class 处理）
- **WHEN** 传入多分类预测结果和真实标签（测试集可能包含训练未见类）
- **THEN** 返回 full_accuracy（全量）、known_class_accuracy（剔除未见/稀有类）、f1_macro 三个指标

#### Scenario: sklearn 模型评估
- **WHEN** 传入已训练的 sklearn 模型和测试数据 DataFrame/ndarray
- **THEN** 自动调用 model.predict() 和 model.predict_proba() 后计算指标

#### Scenario: PyTorch 模型评估
- **WHEN** 传入已加载的 PyTorch nn.Module 和测试数据
- **THEN** 自动调用 model.eval() + torch.no_grad() 后计算指标

#### Scenario: 攻击大类 F1 拆解
- **WHEN** 传入预测结果、真实标签和 label→category 映射字典
- **THEN** 返回各攻击大类（DoS、Probe、R2L、U2R）的 F1 score

### Requirement: 对比可视化生成
系统 SHALL 生成论文级别的静态对比图表，包括混淆矩阵热力图、性能对比柱状图、ROC 曲线叠加图和特征重要度对比图。

#### Scenario: 混淆矩阵热力图
- **WHEN** 传入二分类的预测结果和真实标签
- **THEN** 生成 2×2 混淆矩阵热力图 PNG（annot=True, fmt='d', cmap='Blues'），保存到 `outputs/figures/`，含标题和轴标签

#### Scenario: 性能对比柱状图
- **WHEN** 传入多个模型在 4 大攻击类别上的 F1 字典
- **THEN** 生成分组柱状图 PNG（x=攻击类别，group=模型，y=F1），含图例、数值标注

#### Scenario: ROC 曲线叠加图
- **WHEN** 传入多个二分类模型的 `(fpr, tpr, auc)` 元组列表
- **THEN** 生成单张 PNG 含多条 ROC 曲线叠加，对角线参考线，图例标注模型名+AUC 值

#### Scenario: 特征重要度对比图
- **WHEN** 传入 DT 和 RF 的 Top-K 特征重要度 Series
- **THEN** 生成并排水平柱状图 PNG（左侧 DT、右侧 RF，共享特征标签）

### Requirement: 对比分析报告生成
系统 SHALL 自动生成包含全部 7 个对比维度的综合分析报告，涵盖指标汇总表、图表引用、结论分析和局限讨论。

#### Scenario: 报告结构完整性
- **WHEN** 编排脚本完成所有评估和图表生成后
- **THEN** 生成的 `docs/comparison_report.md` 包含以下章节：模型指标汇总、混淆矩阵分析、攻击大类性能对比、ROC 曲线分析、特征重要度对比、深度学习 vs 传统 ML、SMOTE 实验分析、总结与展望

#### Scenario: 指标汇总表
- **WHEN** 报告生成时
- **THEN** 汇总表包含所有模型的 accuracy/precision/recall/f1/auc（二分类）和 full_accuracy/known_class_accuracy/f1_macro（多分类），与图表中的数值一致

#### Scenario: SMOTE 失败实验诚实记录
- **WHEN** 报告提及 SMOTE 实验时
- **THEN** 明确标注"SMOTE 在本项目中效果负面（full_accuracy -72%）"，引用优化记录 O-NN-01，不将其混入主对比图表
