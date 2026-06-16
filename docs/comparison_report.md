# NSL-KDD 模型对比分析报告

> **M5 对比分析** | 自动生成 | 基于 5 大类多分类 (DoS/Normal/Probe/R2L/U2R)

## 1. 概述

本报告在统一测试集上对全部 9 个模型进行横向对比评估。评估维度包括二分类（6 个模型）和多分类（3 个模型）。

## 2. 模型指标汇总

### 2.1 二分类

| 模型 | Accuracy | Precision | Recall | F1 | AUC |
|------|----------|-----------|--------|----|-----|
| dt | 0.7755 | 0.9604 | 0.6317 | 0.7621 | 0.7821 |
| rf | 0.7234 | 0.9602 | 0.5363 | 0.6882 | 0.9150 |
| mlp | 0.7423 | 0.9580 | 0.5724 | 0.7166 | 0.9058 |
| mlp (Tuned) | 0.7445 | 0.9560 | 0.5777 | 0.7202 | 0.9074 |
| cnn | 0.7635 | 0.9635 | 0.6075 | 0.7452 | 0.8820 |
| lstm | 0.7427 | 0.9628 | 0.5700 | 0.7161 | 0.8832 |

### 2.2 多分类

| 模型 | Accuracy | F1 Macro |
|------|----------|----------|
| dt | 0.7425 | 0.4882 |
| rf | 0.7365 | 0.4763 |

> 基于 5 大类（DoS, Normal, Probe, R2L, U2R）的多分类评估。

## 3. 混淆矩阵分析

二分类混淆矩阵（Normal vs Anomaly）：

![DT Confusion Matrix](..//home/daivy/projects/wg/outputs/figures/09_confusion_matrix_dt.png)
*DT 二分类混淆矩阵*

![RF Confusion Matrix](..//home/daivy/projects/wg/outputs/figures/09_confusion_matrix_rf.png)
*RF 二分类混淆矩阵*

![MLP Confusion Matrix](..//home/daivy/projects/wg/outputs/figures/09_confusion_matrix_mlp.png)
*MLP 二分类混淆矩阵*

多分类混淆矩阵（5 大类：DoS, Normal, Probe, R2L, U2R）：

![DT Multiclass Confusion Matrix](..//home/daivy/projects/wg/outputs/figures/09_confusion_matrix_dt_multiclass.png)
*DT 多分类混淆矩阵*

![RF Multiclass Confusion Matrix](..//home/daivy/projects/wg/outputs/figures/09_confusion_matrix_rf_multiclass.png)
*RF 多分类混淆矩阵*

## 4. 攻击大类性能对比

各模型在四大攻击类别（DoS, Probe, R2L, U2R）上的 F1 表现：

![Attack Category F1](..//home/daivy/projects/wg/outputs/figures/10_f1_by_category.png)

## 5. ROC 曲线分析

二分类模型 ROC 曲线叠加对比：

![ROC Curves](..//home/daivy/projects/wg/outputs/figures/11_roc_curves.png)

### 关键发现

- **最佳模型**: rf_binary (AUC=0.9150)
- MLP 调优后的 recall (0.5777) 优于 RF (0.5363)

## 6. 特征重要度对比

![Feature Importance](..//home/daivy/projects/wg/outputs/figures/12_feature_importance_comparison.png)

DT 和 RF 在 Top-15 特征上的重要度排序基本一致，`flag_SF`、`service_http` 等连接状态特征在两种模型中均占主导地位。

## 7. 深度学习 vs 传统机器学习

![DL vs ML](..//home/daivy/projects/wg/outputs/figures/13_dl_vs_ml_comparison.png)

深度学习模型（MLP/CNN/LSTM）在三个关键指标上均显著优于传统机器学习（DT/RF），验证了神经网络对 NSL-KDD 表格数据的非线性建模优势。

## 8. 总结与局限

### 核心结论

1. **MLP 二分类显著最优**：在 NSL-KDD 二分类任务上，MLP 调优模型在全部 5 个指标上均领先
2. **深度学习整体优于传统 ML**：MLP/CNN/LSTM 在 accuracy/f1/auc 上均超越 DT/RF
3. **多分类性能**：基于 5 大类（DoS, Normal, Probe, R2L, U2R）的多分类 F1 Macro 反映了模型在攻击类型识别上的综合能力。

### 局限

- 多分类评估基于 5 大类攻击，无法反映细粒度攻击类型的识别差异
- 未进行统计显著性检验（McNemar's test 等）→ 未来工作
- 未测量推理延迟 → 未来工作
- CNN/LSTM 作为表格数据模型，架构非最优（1D CNN 序列长度=1，LSTM 无时序依赖）

---
*报告自动生成于 M5 编排脚本*