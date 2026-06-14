# nn-model-reporting Specification

> Version: 1.0.0 | Created: 2026-06-14

## Purpose

定义神经网络（M4）模型训练报告能力。报告需含所有模型指标（MLP/CNN/LSTM × 二分类/多分类）、与 M3 基线（DT/RF）对比、SMOTE 效果对比、局限诚实记录。

## ADDED Requirements

### Requirement: 神经网络训练报告结构

模块 SHALL 生成 `docs/model_report_mlp_dl.md` 报告，含以下章节：

#### Scenario: 报告章节完整（Happy Path）

- **WHEN** `scripts/train_m4.py` 训练完成
- **THEN** 生成的报告包含章节：1.概述 / 2.数据 / 3.MLP 二分类基线 / 4.MLP 调优 / 5.MLP 多分类 / 6.CNN / 7.LSTM / 8.SMOTE 效果 / 9.与 M3 对比 / 10.局限诚实记录 / 11.结论 / 12.模型持久化清单

#### Scenario: 多分类双数字报告（Edge）

- **WHEN** 报告 MLP 多分类结果
- **THEN** 必须同时给出"全量 23 类 acc"和"已知类 acc（剔除 16 unseen 攻击）"两个数字

#### Scenario: SMOTE 对比章节（Edge）

- **WHEN** 报告 SMOTE 效果
- **THEN** 包含 "SMOTE 前 vs 后" 的 per-class recall 对比表，重点关注 U2R / R2L

### Requirement: 评估指标完整性

每个模型的报告 SHALL 含以下指标：

#### Scenario: 二分类指标（Happy Path）

- **WHEN** 报告二分类结果
- **THEN** 包含：accuracy / precision / recall / f1 / auc / 混淆矩阵

#### Scenario: 多分类指标（Happy Path）

- **WHEN** 报告多分类结果
- **THEN** 包含：accuracy / f1_macro / per-class f1 字典 / 全量 vs 已知类 acc 双数字

#### Scenario: 调优对比表（Edge）

- **WHEN** 报告 MLP 调优结果
- **THEN** 包含 baseline vs tuned 指标对比表 + best_params + CV f1 vs val f1 双数字

### Requirement: 诚实记录局限

报告 SHALL 诚实记录局限，不裁剪不回避。

#### Scenario: tabular CNN/LSTM 局限（Edge）

- **WHEN** 报告 CNN/LSTM 结果
- **THEN** 包含 "tabular CNN/LSTM 优势不明确 / NSL-KDD 样本独立无时序关系" 等诚实说明

#### Scenario: 多分类全量 acc 接近随机（Sad Path）

- **WHEN** MLP 多分类全量 acc 仍接近 1/23 = 0.043
- **THEN** 不裁剪，诚实记录"全量 acc 失败，强烈需要 OvR 拆解或半监督方法"，并提供已知类 acc 缓解

#### Scenario: SMOTE 副作用（Edge）

- **WHEN** SMOTE 后大类 acc 下降
- **THEN** 报告诚实记录副作用，提供 baseline vs SMOTE 两个模型

### Requirement: 与 M3 对比

报告 SHALL 含 M3（DT/RF） vs M4（MLP/CNN/LSTM）的对比章节。

#### Scenario: 二分类对比（Happy Path）

- **WHEN** 报告对比章节
- **THEN** 包含 DT/RF/MLP/CNN/LSTM 的二分类指标对比表（accuracy / f1 / auc）

#### Scenario: 多分类对比（Edge）

- **WHEN** 报告多分类对比
- **THEN** 包含 RF vs MLP 的全量 acc 和已知类 acc 对比

#### Scenario: 对比结论（Edge）

- **WHEN** 报告 M4 是否突破 M3 多分类痛点
- **THEN** 明确说明 yes/no + 原因（不模糊）
