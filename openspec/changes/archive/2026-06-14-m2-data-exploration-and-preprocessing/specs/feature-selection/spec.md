# feature-selection Spec

> Version: 1.0.0 | Last Updated: 2026-06-14

## ADDED Requirements

### Requirement: 方差阈值过滤

模块 SHALL 提供方差阈值过滤函数，去除近零方差特征。

#### Scenario: 过滤低方差特征（Happy Path）

- **WHEN** 调用 `variance_threshold_filter(X, threshold=0.01)`
- **THEN** 返回保留的列名 list（方差 < threshold 的列被过滤）

#### Scenario: 阈值参数化（Edge）

- **WHEN** 调整 threshold 参数（0.001 / 0.1）
- **THEN** 过滤强度对应变化（threshold 越小保留越多列）

### Requirement: 随机森林特征重要度

模块 SHALL 提供基于 RF 的特征重要度评分函数。

#### Scenario: RF 训练并返回重要度（Happy Path）

- **WHEN** 调用 `rf_feature_importance(X, y, n_estimators=100, random_state=42)`
- **THEN** 返回 pd.Series，索引为列名，值为 feature_importances_

#### Scenario: Top-K 特征选择（Edge）

- **WHEN** 调用 `select_top_k_features(importance, k=20)`
- **THEN** 返回 Top-K 列名 list，按重要度降序

### Requirement: 双保险特征选择管线

模块 SHALL 提供 `feature_selection_pipeline()` 整合函数，先方差阈值再 RF Top-K。

#### Scenario: 一站式调用（Happy Path）

- **WHEN** 调用 `feature_selection_pipeline(X_train, y_train, top_k=20)`
- **THEN** 返回保留列名 list + RF 训练模型（供后续 M3 复用）

#### Scenario: 训练集 fit + 测试集 transform（Edge）

- **WHEN** 用训练集 fit，对测试集 transform
- **THEN** 测试集保留与训练集相同的列（无列错位）