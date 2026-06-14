# decision-tree-training Spec

> Version: 1.0.0 | Last Updated: 2026-06-14

## ADDED Requirements

### Requirement: DT 二分类基线训练

模块 SHALL 提供 DecisionTreeClassifier 二分类基线训练函数。

#### Scenario: 基线模型训练成功（Happy Path）

- **WHEN** 调用 `train_dt_binary(X_train, y_train, random_state=42)`
- **THEN** 返回训练好的 `DecisionTreeClassifier`，accuracy ≥ 0.80

#### Scenario: 基线模型评估含 5 指标（Happy Path）

- **WHEN** 调用 `evaluate_model(model, X_test, y_test)`
- **THEN** 返回 dict 含 accuracy/precision/recall/f1/auc 5 指标

### Requirement: DT 多分类训练

模块 SHALL 提供多分类训练函数。

#### Scenario: 多分类基线训练（Happy Path）

- **WHEN** 调用 `train_dt_multiclass(X_train_multi, y_train_multi, random_state=42)`
- **THEN** 返回训练好的模型，accuracy ≥ 0.75

#### Scenario: 多分类评估含 f1_macro（Happy Path）

- **WHEN** 评估多分类模型
- **THEN** 返回 dict 含 f1_macro 指标

### Requirement: DT GridSearchCV 调优

模块 SHALL 提供 DT 超参数网格搜索函数。

#### Scenario: 网格搜索 30 组合（Happy Path）

- **WHEN** 调用 `grid_search_dt(X_train, y_train, task='binary')`
- **THEN** 返回 (best_model, best_params, best_score)

#### Scenario: 最优参数覆盖关键超参数（Happy Path）

- **WHEN** 检查 best_params
- **THEN** 包含 max_depth / min_samples_split / criterion 三个键

### Requirement: class_weight='balanced'

所有 DT 模型 SHALL 使用 class_weight='balanced' 应对类别不均衡。

#### Scenario: 模型使用 balanced 权重（Happy Path）

- **WHEN** 训练 DT 模型
- **THEN** model.class_weight == 'balanced'