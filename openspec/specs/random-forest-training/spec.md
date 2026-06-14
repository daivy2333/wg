# random-forest-training Specification

## Purpose
TBD - created by archiving change m3-traditional-ml. Update Purpose after archive.
## Requirements
### Requirement: RF 二分类基线训练

模块 SHALL 提供 RandomForestClassifier 二分类基线训练函数。

#### Scenario: 基线模型训练成功（Happy Path）

- **WHEN** 调用 `train_rf_binary(X_train, y_train, random_state=42)`
- **THEN** 返回训练好的 `RandomForestClassifier`，accuracy ≥ 0.85

#### Scenario: RF 包含多棵树（Edge）

- **WHEN** 检查 model.n_estimators
- **THEN** n_estimators ≥ 100

### Requirement: RF 多分类训练

模块 SHALL 提供多分类训练函数。

#### Scenario: 多分类基线训练（Happy Path）

- **WHEN** 调用 `train_rf_multiclass(X_train_multi, y_train_multi, random_state=42)`
- **THEN** 返回训练好的模型，accuracy ≥ 0.78

### Requirement: RF GridSearchCV 调优

模块 SHALL 提供 RF 超参数网格搜索函数。

#### Scenario: 网格搜索 24 组合（Happy Path）

- **WHEN** 调用 `grid_search_rf(X_train, y_train, task='binary')`
- **THEN** 返回 (best_model, best_params, best_score)

#### Scenario: 最优参数覆盖关键超参数（Happy Path）

- **WHEN** 检查 best_params
- **THEN** 包含 n_estimators / max_depth / min_samples_split 三个键

### Requirement: RF 特征重要度

模块 SHALL 提供 RF 特征重要度提取函数。

#### Scenario: Top-K 重要特征（Happy Path）

- **WHEN** 调用 `get_top_k_features(model, k=20)`
- **THEN** 返回 pd.Series，按 feature_importances_ 降序

#### Scenario: 重要度之和 ≈ 1（Edge）

- **WHEN** 检查 sum(feature_importances_)
- **THEN** abs(sum - 1.0) < 1e-6

### Requirement: 4 大攻击类别 F1 对比

模块 SHALL 提供按攻击大类拆解的 F1 评估函数。

#### Scenario: F1 按 4 大类拆解（Happy Path）

- **WHEN** 调用 `f1_by_category(model, X_test, y_test_raw, label_mapping)`
- **THEN** 返回 dict 含 DoS/Probe/R2L/U2R 各自的 F1

