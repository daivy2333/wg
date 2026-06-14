# model-persistence Specification

## Purpose
TBD - created by archiving change m3-traditional-ml. Update Purpose after archive.
## Requirements
### Requirement: joblib 模型持久化

模块 SHALL 提供 joblib save/load 函数（sklearn 官方推荐）。

#### Scenario: 保存模型到 joblib（Happy Path）

- **WHEN** 调用 `save_model(model, path)`
- **THEN** 模型被序列化为 `.joblib` 文件

#### Scenario: 加载模型还原（Happy Path）

- **WHEN** 调用 `load_model(path)`
- **THEN** 返回原始 sklearn 模型，可直接调用 .predict()

#### Scenario: 文件不存在抛错（Sad Path）

- **WHEN** 调用 `load_model(path)` 且 path 不存在
- **THEN** 抛出 FileNotFoundError

### Requirement: M3 最佳模型持久化

模块 SHALL 提供 `save_best_models()` 整合函数，保存 DT/RF 二分类+多分类共 4 个模型。

#### Scenario: 4 个模型保存（Happy Path）

- **WHEN** 调用 `save_best_models(dt_bin, rf_bin, dt_multi, rf_multi, output_dir)`
- **THEN** 保存 4 个 `.joblib` 文件：
  - `dt_binary_best.joblib` / `rf_binary_best.joblib`
  - `dt_multiclass_best.joblib` / `rf_multiclass_best.joblib`

#### Scenario: 输出目录自动创建（Edge）

- **WHEN** output_dir 不存在
- **THEN** 自动创建目录

