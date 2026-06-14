# data-persistence Spec

> Version: 1.0.0 | Last Updated: 2026-06-14

## ADDED Requirements

### Requirement: pickle 持久化

模块 SHALL 提供 pickle 序列化与加载函数。

#### Scenario: 保存 DataFrame 到 pickle（Happy Path）

- **WHEN** 调用 `save_pickle(obj, path)`
- **THEN** 对象被序列化为二进制文件，文件可被后续 `load_pickle` 还原

#### Scenario: 加载 pickle 还原对象（Happy Path）

- **WHEN** 调用 `load_pickle(path)`
- **THEN** 返回原始对象（DataFrame / ndarray / 任意 pickle 支持类型）

#### Scenario: 文件不存在抛错（Sad Path）

- **WHEN** 调用 `load_pickle(path)` 且 path 不存在
- **THEN** 抛出 FileNotFoundError，错误消息含路径

### Requirement: M2 数据集持久化

模块 SHALL 提供 `save_m2_datasets()` 整合函数，保存二分类 + 多分类双套数据。

#### Scenario: 二分类数据保存（Happy Path）

- **WHEN** 调用 `save_m2_datasets(X_train, X_test, y_train, y_test, output_dir, task='binary')`
- **THEN** 保存 `X_train.pkl` / `X_test.pkl` / `y_train.pkl` / `y_test.pkl` 到 output_dir

#### Scenario: 多分类数据保存（Happy Path）

- **WHEN** 调用同一函数但 `task='multiclass'`
- **THEN** 保存 `X_train_multi.pkl` / `X_test_multi.pkl` / `y_train_multi.pkl` / `y_test_multi.pkl`

#### Scenario: 输出目录自动创建（Edge）

- **WHEN** output_dir 不存在
- **THEN** 函数自动创建目录（含中间层级）