# multiclass-5cat-label

> 5 大类标签生成
> Version: 1.0.0 | Last Updated: 2026-06-16

## ADDED Requirements

### Requirement: 5 类标签生成
`make_labels()` SHALL 支持 `task="multiclass_5cat"`，先以 LabelEncoder 编码为 40 类标签，再通过 `label_id_to_category.json` 映射为 5 类（0-4）。

#### Scenario: 生成 5 类训练标签
- **WHEN** 调用 `make_labels(y_train, task="multiclass_5cat")`
- **THEN** 返回 0-4 整数标签 Series
- **AND** 标签分布为 Normal/DoS/Probe/R2L/U2R 对应 0-4（按字母序）

#### Scenario: 生成 5 类测试标签
- **WHEN** 调用 `make_labels(y_test, task="multiclass_5cat")`
- **THEN** 返回 0-4 整数标签 Series
- **AND** 所有测试样本均落入 5 类（无未见类别）

### Requirement: 预处理管线支持 5 类
`preprocess_pipeline()` SHALL 接受 `task="multiclass_5cat"`，输出 5 类标签。

#### Scenario: 预处理管线生成 5 类数据
- **WHEN** 调用 `preprocess_pipeline(df, task="multiclass_5cat", fit=True)`
- **THEN** y 包含 0-4 整数标签
- **AND** 持久化文件名为 `y_train_multi.pkl`（覆盖原 40 类文件）
