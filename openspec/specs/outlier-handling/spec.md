# outlier-handling Specification

## Purpose
TBD - created by archiving change m2-data-exploration-and-preprocessing. Update Purpose after archive.
## Requirements
### Requirement: IQR 异常值检测

模块 SHALL 提供 IQR 异常值检测函数，识别每列超限值数量。

#### Scenario: 检测数值列异常值（Happy Path）

- **WHEN** 调用 `detect_outliers_iqr(df, columns)` 处理含极端值的 DataFrame
- **THEN** 返回 dict，键为列名，值为超限值数量（Q1 - 1.5*IQR 之外或 Q3 + 1.5*IQR 之外）

#### Scenario: 非数值列跳过（Edge）

- **WHEN** columns 参数含 OneHot/0-1 特征
- **THEN** 跳过该列（不参与 IQR 计算）

### Requirement: IQR + clip 异常值替换

模块 SHALL 提供 clip 函数，将超限值替换为边界值（不丢样本）。

#### Scenario: clip 超限值到边界（Happy Path）

- **WHEN** 调用 `clip_outliers(df, columns)`
- **THEN** 超限值被替换为 Q1 - 1.5*IQR（下界）或 Q3 + 1.5*IQR（上界），行数保持不变

#### Scenario: clip 后无 NaN（Sad Path 防御）

- **WHEN** 调用 `clip_outliers(df, columns)` 后检查 NaN
- **THEN** 无新增 NaN（clip 不会引入缺失）

### Requirement: log1p 长尾变换

模块 SHALL 提供 log1p 变换函数，处理数值长尾特征。

#### Scenario: log1p 变换 src_bytes 等长尾特征（Happy Path）

- **WHEN** 调用 `log1p_transform(df, columns=['src_bytes', 'dst_bytes', 'count', 'srv_count'])`
- **THEN** 这些列被 `np.log1p()` 变换，列名加 `_log1p` 后缀

#### Scenario: log1p 处理 0 值（Edge）

- **WHEN** 特征含 0 值
- **THEN** `np.log1p(0) = 0`，不产生 -inf 或 NaN

### Requirement: 异常值处理管线整合

模块 SHALL 提供 `outlier_pipeline()` 整合函数，串联 IQR 检测 + clip + log1p 变换。

#### Scenario: 一站式调用（Happy Path）

- **WHEN** 调用 `outlier_pipeline(df, numeric_long_tail_cols)`
- **THEN** 返回处理后 DataFrame + 异常值统计 dict

