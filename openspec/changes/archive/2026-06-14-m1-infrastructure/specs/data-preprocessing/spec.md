# data-preprocessing Spec

> Version: 1.0.0 | Last Updated: 2026-06-14

## ADDED Requirements

### Requirement: 缺失值处理

preprocessor SHALL 检查并处理缺失值（NSL-KDD 数据集官方宣称无缺失值，但需断言验证）。

#### Scenario: 缺失值检查（Happy Path）

- **WHEN** 对训练/测试集调用 `check_missing(df)`
- **THEN** 返回 `(0, 0)`（缺失值数量为 0）

#### Scenario: 缺失值非零处理（Sad Path）

- **WHEN** 检测到缺失值
- **THEN** 输出警告日志但**不自动填充**，由调用方决定策略（M2 阶段细化）

### Requirement: 分类特征编码（混合策略）

preprocessor SHALL 使用 OneHot + Label 混合策略处理分类特征：

- `protocol_type`（3 种）：OneHotEncoder
- `flag`（11 种）：OneHotEncoder
- `service`（70+ 种高基数）：LabelEncoder

#### Scenario: 低基数 OneHot 编码

- **WHEN** 调用 `encode_categorical(df)` 处理 `protocol_type` 和 `flag`
- **THEN** 两列被替换为多列二进制列（`protocol_type_tcp`, `protocol_type_udp`, `flag_SF` 等）

#### Scenario: 高基数 Label 编码

- **WHEN** 调用 `encode_categorical(df)` 处理 `service`
- **THEN** `service` 列被替换为整数列（0-69 范围），原值域映射保留

#### Scenario: 训练/测试集编码一致（Edge）

- **WHEN** 分别对训练集和测试集调用编码
- **THEN** 测试集的 `service` 整数映射与训练集共用同一 LabelEncoder，避免 unseen label 报错

### Requirement: 数值特征标准化

preprocessor SHALL 使用 `StandardScaler` 对数值特征进行标准化。

#### Scenario: 标准化后均值 ≈ 0（Happy Path）

- **WHEN** 对训练集数值列调用 `standardize(X_train)`
- **THEN** 各列均值在 `[-1e-6, 1e-6]` 范围内

#### Scenario: 标准化后标准差 ≈ 1

- **WHEN** 对训练集数值列调用 `standardize(X_train)`
- **THEN** 各列标准差在 `[1 - 1e-6, 1 + 1e-6]` 范围内

#### Scenario: 训练集 fit + 测试集 transform 一致（Edge）

- **WHEN** 用训练集 fit StandardScaler，对测试集 transform
- **THEN** 测试集使用训练集的均值/标准差，避免数据泄漏

### Requirement: 难度列丢弃

preprocessor SHALL 在最终输出前丢弃 `difficulty` 列。

#### Scenario: difficulty 列被移除

- **WHEN** 调用预处理管线
- **THEN** 输出 DataFrame 不含 `difficulty` 列，但保留 `label` 列

### Requirement: 数据集划分（保留原始 KDD 划分）

preprocessor SHALL 保留 KDDTrain+/KDDTest+ 原始划分，不进行随机再划分。

#### Scenario: 原始划分保留（Edge）

- **WHEN** 调用 `split_data(df_train, df_test)` 或直接返回已分好的数据
- **THEN** 输出 X_train/X_test/y_train/y_test 的样本数与原始一致（125973 / 22544）

### Requirement: 二分类与多分类标签

preprocessor SHALL 同时支持二分类和 23 类多分类标签输出。

#### Scenario: 二分类标签生成

- **WHEN** 调用 `make_labels(df, task='binary')`
- **THEN** 返回 Series，值为 `0`（normal）或 `1`（anomaly 任意攻击）

#### Scenario: 多分类标签生成

- **WHEN** 调用 `make_labels(df, task='multiclass')`
- **THEN** 返回 Series，包含 23 个唯一值（normal + 22 种攻击类型）

### Requirement: 预处理管线整合

preprocessor SHALL 提供 `preprocess_pipeline(df, task='binary')` 整合函数，串联缺失值检查 + 编码 + 标准化 + 标签生成 + 难度列丢弃。

#### Scenario: 一站式调用

- **WHEN** 调用 `preprocess_pipeline(df_train, task='binary')`
- **THEN** 返回 `(X, y)` 元组，X 为处理后特征 DataFrame，y 为标签 Series