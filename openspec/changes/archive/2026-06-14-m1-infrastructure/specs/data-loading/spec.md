# data-loading Spec

> Version: 1.0.0 | Last Updated: 2026-06-14

## ADDED Requirements

### Requirement: NSL-KDD 41 特征名常量

模块 SHALL 定义 41 列 NSL-KDD 特征名的硬编码常量，按 NSL-KDD 官方论文顺序排列。

#### Scenario: 特征名数量正确

- **WHEN** 读取 `src/data/loader.py` 的特征名常量
- **THEN** 列表长度恰好为 41（不含 label 和 difficulty）

#### Scenario: 特征名顺序符合 NSL-KDD 规范

- **WHEN** 对比特征名常量与 NSL-KDD 官方文档
- **THEN** 顺序完全一致（前 9 个为 TCP 连接基本特征：duration, protocol_type, service, flag, src_bytes, dst_bytes, land, wrong_fragment, urgent）

### Requirement: 加载训练集

`load_train()` 函数 SHALL 从 `dataset/KDDTrain+.txt` 加载训练集，返回带正确列名的 pandas DataFrame。

#### Scenario: 训练集加载成功（Happy Path）

- **WHEN** 调用 `loader.load_train()` 且 `dataset/KDDTrain+.txt` 存在
- **THEN** 返回 DataFrame，shape = (125973, 43)，列名包含 41 特征 + `label` + `difficulty`

#### Scenario: 训练集文件不存在（Sad Path）

- **WHEN** 调用 `loader.load_train()` 且 `dataset/KDDTrain+.txt` 不存在
- **THEN** 抛出 `FileNotFoundError`，错误消息含路径提示

#### Scenario: 训练集列数验证（Edge）

- **WHEN** 加载成功后断言 `df.shape[1] == 43`
- **THEN** 断言通过；若数据源变更导致列数变化，断言失败提示

### Requirement: 加载测试集

`load_test()` 函数 SHALL 从 `dataset/KDDTest+.txt` 加载测试集，返回带正确列名的 pandas DataFrame。

#### Scenario: 测试集加载成功（Happy Path）

- **WHEN** 调用 `loader.load_test()` 且 `dataset/KDDTest+.txt` 存在
- **THEN** 返回 DataFrame，shape = (22544, 43)，列名包含 41 特征 + `label` + `difficulty`

#### Scenario: 测试集文件不存在（Sad Path）

- **WHEN** 调用 `loader.load_test()` 且 `dataset/KDDTest+.txt` 不存在
- **THEN** 抛出 `FileNotFoundError`，错误消息含路径提示

### Requirement: 文件编码处理

加载函数 SHALL 使用 `latin-1` 编码读取 KDD 数据文件。

#### Scenario: Latin-1 编码读取

- **WHEN** 加载含特殊字符的 KDD 文件
- **THEN** 成功读取，无 `UnicodeDecodeError`

### Requirement: 数据文件无 header

加载函数 SHALL 在数据文件无 header 的前提下，正确为每列赋予预定义名称。

#### Scenario: 无 header 自动命名

- **WHEN** 加载无 header 的 txt 文件
- **THEN** DataFrame 列名匹配 41 特征 + label + difficulty，且不出现 `Unnamed: 0` 类默认列名