# ml-model-reporting Specification

## Purpose
TBD - created by archiving change m3-traditional-ml. Update Purpose after archive.
## Requirements
### Requirement: M3 模型训练报告

模块 SHALL 产出 `docs/model_report_dt_rf.md`，含 DT/RF 训练全过程。

#### Scenario: 报告章节完整（Happy Path）

- **WHEN** 检查 `docs/model_report_dt_rf.md`
- **THEN** 必须包含：1. 概述 / 2. 数据 / 3. DT 基线 / 4. DT 调优 / 5. RF 基线 / 6. RF 调优 / 7. 4 大攻击类别 F1 对比 / 8. 结论

#### Scenario: 数据章节含数据规模（Happy Path）

- **WHEN** 阅读第 2 章
- **THEN** 含训练集/测试集 shape + Top-20 特征名

#### Scenario: DT 基线章节含基线指标（Happy Path）

- **WHEN** 阅读第 3 章
- **THEN** 含 DT 二分类/多分类基线 accuracy + 5 指标

#### Scenario: DT 调优章节含最优参数（Happy Path）

- **WHEN** 阅读第 4 章
- **THEN** 含 GridSearchCV 最优参数 + 调优前后对比

#### Scenario: RF 调优章节含最优参数（Happy Path）

- **WHEN** 阅读第 6 章
- **THEN** 含 GridSearchCV 最优参数 + Top-20 特征重要度图

#### Scenario: 4 大攻击类别 F1 对比（Happy Path）

- **WHEN** 阅读第 7 章
- **THEN** 含 DT/RF 在 DoS/Probe/R2L/U2R 上的 F1 对比表

#### Scenario: 结论章节含 M4/M5 启示（Edge）

- **WHEN** 阅读第 8 章
- **THEN** 含对 M4（MLP/DL）和 M5（对比分析）的建议

### Requirement: M3 训练脚本

模块 SHALL 提供 `scripts/train_m3.py` 一键跑完 DT + RF 基线 + 调优 + 持久化。

#### Scenario: 脚本全流程执行成功（Happy Path）

- **WHEN** 运行 `/home/daivy/miniconda3/bin/python scripts/train_m3.py`
- **THEN** 返回 0 退出码，4 个模型持久化 + 报告生成

