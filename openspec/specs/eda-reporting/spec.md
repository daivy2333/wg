# eda-reporting Specification

## Purpose
TBD - created by archiving change m2-data-exploration-and-preprocessing. Update Purpose after archive.
## Requirements
### Requirement: EDA 报告生成

模块 SHALL 产出 `docs/eda_report.md`，含 5 大章节。

#### Scenario: 报告章节完整（Happy Path）

- **WHEN** 检查 `docs/eda_report.md` 文件
- **THEN** 必须包含：1. 数据概览 / 2. 特征分析 / 3. 异常值处理 / 4. 特征选择 / 5. 结论

#### Scenario: 数据概览含基本统计（Happy Path）

- **WHEN** 阅读第 1 章
- **THEN** 含训练集/测试集 shape、缺失值、标签分布、攻击大类分布

#### Scenario: 特征分析含分布与相关（Happy Path）

- **WHEN** 阅读第 2 章
- **THEN** 含数值特征分布描述、分类特征分布描述、相关系数热力图引用

#### Scenario: 异常值章节含处理方法（Happy Path）

- **WHEN** 阅读第 3 章
- **THEN** 含 IQR+clip 方法说明 + log1p 变换列清单 + 处理前后对比

#### Scenario: 特征选择章节含 Top-K（Happy Path）

- **WHEN** 阅读第 4 章
- **THEN** 含方差阈值 + RF Top-20 双重过滤说明 + Top-20 特征列表

#### Scenario: 结论章节含 M3/M4 启示（Edge）

- **WHEN** 阅读第 5 章
- **THEN** 含对 M3（DT/RF）和 M4（MLP/DL）的具体建议（如：训练时间预估、关键特征等）

### Requirement: EDA 脚本可从头运行

模块 SHALL 确保 `notebooks/01_data_exploration.py` 可在 conda 环境从头执行到底。

#### Scenario: 脚本全流程执行成功（Happy Path）

- **WHEN** 运行 `/home/daivy/miniconda3/bin/python notebooks/01_data_exploration.py`
- **THEN** 返回 0 退出码，所有 14 个场景执行成功，输出 8 个 PNG 图表到 `outputs/figures/`

#### Scenario: 14 场景全覆盖（Happy Path）

- **WHEN** 检查脚本输出的 14 个场景统计 + 图表文件
- **THEN** 含 14 个分析场景（覆盖 Phase 1.3 Scenario Sketch 全部 17 场景中的 14 个 cell 化场景）

