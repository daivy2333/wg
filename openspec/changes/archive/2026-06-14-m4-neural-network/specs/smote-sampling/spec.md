# smote-sampling Specification

> Version: 1.0.0 | Created: 2026-06-14

## Purpose

定义基于 imbalanced-learn 的 SMOTE 样本不均衡处理能力。针对 M3 多分类失败痛点（U2R 仅 52 条训练样本导致 recall 极低），仅对训练集中的 U2R 和 R2L 小样本类做 SMOTE 过采样，不影响 DoS / Probe / normal 等样本充足的类。

## ADDED Requirements

### Requirement: SMOTE 选择性过采样

模块 SHALL 提供 `apply_smote_to_minority_classes(X_train, y_train, target_classes, sampling_strategy='auto', k_neighbors=5)` 函数，仅对 `target_classes` 列表中的类做 SMOTE。

#### Scenario: 仅 U2R/R2L 过采样（Happy Path）

- **WHEN** 调用 `apply_smote_to_minority_classes(X_train, y_train, target_classes=['U2R', 'R2L'])`
- **THEN** 仅 U2R 和 R2L 类的样本数增加，DoS / Probe / normal 类样本数不变

#### Scenario: U2R 样本数从 52 提升（Happy Path）

- **WHEN** U2R 原始样本数 = 52，R2L = 995
- **THEN** SMOTE 后 U2R 和 R2L 样本数提升（具体目标由 `sampling_strategy` 决定，'auto' 默认提升到多数类样本数）

#### Scenario: SMOTE 引入噪声（Edge）

- **WHEN** U2R 仅 52 条 + k_neighbors=5 → SMOTE 在边界生成合成样本可能引入噪声
- **THEN** 报告需诚实记录此风险，训练对比实验保留 baseline 模型

#### Scenario: imbalanced-learn 未安装（Sad Path）

- **WHEN** `from imblearn.over_sampling import SMOTE` 报 ImportError
- **THEN** 抛出 `ImportError` 含 `pip install imbalanced-learn` 安装提示

### Requirement: SMOTE 训练集对比

模块 SHALL 提供 `compare_with_without_smote(train_fn, X_train, y_train, X_test, y_test)` 函数，对比 SMOTE 前后的模型指标。

#### Scenario: 对比实验完成（Happy Path）

- **WHEN** 调用对比函数，训练两次（一次 SMOTE，一次 baseline）
- **THEN** 返回 `(baseline_metrics, smote_metrics)` 含 U2R / R2L 的 per-class recall / precision / f1

#### Scenario: U2R recall 显著提升（Happy Path）

- **WHEN** SMOTE 后 U2R 的 recall 从 < 0.05 提升到 ≥ 0.20
- **THEN** 在 `docs/model_report_mlp_dl.md` 报告作为 SMOTE 效果证据

#### Scenario: SMOTE 后大类 acc 下降（Edge）

- **WHEN** DoS / Probe 等大类的 acc 下降 > 2%
- **THEN** 报告诚实记录此副作用，论文讨论部分需说明

### Requirement: SMOTE 持久化

SMOTE 生成的合成训练集 SHALL 可保存到磁盘供复用，避免每次重跑都重新生成。

#### Scenario: 保存 SMOTE 数据（Happy Path）

- **WHEN** 调用 `save_smote_data(X_train_smote, y_train_smote, output_path)`
- **THEN** 保存为 `.pkl` 文件（M2 已有的 `src/data/persistence.py` 工具）

#### Scenario: 加载 SMOTE 数据（Happy Path）

- **WHEN** 调用 `load_smote_data(path)`
- **THEN** 返回 `(X_train_smote, y_train_smote)` 与 SMOTE 前维度一致（feature 数不变）
