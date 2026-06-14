# model-persistence Specification (M4 Delta)

> Version: 1.1.0 | Created: 2026-06-14（M3 → M4 delta）
> Modified by: m4-neural-network change

## Purpose

扩展模型持久化能力，新增 PyTorch `torch.save` 后端支持，与现有 sklearn joblib 后端共存但职责分明（joblib 负责 sklearn，torch.save 负责 PyTorch，符合 ADR-005 职责分离原则）。

## ADDED Requirements

### Requirement: PyTorch 模型持久化

模块 SHALL 提供 `save_torch_model(model, path)` 和 `load_torch_model(model_class, path, **model_kwargs)` 函数。

#### Scenario: 保存 state_dict（Happy Path）

- **WHEN** 调用 `save_torch_model(model, "outputs/models/mlp_binary_best.pt")`
- **THEN** 模型参数以 `state_dict` 格式保存到 `.pt` 文件

#### Scenario: 加载 state_dict 还原（Happy Path）

- **WHEN** 调用 `load_torch_model(MLPClassifier, "outputs/models/mlp_binary_best.pt", input_dim=20, output_dim=2)`
- **THEN** 返回原始模型实例，`model.load_state_dict()` 加载参数，模型可调用 `.predict(X)` 或前向传播

#### Scenario: save/load 一致性（Edge）

- **WHEN** 保存模型后加载，对相同输入预测
- **THEN** 输出结果与保存前一致（数值精度范围内）

#### Scenario: 文件不存在（Sad Path）

- **WHEN** 调用 `load_torch_model` 但 path 不存在
- **THEN** 抛出 `FileNotFoundError` 含 path 信息

#### Scenario: 架构不匹配（Sad Path）

- **WHEN** 加载时传入的 `model_kwargs` 与保存的 state_dict 维度不匹配
- **THEN** `model.load_state_dict()` 抛出 `RuntimeError` 含 size mismatch 信息

### Requirement: M4 最佳模型持久化

模块 SHALL 提供 `save_best_nn_models(output_dir)` 整合函数，保存 M4 训练的 6-9 个神经网络模型。

#### Scenario: 6 个核心模型保存（Happy Path）

- **WHEN** 调用 `save_best_nn_models(output_dir)`
- **THEN** 保存 6 个 `.pt` 文件：
  - `mlp_binary_best.pt` / `mlp_binary_tuned.pt`
  - `mlp_multiclass_best.pt`
  - `cnn_binary_best.pt` / `lstm_binary_best.pt`
  - `mlp_multiclass_smote.pt`（SMOTE 后）

#### Scenario: 输出目录自动创建（Edge）

- **WHEN** output_dir 不存在
- **THEN** 自动创建目录（沿用 M3 `save_best_models` 行为）

#### Scenario: 部分模型未训练（Edge）

- **WHEN** 调用 `save_best_nn_models` 但只有 4 个模型已训练
- **THEN** 保存已训练的 4 个，跳过未训练的并打印警告日志
