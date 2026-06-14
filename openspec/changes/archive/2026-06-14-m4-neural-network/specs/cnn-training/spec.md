# cnn-training Specification

> Version: 1.0.0 | Created: 2026-06-14

## Purpose

定义基于 PyTorch 自建的 1D 卷积神经网络（CNN）训练能力。tabular 数据（20 维）被视为长度 20 的 1D 序列，Conv1D 提取局部特征交互模式。

## ADDED Requirements

### Requirement: 1D CNN 模型定义

模块 SHALL 提供 `CNN1DClassifier` PyTorch `nn.Module` 类，1D 卷积 + 池化 + 全连接架构。

#### Scenario: 默认架构实例化（Happy Path）

- **WHEN** 调用 `CNN1DClassifier(input_length=20, output_dim=2)`
- **THEN** 返回 `nn.Module` 实例，架构：Conv1d(1→16, kernel=3, padding=1) → ReLU → MaxPool1d(2) → Flatten → Linear(16*10 → output_dim)

#### Scenario: 自定义卷积核（Happy Path）

- **WHEN** 调用 `CNN1DClassifier(input_length=20, output_dim=2, conv_channels=(32, 64), kernel_size=5)`
- **THEN** 模型包含 2 层 Conv1d（1→32→64），kernel=5，全连接输入维度相应调整

#### Scenario: 前向传播（Happy Path）

- **WHEN** 调用 `model(torch.randn(32, 1, 20))` 其中 (32, 1, 20) = (batch, channels=1, length=20)
- **THEN** 输出形状为 `(32, output_dim)` 的 logits 张量

### Requirement: CNN 二分类训练

模块 SHALL 提供 `train_cnn_binary(X_train, y_train, X_val, y_val, **kwargs)` 函数。

#### Scenario: 训练完成（Happy Path）

- **WHEN** 用训练集调用 `train_cnn_binary`（默认 50 epoch, lr=1e-3, batch_size=256）
- **THEN** 返回 `(model, metrics)` 含 val_accuracy / val_f1 / val_auc

#### Scenario: tabular CNN 局限（Edge）

- **WHEN** CNN 在 NSL-KDD 20 维 tabular 数据上训练
- **THEN** 报告需注明"tabular CNN 优势不明确，仅作为论文对比维度"

#### Scenario: 输出形状不匹配（Sad Path）

- **WHEN** 输入数据维度不是 (N, 20) 而是 (N, 41) 等其他维度
- **THEN** 抛出 `ValueError` 含明确错误信息

### Requirement: CNN 输入维度校验

模块 SHALL 在训练前校验输入数据 shape。

#### Scenario: 输入维度不匹配（Sad Path）

- **WHEN** 输入数据维度不是 (N, 20) 而是 (N, 41) 等其他维度
- **THEN** 抛出 `ValueError` 含明确错误信息（"CNN1DClassifier 需要 input_length=20，当前为 {actual}"）
