# lstm-training Specification

> Version: 1.0.0 | Created: 2026-06-14

## Purpose

定义基于 PyTorch 自建的 LSTM 神经网络训练能力。tabular 样本的 20 维特征视为长度 1 的序列输入 LSTM（每个特征作为独立时间步）。注：NSL-KDD 样本独立无时序关系，本能力作为论文对比维度，报告中需诚实记录其理论局限。

## ADDED Requirements

### Requirement: LSTM 模型定义

模块 SHALL 提供 `LSTMClassifier` PyTorch `nn.Module` 类，单层 LSTM + 全连接架构。

#### Scenario: 默认架构实例化（Happy Path）

- **WHEN** 调用 `LSTMClassifier(input_size=20, output_dim=2)`
- **THEN** 返回 `nn.Module` 实例，架构：LSTM(input_size=20, hidden_size=32, num_layers=1, batch_first=True) → 取最后 hidden state → Linear(32 → output_dim)

#### Scenario: 自定义隐藏维度（Happy Path）

- **WHEN** 调用 `LSTMClassifier(input_size=20, output_dim=2, hidden_size=64, num_layers=2, dropout=0.3)`
- **THEN** 模型包含 2 层 LSTM（hidden_size=64），dropout=0.3

#### Scenario: 前向传播（Happy Path）

- **WHEN** 调用 `model(torch.randn(32, 1, 20))` 其中 (32, 1, 20) = (batch, seq_len=1, features=20)
- **THEN** 输出形状为 `(32, output_dim)` 的 logits 张量

### Requirement: LSTM 二分类训练

模块 SHALL 提供 `train_lstm_binary(X_train, y_train, X_val, y_val, **kwargs)` 函数。

#### Scenario: 训练完成（Happy Path）

- **WHEN** 用训练集调用 `train_lstm_binary`（默认 50 epoch, lr=1e-3, batch_size=256）
- **THEN** 返回 `(model, metrics)` 含 val_accuracy / val_f1 / val_auc

#### Scenario: LSTM 局限诚实记录（Edge）

- **WHEN** LSTM 在 NSL-KDD 20 维 tabular 独立样本上训练
- **THEN** 报告需注明"NSL-KDD 样本独立无时序关系，LSTM 优势有限，仅作为论文对比维度"

#### Scenario: 输入序列长度错误（Sad Path）

- **WHEN** 输入数据 shape 不是 (N, 1, 20) 或 reshape 失败
- **THEN** 抛出 `ValueError` 含 reshape 期望信息
