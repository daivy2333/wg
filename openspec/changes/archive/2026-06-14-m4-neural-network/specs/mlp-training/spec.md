# mlp-training Specification

> Version: 1.0.0 | Created: 2026-06-14

## Purpose

定义基于 PyTorch 自建的 MLP（多层感知器）神经网络训练能力，覆盖二分类基线、参数调优、多分类训练三种模式。输入 20 维 Top-K 特征（M2 产出），输出二分类（normal/anomaly）或 23 类多分类。

## ADDED Requirements

### Requirement: MLP 模型定义

模块 SHALL 提供 `MLPClassifier` PyTorch `nn.Module` 类，接受可变隐藏层配置。

#### Scenario: 默认架构实例化（Happy Path）

- **WHEN** 调用 `MLPClassifier(input_dim=20, output_dim=2)`
- **THEN** 返回一个 `nn.Module` 实例，输入层 20 维，输出层 2 维（logits），默认隐藏层为 (128, 64) + ReLU + Dropout 0.3

#### Scenario: 自定义隐藏层（Happy Path）

- **WHEN** 调用 `MLPClassifier(input_dim=20, output_dim=2, hidden_dims=(256, 128, 64), dropout=0.5)`
- **THEN** 模型包含 3 个隐藏层（256, 128, 64）+ ReLU + Dropout 0.5

#### Scenario: 前向传播（Happy Path）

- **WHEN** 调用 `model(torch.randn(32, 20))` 其中 32 是 batch_size
- **THEN** 输出形状为 `(32, 2)` 的 logits 张量

### Requirement: MLP 二分类训练

模块 SHALL 提供 `train_mlp_binary(X_train, y_train, X_val, y_val, **kwargs)` 函数，返回训练好的模型和评估指标。

#### Scenario: 训练完成（Happy Path）

- **WHEN** 用 100778 条训练样本 + 25195 条验证样本调用 `train_mlp_binary`（默认 50 epoch, lr=1e-3, batch_size=256）
- **THEN** 返回 `(model, metrics)` 其中 metrics 含 `val_accuracy` / `val_f1` / `val_auc` / `val_precision` / `val_recall`

#### Scenario: 早停触发（Edge）

- **WHEN** 验证集 f1 在 5 个连续 epoch 内未提升
- **THEN** 训练停止并返回最佳 epoch 的模型（patience=5）

#### Scenario: 训练不收敛（Sad Path）

- **WHEN** 训练 50 epoch 后验证集 acc 仍 < 0.6
- **THEN** 打印警告日志并返回当前模型（不抛异常，便于人工干预）

### Requirement: MLP 参数调优

模块 SHALL 提供 `grid_search_mlp(X_train, y_train, X_val, y_val, param_grid)` 函数，对 hidden_layer_sizes / activation / alpha / learning_rate 做网格搜索。

#### Scenario: 网格搜索完成（Happy Path）

- **WHEN** 调用 `grid_search_mlp` 用 4 组 hidden_dims × 2 组 activation × 2 组 alpha × 2 组 learning_rate = 32 组合
- **THEN** 返回 `(best_model, best_params, best_cv_score)`，按验证集 f1 排序

#### Scenario: CV vs Test 双数字记录（Happy Path）

- **WHEN** 网格搜索选出 best_params
- **THEN** 报告 `cv_f1`（在训练集上）和 `val_f1`（在验证集上）两个数字（避免 M3 CV 0.99 vs test 0.78 痛点）

#### Scenario: 调优后提升不足（Sad Path）

- **WHEN** 调优后 val_f1 相对 baseline 提升 < 1%
- **THEN** 打印警告并保留 baseline 模型

### Requirement: MLP 多分类训练

模块 SHALL 提供 `train_mlp_multiclass(X_train, y_train, X_val, y_val, num_classes=23, **kwargs)` 函数。

#### Scenario: 23 类训练（Happy Path）

- **WHEN** 用 23 类标签（0-22）调用 `train_mlp_multiclass`
- **THEN** 返回 `(model, metrics)` 含 `val_accuracy` / `val_f1_macro` / `per_class_f1` 字典

#### Scenario: 已知类准确率（Edge）

- **WHEN** 调用 `compute_known_class_accuracy(model, X_test, y_test, known_classes=range(0, 22), unseen_classes=[22])`
- **THEN** 返回剔除 16 unseen 攻击后的已知类 acc（23 - 16 = 7 已知类，注：实际为 23 - 16 unseen 训练中存在的类 = 7 + 0 = ... 见具体映射）

#### Scenario: class_weight balanced（Edge）

- **WHEN** 设置 `class_weight='balanced'`
- **THEN** CrossEntropyLoss 的 weight 参数按 1/class_frequency 计算，缓解不均衡

### Requirement: MLP 数据集与训练器

模块 SHALL 提供 PyTorch `Dataset` 和 `DataLoader` 工具函数。

#### Scenario: NSLKDDDataset 实例化（Happy Path）

- **WHEN** 调用 `NSLKDDDataset(X, y)` 其中 X.shape=(N, 20) y.shape=(N,)
- **THEN** 返回 Dataset，`__len__` 返回 N，`__getitem__(i)` 返回 `(X[i], y[i])` 张量

#### Scenario: DataLoader 批处理（Happy Path）

- **WHEN** 调用 `DataLoader(dataset, batch_size=256, shuffle=True)`
- **THEN** 每个 batch 输出 `(X_batch, y_batch)` 形状 `(256, 20)` 和 `(256,)`
