## Why

M3 传统机器学习（DT/RF）虽二分类表现可用（f1=0.69-0.76），但**多分类接近随机猜测**（acc=0.046），根本原因是训练集 16 种攻击类型在测试集完全 unseen，传统 ML 对高维稀疏 one-hot 类别特征 + unseen 类的组合能力受限。本变更引入 PyTorch 神经网络（MLP + CNN + LSTM）以期：(1) 验证神经网络对 tabular 数据的非线性建模能力能否突破 M3 局限；(2) 配合 SMOTE 处理 U2R/R2L 极小样本不均衡问题；(3) 产出与 M3 的公平对比基线供 M5 分析。

## What Changes

- **新增** `src/models/mlp.py` — PyTorch MLP 模型定义 + 训练/调优/评估函数（二分类+多分类）
- **新增** `src/models/cnn.py` — 1D CNN 模型（Conv1D → 池化 → 全连接）
- **新增** `src/models/lstm.py` — LSTM 模型（样本当序列）
- **新增** `src/data/smote.py` — SMOTE 样本不均衡处理（仅 U2R/R2L）
- **新增** `scripts/train_m4.py` — 端到端训练编排
- **修改** `src/models/persistence.py` — 增加 `save_torch_model` / `load_torch_model`（torch.save 后端，与 joblib 职责分离，符合 ADR-005）
- **新增** 6 个测试文件 + 1 个持久化测试
- **新增** 6-9 个 `.pt` 模型文件到 `outputs/models/`
- **新增** `docs/model_report_mlp_dl.md` 完整训练报告

## Capabilities

### New Capabilities

- `mlp-training`: PyTorch 自建 MLP 模型，包含基线训练 / 网格搜索调优 / 多分类训练三种模式；输入 20 维 Top-K 特征（M2 产出），输出二分类（sigmoid）或 23 类多分类（softmax）
- `cnn-training`: 1D 卷积神经网络，Conv1D(20→16) → MaxPool → Flatten → 全连接 → 输出，tabular 数据适配
- `lstm-training`: LSTM 神经网络，将每个样本的 20 维特征视为长度 1 的序列（注：NSL-KDD 样本独立无时序关系，本能力作为论文对比维度）
- `smote-sampling`: SMOTE 样本过采样实现，仅作用于 U2R（52 条）和 R2L（995 条）小样本类
- `nn-model-reporting`: 神经网络模型报告能力，包含所有模型指标 + 与 M3 基线对比 + 局限诚实记录

### Modified Capabilities

- `model-persistence`: 增加 PyTorch 模型持久化支持（`save_torch_model` / `load_torch_model`），与现有 joblib 函数共存但职责分明（joblib 负责 sklearn，torch.save 负责 PyTorch）

## Impact

**新增代码**:
- `src/models/mlp.py` (~200 行)
- `src/models/cnn.py` (~150 行)
- `src/models/lstm.py` (~150 行)
- `src/data/smote.py` (~80 行)
- `src/models/persistence.py` 扩展 (~40 行)
- `scripts/train_m4.py` (~250 行)
- `tests/test_mlp.py` + `test_cnn.py` + `test_lstm.py` + `test_smote.py` + `test_nn_persistence.py` (共 ~250 行)

**新增数据**:
- `outputs/models/mlp_binary_best.pt`
- `outputs/models/mlp_binary_tuned.pt`
- `outputs/models/mlp_multiclass_best.pt`
- `outputs/models/cnn_binary_best.pt`
- `outputs/models/lstm_binary_best.pt`
- `outputs/models/mlp_multiclass_smote.pt`（SMOTE 后）
- 6-9 个 .pt 文件

**依赖**:
- PyTorch 2.9.0 ✅ 已安装
- torchvision 0.26.0 ✅ 已安装
- imbalanced-learn（待安装，用于 SMOTE）

**风险**:
- 训练时间：MLP/CNN 50 epoch 预计 5-15 分钟（CPU 环境），LSTM 类似
- 显存：20 维特征 + 128 隐藏层 + 125973 训练样本 → 远低于 1GB 显存，CPU 即可
- WSL 断开：必须用 `torch.set_num_threads(1)` 测试 + 智能公式调节生产脚本（沿用踩坑-003）

**回滚方案**:
- 删除新增的 4 个 src/models/* 文件 + 1 个 src/data/smote.py + 1 个 scripts/train_m4.py
- 还原 `src/models/persistence.py` 到 M3 版本（git checkout）
- 删除新增的 6-9 个 .pt 文件
- 删除新增测试文件
- 删除 `docs/model_report_mlp_dl.md`

**不做什么**:
- 不修改 M2 数据加载/预处理/特征选择流程（直接复用 outputs/processed/*.pkl）
- 不修改 M3 训练代码（DT/RF 是独立基线）
- 不引入新框架（PyTorch 已经是 ADR-001 选型）
- 不替换 M3 持久化（joblib 保留，仅扩展 torch.save 接口）
