## Context

**当前状态**：
- M1+M2+M3 已完成：M1 搭建代码骨架，M2 产出 EDA + 8 个预处理 .pkl（M1 整合管线 + M2 异常值/特征选择），M3 训练 DT/RF 并产出 4 个 .joblib 模型 + 完整报告
- `outputs/processed/*.pkl` 共 8 个文件：X_train/test × {binary, multi}（20 维 Top-K 特征，M2 RF 特征选择产出）
- `src/data/loader.py` + `preprocessor.py` + `outlier.py` + `feature_selector.py` + `persistence.py` 已稳定
- `src/models/decision_tree.py` + `random_forest.py` + `persistence.py`（joblib 后端）已稳定
- PyTorch 2.9.0 + torchvision 0.26.0 已安装（conda 环境）
- 踩坑-003 WSL OMP 防护已落地（`tests/conftest.py` 强制 OMP=1/MKL=1/OPENBLAS=1）

**约束**：
- 复用 M2 产出数据（20 维 Top-K 特征），不重做特征工程
- 三层防御：测试 `torch.set_num_threads(1)`，生产脚本用 `n_jobs`-like 智能公式
- 模型持久化职责分离：PyTorch `torch.save` 与 sklearn joblib 共存于 `src/models/persistence.py`，调用接口不同（不混用）
- 输出诚实：全量 23 类 acc + 已知类 acc（剔除 16 unseen）双数字
- 训练脚本用 `.py` 而非 `.ipynb`（ADR-004）

**利益相关方**：
- C同学（神经网络 + 对比分析）— 主导实施
- B同学（传统 ML）— M3 产出为基线对比
- A同学（数据）— M2 产出为输入
- M5 对比分析 — 依赖本变更产出所有模型指标

## Goals / Non-Goals

**Goals:**
1. 训练 PyTorch MLP 二分类基线（acc ≥ 0.85 / f1 ≥ 0.80 / auc ≥ 0.90）
2. MLP 参数调优后 f1 提升 ≥ 2%
3. MLP 多分类报告双数字（全量 + 已知类）
4. CNN 二分类模型（1D Conv，tabular 适配）
5. LSTM 二分类模型（论文对比维度，诚实记录 NSL-KDD 无时序局限）
6. SMOTE 仅对 U2R/R2L 过采样，recall 显著提升
7. 全部模型持久化（`torch.save` + state_dict），加载后预测一致
8. 输出 `docs/model_report_mlp_dl.md` 完整报告（含局限诚实记录）

**Non-Goals:**
- 不替换 M3 DT/RF 训练流程（独立基线）
- 不重做 M2 特征工程（直接复用 20 维特征）
- 不引入 Transformer / Attention 等新架构（PyTorch 标准 MLP/CNN/LSTM 即可）
- 不实现端到端 GPU 训练（CPU 环境足够，125973 样本 + 20 维 → 单 epoch < 5s）
- 不修改 `src/data/loader.py` / `preprocessor.py`（M1/M2 稳定）

## Decisions

### 决策 1: PyTorch 自建而非 sklearn MLPClassifier

- **选择**: PyTorch 2.9.0 + `nn.Module` 自建
- **原因**:
  1. ADR-001 已明确 PyTorch 为深度学习技术栈
  2. 平滑扩展到 CNN/LSTM（sklearn 无此能力）
  3. PyTorch state_dict 持久化是工业标准
  4. 学习价值高，符合 C同学神经网络方向
- **替代方案 A**: sklearn `MLPClassifier` — 快速稳定但无法扩展 CNN/LSTM，违反 ADR-001
- **替代方案 B**: 两者都做 — 工作量翻倍，O-NN-01/O-NN-02 在两套框架都要验证

### 决策 2: MLP 架构为 (20→128→64→{2|23})

- **选择**: 输入层 20 维 → 隐藏层 1 (128, ReLU + Dropout 0.3) → 隐藏层 2 (64, ReLU + Dropout 0.3) → 输出层
- **原因**:
  1. 20 维输入 + 二分类输出（logits=2）/ 多分类输出（logits=23）
  2. 两层隐藏（128→64）属"浅而宽"模式，适合 tabular 数据
  3. Dropout 0.3 缓解过拟合（M3 痛点：CV 0.99 vs test 0.78）
- **替代方案 A**: 三层隐藏（128→64→32）— 更深但梯度消失风险高，20 维输入不需要
- **替代方案 B**: 单层 256 — 简单但欠拟合风险（M2 EDA 显示 Top-5 特征贡献 52%，需要非线性）
- **调优空间**: hidden_layer_sizes 网格 = [(64,), (128, 64), (256, 128, 64), (128, 128, 128)]

### 决策 3: CNN 使用 1D 卷积

- **选择**: Conv1D(in=1, out=16, kernel=3) → ReLU → MaxPool1d(2) → Flatten → Linear → 输出
- **原因**:
  1. tabular 数据 (20 维) 视为长度 20 的 1D 序列
  2. 1D Conv 提取局部特征组合（如连续特征交互）
  3. kernel=3 滑动窗口抓取相邻特征关系
- **替代方案 A**: 2D Conv (4×5 reshape) — 强行制造空间结构，无意义
- **替代方案 B**: TabCNN（specific tabular CNN 架构）— 文献支持但实现复杂，偏离主线
- **诚实记录**: tabular CNN 优势不明确，论文需注明

### 决策 4: LSTM 将样本当序列（length=1, features=20）

- **选择**: LSTM(input_size=20, hidden_size=32, num_layers=1) → 取最后 hidden state → Linear → 输出
- **原因**:
  1. NSL-KDD 样本独立无时序，LSTM 理论优势有限
  2. 但作为论文可写点（"LSTM on tabular IDS"）
  3. 实现简单，作为基线对比
- **诚实记录**: 报告需注明"NSL-KDD 样本独立无时序关系，LSTM 优势有限"（不裁剪局限）
- **替代方案**: 不做 — 但任务 4.5 已确认做，工作量可接受

### 决策 5: SMOTE 仅对 U2R/R2L

- **选择**: imbalanced-learn `SMOTE` 仅对训练集 U2R（52 条）和 R2L（995 条）过采样
- **原因**:
  1. 针对性解决 O-NN-01（M3 多分类失败根因：U2R recall 极低）
  2. DoS（45927）/ Probe（11656）样本充足，不过采样避免引入噪声
  3. normal 类（67343）保持原样（负样本足够多）
- **替代方案 A**: 全部 23 类统一过采样 — DoS 45927 条过采样后过拟合风险
- **替代方案 B**: 跳过 SMOTE — 任务 4.6 已规划必做
- **替代方案 C**: SMOTETomek 混合 — 实现复杂，依赖 imbalanced-learn（待安装）
- **依赖**: `pip install imbalanced-learn`（conda 环境）

### 决策 6: torch.save 保存 state_dict 而非整个 model

- **选择**: `torch.save(model.state_dict(), path)` + 加载时 `model.load_state_dict()`
- **原因**:
  1. PyTorch 官方推荐（state_dict 更轻量、跨版本兼容）
  2. 不绑定 Python 类路径，避免 pickle 风险
  3. 与 M3 joblib 职责分明（joblib 存 sklearn，torch.save 存 PyTorch）
- **替代方案 A**: `torch.save(model, path)` 存整个对象 — 体积大、依赖类路径
- **替代方案 B**: ONNX 导出 — 跨框架但本项目不需要

### 决策 7: 测试与生产的三层线程防护

- **选择**:
  1. 测试：`torch.set_num_threads(1)` + 测试用例小数据
  2. 生产脚本：`N_THREADS = min(4, max(1, os.cpu_count() // 8))`
  3. `tests/conftest.py` 已设 OMP=1/MKL=1/OPENBLAS=1，新增 `torch.set_num_threads(1)`
- **原因**: 沿用踩坑-003 防御机制，避免 WSL 断开
- **替代方案**: 不设 — 历史教训（pytest 触发 WSL OOM）

## Risks / Trade-offs

| Risk | 概率 | 影响 | Mitigation |
|------|------|------|------------|
| 神经网络训练不收敛 | 中 | 中 | 调低学习率（1e-3 → 1e-4），加 BatchNorm，监控 loss 曲线 |
| WSL 断开（多线程 OOM） | 低 | 高 | torch.set_num_threads(1) 测试 + 智能公式调节生产 |
| imbalanced-learn 依赖未安装 | 低 | 低 | scripts/train_m4.py 启动时检查 import，无则提示 `pip install imbalanced-learn` |
| LSTM 在 tabular 数据表现差 | 高 | 低 | 已在决策 4 诚实记录，报告注明"NSL-KDD 无时序" |
| SMOTE 引入噪声反而降低大类 acc | 中 | 中 | 对比实验：保留 baseline 模型，论文报告双数字 |
| 多分类全量 acc 仍接近 0.05 | 高 | 中 | 报告"已知类 acc"独立数字，避免单数字误导（M3 教训） |
| MLP 调优耗时（30+ 组合） | 中 | 中 | 网格搜索用 3-fold CV 而非 5-fold，先粗调再细调 |
| torch.save 模型与训练代码版本不兼容 | 低 | 中 | 持久化测试：save → load → predict 一致性验证 |

## Migration Plan

**部署步骤**:
1. 写新文件（不修改 M3 已有）：`src/models/mlp.py` + `cnn.py` + `lstm.py` + `src/data/smote.py` + `scripts/train_m4.py`
2. 扩展 `src/models/persistence.py`：新增 `save_torch_model` / `load_torch_model`，保留 joblib 函数
3. 写测试：`tests/test_mlp.py` + `test_cnn.py` + `test_lstm.py` + `test_smote.py` + `test_nn_persistence.py`
4. 跑测试：`pytest tests/test_mlp.py tests/test_cnn.py tests/test_lstm.py tests/test_smote.py tests/test_nn_persistence.py -v`
5. 跑训练：`python scripts/train_m4.py`（含 SMOTE 效果对比）
6. 验证产出：`outputs/models/*.pt` 共 6-9 个文件
7. 撰写 `docs/model_report_mlp_dl.md`

**回滚策略**:
- 删 4 个新增 src/models/* + 1 个 src/data/smote.py + scripts/train_m4.py
- `git checkout src/models/persistence.py`（恢复 M3 版本）
- 删新增测试文件
- 删 `docs/model_report_mlp_dl.md`
- 删 `outputs/models/*.pt`（保留 .joblib）

## Open Questions

1. **imbalanced-learn 安装方式**: `pip install` 还是 `conda install`？默认 `pip install imbalanced-learn`（conda 已激活的 Python 3.13.5 环境）
2. **学习率调度器**: 用 `ReduceLROnPlateau` 还是固定 lr？默认 ReduceLROnPlateau（验证 loss 不降时减半）
3. **早停 (Early Stopping)**: 是否启用？默认启用，patience=5，避免过拟合（M3 痛点）
4. **训练集/验证集划分**: 8:2 还是 M3 经验？默认 8:2（125973 → 100778 train + 25195 val）
5. **batch_size**: 256 / 512 / 1024？默认 256（与样本量匹配）
