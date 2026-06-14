# M4 神经网络模型训练 — 实施任务清单

> Created: 2026-06-14 | Change: m4-neural-network
> Specs: 5 新 + 1 modified（mlp/cnn/lstm/smote/nn-model-reporting + model-persistence delta）
> Design: 7 个核心决策 + 8 个风险 + 5 个开放问题（详见 design.md）

## 1. 环境与基础设施

- [x] 1.1 安装 imbalanced-learn（`pip install imbalanced-learn`）+ 验证 `from imblearn.over_sampling import SMOTE` 成功
- [x] 1.2 扩展 `tests/conftest.py`，新增 `torch.set_num_threads(1)`（沿用踩坑-003 三层防御）
- [x] 1.3 验证 PyTorch 2.9.0 + torchvision 0.26.0 已就绪（`import torch; print(torch.__version__)`）

## 2. MLP 基线模型训练（任务 4.1）

- [x] 2.1 创建 `src/models/mlp.py`：实现 `MLPClassifier(nn.Module)` 类（输入 20 维，输出 2/23 维，可变 hidden_dims + dropout）
- [x] 2.2 实现 `NSLKDDDataset` + `DataLoader` 工具函数（PyTorch Dataset/Dataset 接口）
- [x] 2.3 实现 `train_mlp_binary(X_train, y_train, X_val, y_val, **kwargs)`：含早停（patience=5）+ ReduceLROnPlateau + 50 epoch 默认
- [x] 2.4 创建 `tests/test_mlp.py`：覆盖前向传播 / 默认架构 / 自定义架构 / 训练接口（10+ 测试）
- [x] 2.5 跑测试：`pytest tests/test_mlp.py -v`，要求全部通过（**Gate 5 验证**）
- [x] 2.6 跑小规模验证：取 10000 样本训练，确认 loss 下降 + val_f1 ≥ 0.7（CI smoke test）
- [x] 2.7 验收：binary acc ≥ 0.85 / f1 ≥ 0.80 / auc ≥ 0.90（与 M3 RF acc=0.72 / auc=0.92 对比）

## 3. MLP 参数调优（任务 4.2）

- [x] 3.1 实现 `grid_search_mlp(param_grid)`：hidden_dims × activation × alpha × learning_rate，4×2×2×2=32 组合
- [x] 3.2 调优记录：同时输出 CV f1（训练集 3-fold）和 val f1（验证集）双数字（避免 M3 CV 0.99 vs test 0.78 痛点）
- [x] 3.3 创建 `tests/test_grid_search_mlp.py`：覆盖网格搜索函数 / 评分函数 / best_params 选择逻辑（5+ 测试）
- [x] 3.4 跑测试 + 调优：`pytest tests/test_grid_search_mlp.py -v` + 跑实际调优（约 10-20 分钟）
- [x] 3.5 验收：调优后 val_f1 提升 ≥ 2%

## 4. MLP 多分类训练（任务 4.3）

- [x] 4.1 扩展 `src/models/mlp.py`：实现 `train_mlp_multiclass(num_classes=23)` + class_weight='balanced'
- [x] 4.2 实现 `compute_known_class_accuracy(model, X_test, y_test, known_classes, unseen_classes)`：剔除 16 unseen 攻击后计算 acc
- [x] 4.3 创建 `tests/test_mlp_multiclass.py`：覆盖 23 类输出 / 类别映射 / 双数字报告（8+ 测试）
- [x] 4.4 跑测试 + 训练 + 报告双数字（全量 acc + 已知类 acc）
- [x] 4.5 验收：报告同时含两个数字，全量 acc 即使低也诚实记录

## 5. CNN 模型搭建（任务 4.4）

- [x] 5.1 创建 `src/models/cnn.py`：实现 `CNN1DClassifier(nn.Module)`（Conv1d(1→16) → MaxPool → Flatten → Linear）
- [x] 5.2 实现 `train_cnn_binary(X_train, y_train, X_val, y_val, **kwargs)`：50 epoch + 早停
- [x] 5.3 创建 `tests/test_cnn.py`：覆盖前向传播 / Conv1D 输入输出维度 / tabular 输入 reshape（8+ 测试）
- [x] 5.4 跑测试 + 训练 + 输出二分类指标
- [x] 5.5 验收：binary acc 报告 + 诚实记录"tabular CNN 优势不明确"

## 6. LSTM 模型搭建（任务 4.5）

- [x] 6.1 创建 `src/models/lstm.py`：实现 `LSTMClassifier(nn.Module)`（LSTM(input_size=20, hidden=32) → Linear）
- [x] 6.2 实现 `train_lstm_binary(...)`：50 epoch + 早停 + 输入 reshape (N, 1, 20)
- [x] 6.3 创建 `tests/test_lstm.py`：覆盖 LSTM 序列输入 / hidden state 形状 / reshape（8+ 测试）
- [x] 6.4 跑测试 + 训练 + 输出二分类指标
- [x] 6.5 验收：binary acc 报告 + 诚实记录"NSL-KDD 样本独立无时序，LSTM 优势有限"

## 7. SMOTE 样本不均衡处理（任务 4.6）

- [x] 7.1 创建 `src/data/smote.py`：实现 `apply_smote_to_minority_classes(X_train, y_train, target_classes=['U2R', 'R2L'])`
- [x] 7.2 实现 `compare_with_without_smote(train_fn, ...)`：对比 SMOTE 前后的 per-class recall
- [x] 7.3 实现 SMOTE 数据持久化（`save_smote_data` / `load_smote_data` 用 M2 pickle 工具）
- [x] 7.4 创建 `tests/test_smote.py`：覆盖 SMOTE 样本数变化 / target_classes 过滤 / imblearn 缺失报错（6+ 测试）
- [x] 7.5 跑测试 + 跑对比实验 + 报告 U2R/R2L recall 提升
- [x] 7.6 验收：U2R recall 从 < 0.05 提升到 ≥ 0.20

## 8. 模型持久化（任务 4.7）

- [x] 8.1 扩展 `src/models/persistence.py`：新增 `save_torch_model(model, path)` + `load_torch_model(model_class, path, **kwargs)`
- [x] 8.2 实现 `save_best_nn_models(output_dir)`：保存 6-9 个 .pt 文件
- [x] 8.3 创建 `tests/test_nn_persistence.py`：覆盖 save/load roundtrip / 一致性验证 / 文件不存在 / 架构不匹配（5+ 测试）
- [x] 8.4 跑测试 + 跑持久化（保存 6-9 个模型到 outputs/models/）
- [x] 8.5 验收：load 后预测与 save 前一致（数值精度范围内）

## 9. 训练编排脚本（任务 4.8 配套）

- [x] 9.1 创建 `scripts/train_m4.py`：端到端编排（加载数据 → MLP 训练 → 调优 → 多分类 → CNN → LSTM → SMOTE → 持久化）
- [x] 9.2 智能公式：`N_THREADS = min(4, max(1, os.cpu_count() // 8))`（沿用模式-004）
- [x] 9.3 三层防御：`torch.set_num_threads(N_THREADS)` + 检查 imblearn 依赖 + 进度日志
- [x] 9.4 端到端跑通：`python scripts/train_m4.py`，产出所有 .pt 模型 + 训练日志

## 10. 撰写训练报告（任务 4.8）

- [x] 10.1 创建 `docs/model_report_mlp_dl.md`，含 12 章节：概述 / 数据 / MLP 基线 / MLP 调优 / MLP 多分类 / CNN / LSTM / SMOTE / M3 对比 / 局限诚实记录 / 结论 / 持久化清单
- [x] 10.2 多分类章节含双数字（全量 acc + 已知类 acc）
- [x] 10.3 SMOTE 章节含 per-class recall 对比表
- [x] 10.4 局限章节诚实记录 tabular CNN/LSTM 局限 + SMOTE 副作用 + 多分类失败
- [x] 10.5 与 M3 对比章节含 DT/RF/MLP/CNN/LSTM 指标对比表
- [x] 10.6 验收：报告完整 + 不裁剪要求

## 11. 集成验证（Phase 3 末步）

- [x] 11.1 跑全量测试：`pytest tests/ -v`（M1+M2+M3+M4 全部测试通过）
- [x] 11.2 跑全量训练：`python scripts/train_m4.py`（无报错，产出 6-9 个 .pt）
- [x] 11.3 跑全量 M3：`python scripts/train_m3.py`（确认 M4 未破坏 M3 产出）
- [x] 11.4 验证 openspec 同步：`openspec validate --changes m4-neural-network`

## 依赖关系

```
1 (环境) → 2 (MLP基线) → 3 (MLP调优)
                      → 4 (MLP多分类) ──┐
1 → 5 (CNN) ──────────────────────────┤
1 → 6 (LSTM) ─────────────────────────┤
1 → 7 (SMOTE) → 4 (依赖多分类训练) ───┤
                                       ↓
1 → 8 (持久化) ←─── 2/3/4/5/6/7 ←─────┘
                                       ↓
                            9 (编排脚本) ← 2-8
                                       ↓
                            10 (报告) ← 9
                                       ↓
                            11 (集成验证) ← 10
```

## 验证检查清单

- [x] 11 个测试文件全部 PASS（含 M3 已有测试不退化）
- [x] scripts/train_m4.py 端到端可运行
- [x] 6-9 个 .pt 模型文件产出 + 加载一致
- [x] docs/model_report_mlp_dl.md 完整（≥ 10 章节）
- [x] imbalanced-learn 依赖已安装
- [x] 三层线程防护在测试中验证（torch.set_num_threads(1)）
- [x] 多分类报告含双数字
- [x] SMOTE 副作用诚实记录
- [x] 与 M3 对比表完整
