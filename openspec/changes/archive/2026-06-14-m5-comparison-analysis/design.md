## Context

M3 和 M4 产出了 10 个训练好的模型（4 个 sklearn joblib + 6 个 PyTorch .pt），评估指标分散在三份独立报告中，且评估数据不统一（M3 用测试集，M4 部分用验证集）。M5 需要在统一的测试集（`X_test.pkl` + `y_test.pkl`）上重新评估全部模型，生成横向对比图表和综合分析报告。

**关键约束**：
- 测试集有 38 个多分类标签，但模型仅在 23 个训练类上训练 → 必须报告 full_accuracy 和 known_class_accuracy 双数字
- LabelEncoder 未持久化 → 需要从原始 CSV 重新推导攻击名→ID 映射
- PyTorch 模型的 constructor kwargs 必须精确匹配（MLP tuned 使用 `hidden_dims=(256,128,64)`）
- WSL 环境需遵守 ADR-007 四层线程防御

## Goals / Non-Goals

**Goals:**
- 在统一的测试集上重新评估全部 10 个模型，输出机器可读的 `metrics_m5.json`
- 生成 7 张论文级别的对比图表（混淆矩阵、柱状图、ROC 曲线、特征重要度等）
- 生成完整的对比分析报告 `docs/comparison_report.md`
- 提供可复现的编排脚本 `scripts/evaluate_m5.py`

**Non-Goals:**
- 不修改任何 M3/M4 的训练代码或模型文件
- 不添加新模型或超参数调优
- 不计算统计显著性检验（报告结论提及"未来工作"）
- 不测量推理延迟（报告提及"未来工作"）
- 不生成交互式图表（Plotly 等）—— 仅静态 matplotlib PNG

## Decisions

### D1: src/evaluation/ 模块化 + scripts/ 编排（双文件架构）

**选择**: `src/evaluation/metrics.py`（纯函数，模型无关）+ `src/evaluation/plot.py`（纯函数，matplotlib）+ `scripts/evaluate_m5.py`（编排，文件 I/O）

**备选**: 单文件脚本（`train_m3.py` 风格）

**理由**: 评估函数（metrics/plot）是纯计算的、可测试的、可复用的，应与文件 I/O 和模型加载分离。遵循现有架构惯例（`src/models/decision_tree.py` 提供 `evaluate_model()`，`scripts/train_m3.py` 编排调用）。

### D2: 使用现有的 `load_model()` 和 `load_torch_model()` 加载模型

**选择**: 复用 `src/models/persistence.py` 的加载函数，不重新实现

**理由**: 已有完善的错误处理（FileNotFoundError、RuntimeError for dimension mismatch），保持 DRY。

### D3: MODEL_CONFIGS 字典存储 PyTorch kwargs

**选择**: 在编排脚本中维护 `MODEL_CONFIGS` dict，映射模型文件名到精确的 `model_class` + `**kwargs`

```python
MODEL_CONFIGS = {
    "mlp_binary_best.pt":     (MLPClassifier, {"input_dim": 20, "output_dim": 2}),
    "mlp_binary_tuned.pt":    (MLPClassifier, {"input_dim": 20, "output_dim": 2, "hidden_dims": (256,128,64), "dropout": 0.3}),
    "mlp_multiclass_best.pt": (MLPClassifier, {"input_dim": 20, "output_dim": 23}),
    "mlp_multiclass_smote.pt":(MLPClassifier, {"input_dim": 20, "output_dim": 23}),
    "cnn_binary_best.pt":     (CNN1DClassifier, {"input_length": 20, "output_dim": 2}),
    "lstm_binary_best.pt":    (LSTMClassifier, {"input_size": 20, "output_dim": 2}),
}
```

**理由**: 防止默认 kwargs 导致的维度不匹配错误（MLP tuned 架构不同于 baseline）。

### D4: 从原始 CSV 推导 attack-name→ID 映射

**选择**: 加载 `dataset/KDDTrain+.txt`，用 `LabelEncoder` 拟合全部 38 个标签，保存 `outputs/label_id_to_name.json`

**理由**: M2 的 LabelEncoder 未持久化，M5 需要此映射来生成"各攻击大类 F1"柱状图（任务 5.3）。

### D5: 多分类混淆矩阵仅展示二分类 2×2

**选择**: 任务 5.2 的混淆矩阵热力图仅展示二分类（normal vs anomaly）的 2×2 矩阵。多分类 23×23 矩阵过大且稀疏，在报告中以文字描述代替。

**理由**: 23×23 矩阵 529 个格子，大部分为 0，无法有效传递信息。报告中将讨论多分类的局限性（unseen attacks）。

### D6: SMOTE 模型独立处理

**选择**: SMOTE 多分类模型不在主对比图表中出现，在报告独立"失败实验"小节中分析，引用优化记录 O-NN-01。

**理由**: SMOTE 在本项目中副作用显著（full_accuracy -72%），混入主对比会误导读者。

### D7: MLP baseline 作为消融对比行

**选择**: 对比表格中 MLP tuned 为主要值，MLP baseline 作为"消融"行展示调优增益。图表中仅展示最佳模型。

**理由**: 论文惯例——主表展示最佳结果，消融研究在独立小节讨论。

## Risks / Trade-offs

| 风险 | 缓解措施 |
|------|----------|
| PyTorch 模型加载时维度不匹配 | MODEL_CONFIGS 字典精确记录 kwargs；加载后做 smoke test（predict 10 样本）|
| RF 多分类模型 97MB，加载慢（~10s） | 一次性加载所有模型，不重复加载 |
| WSL 资源不足（6 个 PyTorch 模型同时 in-memory） | 设置 `CUDA_VISIBLE_DEVICES=""` + `torch.set_num_threads(1)`；评估完一个模型后 `del model` |
| 攻击名→ID 映射可能不正确 | 从原始 CSV 重新拟合，与 M2 的 ATTACK_CATEGORY dict 交叉验证 |
| 多分类 full_accuracy 与 M4 报告不一致 | M4 的 mlp_multiclass 指标是在 test set 上计算的，应一致。若不一致，以 M5 重新计算的值为准。 |
