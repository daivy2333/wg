## Context

M1 + M2 已完成 8 个 pickle 数据文件（二分类/多分类 × X/y 各 2 个，共 8 个），Top-20 特征已稳定。M3 阶段直接消费这些 pickle，无需重新执行数据预处理。

**M3 关键约束**：
- 不修改 M1/M2 既有模块（向后兼容 + 已有 57 tests）
- 训练数据：X_train.pkl (125973, 20) / y_train.pkl (125973,)（二分类）
- 多分类：X_train_multi.pkl + y_train_multi.pkl（23 类）
- class_weight='balanced'（用户决策）→ 必须同时报告 weighted 与 macro 指标
- GridSearchCV 网格规模：DT 5×3×2=30 / RF 3×4×2=24（在 125k 行上 < 5 分钟）

## Goals / Non-Goals

**Goals:**
- 打通 M3 五项产出（DT 训练 / RF 训练 / 模型持久化 / 模型报告 / 训练脚本）
- DT/RF 二分类 + 多分类共 4 个最佳模型持久化
- GridSearchCV 调优超参数 + 报告最优组合
- 4 大攻击类别（DoS/Probe/R2L/U2R）的 F1 对比
- M5 对比分析可直接消费 M3 输出

**Non-Goals:**
- 不实现 SMOTE（属于 M4 范围）
- 不实现神经网络模型（属于 M4 范围）
- 不实现集成学习（XGBoost/LightGBM，超出 M3 范围）
- 不修改 M2 持久化的 pickle 文件（M3 只读取）

## Decisions

### 决策 1: class_weight='balanced'

- **策略**：所有 DT/RF 模型使用 `class_weight='balanced'`
- **原因**：U2R 类仅 52 条训练样本，无加权时模型可能完全忽略；balanced 让稀有类在损失函数中权重提高
- **替代方案**：
  - None（无加权）：简单但 U2R 召回率可能为 0
  - 自定义 dict {0:1, 1:10}：需手动调权重
- **影响**：报告同时含 accuracy 和 f1_macro 指标

### 决策 2: 5 指标多维评估

- **指标**：accuracy / precision / recall / f1 / auc（二分类）/ f1_macro（多分类）
- **原因**：
  - accuracy 单一指标会被 normal 类主导
  - precision/recall 揭示误报 vs 漏报权衡
  - f1_macro 平均所有类的 F1，不受类不平衡影响
  - auc 反映 ROC 曲线下面积（二分类适用）
- **影响**：模型报告含完整 5 指标 + 4 大攻击类别 F1 拆解

### 决策 3: DT 网格搜索范围

```
param_grid = {
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"],
}
# 共 5 × 3 × 2 = 30 组合
```

- **max_depth**: 限制树深度防止过拟合，None 不限制
- **min_samples_split**: 控制叶节点最小样本数
- **criterion**: gini（默认）/ entropy
- **影响**：5-fold CV，30 × 5 = 150 次训练，预计 < 2 分钟（n_jobs=-1）

### 决策 4: RF 网格搜索范围

```
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 15, 20, None],
    "min_samples_split": [2, 5],
}
# 共 3 × 4 × 2 = 24 组合
```

- **n_estimators**: 树数量（100-300），更多树 = 更慢但更稳定
- **max_depth**: 比 DT 略深（10-20），RF 抗过拟合能力强
- **影响**：5-fold CV，24 × 5 = 120 次训练，预计 < 5 分钟

### 决策 5: joblib 模型持久化（替代 pickle）

- **格式**：`joblib.dump(model, path)` 替代 `pickle.dump`
- **原因**：joblib 对 sklearn 模型更高效（内部使用 numpy 二进制 pickle + 压缩）
- **替代方案**：
  - pickle：通用但对 numpy array 不友好
  - ONNX：跨框架但 sklearn → ONNX 转换复杂
  - PMML：标准但工具链不成熟
- **影响**：模型文件扩展名 `.joblib`，加载用 `joblib.load(path)`

### 决策 6: 模块边界（M3 新增 models/）

```
src/models/
├── decision_tree.py    # M3 DT 训练
├── random_forest.py    # M3 RF 训练
└── persistence.py      # M3 模型持久化
```

- **原因**：M1/M2 既有 models/ 是空目录，M3 填充
- **影响**：M3 模块独立，不依赖 M1/M2 业务逻辑（仅消费 pickle 数据）

## Risks / Trade-offs

| 风险 | 影响 | 缓解 |
|------|------|------|
| GridSearchCV 训练耗时 | M3 总时长 > 10 分钟 | n_jobs=-1 并行 + 5-fold CV |
| class_weight='balanced' 降低 accuracy | accuracy 可能从 99% → 95% | 同时报告 f1_macro 反映真实性能 |
| U2R 训练样本极少（52 条） | RF 几乎无法学习 U2R 特征 | 报告分类 F1 + 标注 U2R 召回率 |
| DT 网格 None max_depth 可能过拟合 | 测试集 accuracy 下降 | 报告训练 vs 测试 accuracy 差异 |
| joblib 跨 Python 版本兼容 | M4 误用其他 Python 报错 | 在 model_report.md 顶部固定 Python 3.13.5 |

## Migration Plan

- **部署步骤**：M3 完成后，M4 启动前由 C 同学加载 M3 最佳模型 + M2 特征数据，开始 MLP 训练
- **回滚策略**：删除 `src/models/*.py`、`scripts/train_m3.py`、`outputs/models/*.joblib`、`docs/model_report_dt_rf.md`
- **依赖顺序**：M1 → M2 → **M3** → M4 → M5 → M6（M4 强依赖 M3 的 4 个最佳模型 + 网格搜索经验）