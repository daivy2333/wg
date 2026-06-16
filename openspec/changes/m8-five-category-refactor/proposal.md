## Why

M7 终稿审核发现多分类任务采用 40 个细粒度攻击标签，准确率仅 4.6-10.1%。经排查确认非代码缺陷，而是 NSL-KDD 数据集固有特性（测试集 15 类未见攻击 + 极端类不平衡）。已验证 5 类后处理映射无效（+2pp），结论：需在训练阶段即使用 5 大类标签重训模型。5 大类（Normal/DoS/Probe/R2L/U2R）更符合安全运维实际需求，且为 NSL-KDD 文献常用范式，预期准确率可达 70-90%。

## What Changes

- **BREAKING**: `make_labels()` 新增 `task="multiclass_5cat"`，利用 `label_id_to_category.json` 生成 5 类标签，替代原 40 类编码
- **BREAKING**: 所有模型 `output_dim` 从 23/40 改为 5，SMOTE 多分类实验移除
- 重新生成 `y_train_multi.pkl` / `y_test_multi.pkl`（5 类标签）
- 重训 DT/RF/MLP/CNN/LSTM 多分类模型（RTX 4060 GPU 加速）
- 重跑 M5 评估对比，产出 5×5 混淆矩阵和 per-class F1 图表
- 更新测试文件中的 `output_dim` 断言
- 论文 ch1-6 + appendix 全面更新：40 类→5 类，删除 unseen 攻击讨论，更新所有指标

## Capabilities

### New Capabilities

- `multiclass-5cat-label`: 5 大类标签生成逻辑（`make_labels(task="multiclass_5cat")`）
- `multiclass-5cat-training`: 5 类模型训练管线（DT/RF/MLP/CNN/LSTM，output_dim=5）
- `multiclass-5cat-paper`: 论文 5 类多分类章节更新

### Modified Capabilities

- `mlp-training`: `output_dim` 参数从 23→5，移除 SMOTE 多分类
- `decision-tree-training`: 多分类标签从 40 类→5 类
- `random-forest-training`: 多分类标签从 40 类→5 类

## Impact

- **代码**: `src/data/preprocessor.py`（新增 5cat）、`src/models/mlp.py`（output_dim）、`scripts/train_m3.py`、`scripts/train_m4.py`、`scripts/evaluate_m5.py`
- **数据**: `outputs/processed/y_train_multi.pkl`、`y_test_multi.pkl` 重新生成
- **模型**: `outputs/models/{dt,rf,mlp}_{multiclass}*` 全部重训
- **图表**: `outputs/figures/` 新增 5×5 混淆矩阵和 per-class F1 图
- **论文**: `paper/chapters/ch1-*.tex`~`ch6-*.tex`、`paper/appendix.tex`
- **测试**: `tests/test_decision_tree.py`、`tests/test_random_forest.py`、`tests/test_mlp.py`
- **不涉及**: 二分类实验、CJC 排版、参考文献
