## Why

M1（基础设施）+ M2（数据探索 + Top-20 特征 + 8 个 pickle）已完成。M3 是任务书明确的"传统 ML 模型训练"阶段，由 B 同学主导。若不训练决策树/随机森林基线 + 调优，M5 对比分析无基准模型，论文无传统 ML 章节。

## What Changes

新增 4 项产出物，**不修改 M1/M2 既有模块**：

- **新增** `src/models/decision_tree.py`：DecisionTreeClassifier 基线 + GridSearchCV 调优（二分类 + 多分类）
- **新增** `src/models/random_forest.py`：RandomForestClassifier 基线 + GridSearchCV 调优（二分类 + 多分类）
- **新增** `src/models/persistence.py`：joblib save/load 模型 + 评估指标整合
- **新增** `docs/model_report_dt_rf.md`：DT/RF 模型报告（含参数/指标/4 大攻击类别 F1 对比）
- **新增** `scripts/train_m3.py`：M3 训练编排脚本（一键跑完 DT + RF 基线 + 调优 + 持久化）

**回滚方案**：删除新增 5 类文件即可；M1/M2 零修改，回滚零风险。

## Capabilities

### New Capabilities

- `decision-tree-training`：决策树训练（二分类 + 多分类基线 + GridSearchCV 调优）
- `random-forest-training`：随机森林训练（二分类 + 多分类基线 + GridSearchCV 调优 + 特征重要度）
- `model-persistence`：joblib 模型持久化与加载
- `ml-model-reporting`：DT/RF 模型报告生成

### Modified Capabilities

（无。M1/M2 已归档的 spec 均不变）

## Impact

- **影响范围**：仅新增 `src/models/` 目录下 3 个文件 + `scripts/train_m3.py` + `docs/model_report_dt_rf.md`
- **下游依赖**：
  - M4（MLP/DL）：复用 M2 的 pickle + 借鉴 M3 的网格搜索流程
  - M5（对比分析）：直接 pickle.load M3 最佳模型，与 M4 对比
  - M6（论文）：直接引用 `docs/model_report_dt_rf.md`
- **不影响的范围**：M1 loader/preprocessor + M2 outlier/feature_selector/persistence 零修改
- **风险**：
  - GridSearchCV 在 125k 行 × 20 列上训练耗时 → 缓解：n_jobs=-1 并行 + 精简网格（DT 30 组合、RF 24 组合）
  - class_weight='balanced' 可能降低整体 accuracy → 缓解：同时报告 accuracy + f1_macro，对比两种权重策略
  - U2R 类仅 52 条训练样本 → 缓解：训练集 SMOTE 由 M4 处理，M3 保持 baseline 公平