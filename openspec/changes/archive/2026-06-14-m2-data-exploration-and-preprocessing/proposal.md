## Why

M1 已完成数据加载与基础预处理（loader + preprocessor + 29 tests），但**缺少深度 EDA、异常值处理、特征工程与持久化产出**。M3（M3 决策树/随机森林）和 M4（M4 MLP/DL）训练模型**强依赖** M2 产出的干净 + 精选特征数据文件；若 M2 不产出 `.pkl`，M3/M4 需各自重复执行预处理，违反 DRY 原则。现在必须打通 M2。

## What Changes

新增 5 项产出物，**不修改 M1 既有代码**（向后兼容）：

- **新增** `src/data/outlier.py`：IQR 异常值检测 + clip 替换函数（不丢样本）
- **新增** `src/data/feature_selector.py`：方差阈值 + 随机森林特征重要度双保险选择器
- **新增** `src/data/persistence.py`：pickle 序列化与加载（X/y 二分类 + 多分类双套）
- **新增** `docs/eda_report.md`：完整 EDA 报告 Markdown 文档
- **扩展** `notebooks/01_data_exploration.ipynb`：增加特征分布、相关、异常、特征选择 4 大分析块（17 场景全覆盖）

**回滚方案**：删除新增 5 类文件即可；既有 M1 模块（loader.py/preprocessor.py）零修改，回滚零风险。

## Capabilities

### New Capabilities

- `outlier-handling`：IQR 异常值检测与 clip 替换（M2 任务 2.4）
- `feature-selection`：方差阈值 + 随机森林 Top-K 双保险选择（M2 任务 2.7）
- `data-persistence`：pickle 数据持久化（X/y 二分类+多分类双套，M2 任务 2.9）
- `eda-reporting`：EDA 报告生成（docs/eda_report.md，M2 任务 2.10）

### Modified Capabilities

（无。M1 的 `data-loading` / `data-preprocessing` / `project-skeleton` 三个 spec 不修改，仅在本变更新增 capability。M1 的 S1 简化（缺失值不自动填充）保持不变，因 NSL-KDD 官方数据无缺失）

## Impact

- **影响范围**：仅新增 `src/data/outlier.py`、`src/data/feature_selector.py`、`src/data/persistence.py`、`docs/eda_report.md` 4 个文件 + 扩展 notebook
- **下游依赖**：
  - M3（决策树/随机森林）：直接 `pickle.load("outputs/processed/X_train.pkl")`
  - M4（MLP/DL）：同上
  - M5（对比分析）：复用 M2 持久化的特征 + 标签
- **不影响的范围**：M1 loader/preprocessor 完全不动；OpenSpec 主 specs 不修改
- **风险**：
  - 长尾特征 log1p 变换可能改变原始物理含义（M3 模型可解释性略降）→ 缓解：保留原始版本对照
  - RF Top-20 训练耗时（10 万+ 行 × 53 列）→ 缓解：用训练集 fit，禁用 verbose，单次运行 < 30s
  - pickle 不跨 Python 版本兼容 → 缓解：固定 Python 3.13.5 环境