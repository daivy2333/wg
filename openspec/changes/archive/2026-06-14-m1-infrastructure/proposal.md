## Why

当前 wg 项目处于初始化阶段，仅有数据集与文档体系，**缺少可直接运行的代码骨架**。M1 是整个项目的第一里程碑（M2-M6 的前置依赖），若不先建立 `src/` 目录结构、数据加载/预处理模块与依赖锁定文件，后续 M2（EDA）将无法起步，M3/M4 训练模型也无干净数据可用。因此现在必须先打通 M1。

## What Changes

新增 5 项产出物（任务 1.4-1.8），**不修改任何现有文件**：

- **新增** `src/` 标准目录骨架（`src/data/`、`src/models/`、`src/evaluation/`、`src/utils/`，均含 `__init__.py`）+ `notebooks/` + `outputs/`
- **新增** `requirements.txt`，锁定全部依赖版本（pandas/numpy/scikit-learn/seaborn/matplotlib/torch/pytest）
- **新增** `src/data/loader.py`，提供 `load_train()` / `load_test()` 函数，加载 KDDTrain+/KDDTest+，定义 41 列特征名常量
- **新增** `src/data/preprocessor.py`，提供缺失值处理、分类编码（OneHot + Label 混合）、StandardScaler 标准化、保留原始 KDDTrain+/KDDTest+ 划分
- **新增** `notebooks/01_data_exploration.ipynb`，基于 loader + preprocessor 完成基础 EDA

**回滚方案**：删除新增的 5 类文件/目录即可（git 可直接 revert，因为只新增未修改）。

## Capabilities

### New Capabilities

- `data-loading`：NSL-KDD 数据集加载能力（loader 模块）
- `data-preprocessing`：数据预处理能力（preprocessor 模块）
- `project-skeleton`：项目代码骨架与依赖锁定（目录结构 + requirements.txt）

### Modified Capabilities

（无现有 spec 变更，本次 M1 阶段无对已有规格的需求级别修改）

## Impact

- **影响范围**：仅 `src/`、`notebooks/`、`requirements.txt` 三个新增区域
- **下游依赖**：M2（M2 完全依赖 loader/preprocessor）、M3/M4（依赖预处理后的干净数据）
- **不影响的范围**：不修改 `dataset/`、`openspec/specs/`、`CLAUDE.md`、`.claude/docs/`
- **风险**：loader 的 41 特征名硬编码需与 NSL-KDD 官方一致；preprocessor 的编码策略需与 M2 后续 EDA 协同