# Learned Spec

> Version: 1.1.0 | Last Updated: 2026-06-14（M3 训练补全）

## Purpose

记录项目开发过程中学到的知识，避免重复探索，加速问题解决。

## Requirements

### Requirement: API 路径记录

项目中使用的关键 API 路径 SHALL 记录，包含用途和使用示例。

#### Scenario: 发现新 API

- **WHEN** 开发者发现或使用了新的 API 端点
- **THEN** 必须记录到 learned/spec.md，包含：API 路径、用途、请求/响应格式

### Requirement: 踩坑经验记录

遇到的技术陷阱和解决方案 SHALL 记录，防止重复踩坑。

#### Scenario: 解决棘手问题

- **WHEN** 开发者花费大量时间解决了一个技术问题
- **THEN** 必须记录踩坑档案，包含：症状、根因、解决方案、预防措施

### Requirement: 技巧模式记录

有效的开发技巧和模式 SHALL 记录，促进知识共享。

#### Scenario: 发现高效做法

- **WHEN** 开发者发现了一种高效的开发技巧或模式
- **THEN** 必须记录到技巧模式区，包含：技巧名称、适用场景、使用方法

### Requirement: 文件速查表

关键文件和目录的位置 SHALL 记录，加速代码导航。

#### Scenario: 定位关键文件

- **WHEN** 开发者频繁访问某些文件或目录
- **THEN** 必须记录到文件速查表，包含：文件路径、用途、关键内容

## 踩坑档案

### 踩坑-001: which conda 无输出但 conda 可用

- **症状**: 运行 `which conda` 无输出，但 `conda activate` 可正常工作
- **根因**: conda 通过 shell 函数初始化（`eval "$(conda shell.bash hook)"`），而非作为独立可执行文件安装到 PATH，因此 `which` 找不到
- **解决方案**: 直接使用 `conda activate` 激活环境，不依赖 `which` 检测
- **预防措施**: 检测 conda 可用性时，用 `conda --version` 或 `type conda` 替代 `which conda`
- **结论**: 项目使用 conda（miniconda）管理 Python 环境，后续所有 Python 命令应在 conda 环境中执行

### 踩坑-002: bash 子 shell 中 conda 不可用

- **症状**: Claude Code 的 Bash 工具（子 shell）中 `conda info --envs` 报 `conda: command not found`，但用户终端中 conda 正常工作
- **根因**: conda 通过 `~/.bashrc` 中的 shell 函数初始化，Bash 工具的子 shell 不会 source `~/.bashrc`，因此 conda 函数和路径均不可用
- **解决方案**: 用户在终端中执行 conda 相关操作；Claude Code 中直接使用 `/home/daivy/miniconda3/bin/python` 运行脚本
- **预防措施**: 需要安装包或管理 conda 环境时，提示用户在终端执行
- **结论**: Claude Code 中执行 Python 代码应使用 `/home/daivy/miniconda3/bin/python`（Python 3.13.5），而非 `/usr/bin/python3`（Python 3.10.12）

### 踩坑-003: pytest + sklearn 触发 WSL 连接断开

- **症状**: 运行 `pytest tests/test_decision_tree.py` 后 WSL 连接断开，shell 无响应
- **根因**: 未设 `OMP_NUM_THREADS` 时，sklearn 的 GridSearchCV（n_jobs=-1）默认启动 OMP 并行。在 WSL2 多核 + 内存受限环境下，pytest 收集 11 个测试，每个触发 OMP 进程，组合爆炸（n 进程 × m 线程）导致 OOM 或 fork 失败
- **解决方案**: 两层防护：
  1. 创建 `tests/conftest.py` 强制设置 OMP_NUM_THREADS=1 / MKL_NUM_THREADS=1 / OPENBLAS_NUM_THREADS=1
  2. sklearn 模型（RF / GridSearchCV）的 `n_jobs` 默认值改为 `1`（非 -1），避免进程级并行；生产脚本（`scripts/train_m3.py`）可显式传 `n_jobs=-1` 加速
- **预防措施**:
  - ML 项目测试前**必须**确认 conftest.py 存在
  - sklearn 模型函数的 `n_jobs` 默认值应为 1，让生产脚本显式 opt-in 多进程
- **结论**: 测试环境限制 OMP=1 + n_jobs=1 保证稳定（< 30s 性能损失），生产代码放开

## 技巧模式

### 模式-001: sklearn 并行度的三层配置

- **技巧名称**: 三层并行度防御（OMP / n_jobs / conftest）
- **适用场景**: 在 WSL/Linux 多核机器上跑 sklearn + pytest 时
- **使用方法**:

```python
# 第一层：测试环境（tests/conftest.py）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# 第二层：sklearn 函数默认值（避免隐式 n_jobs=-1）
def train_rf(...):  # 强制 n_jobs=1
    return RandomForestClassifier(..., n_jobs=1)

# 第三层：生产脚本（scripts/train_m3.py）智能公式
import os
N_JOBS = min(4, max(1, os.cpu_count() // 8))  # 32核→4, 8核→1
model = RandomForestClassifier(..., n_jobs=N_JOBS)
```

- **效果**: 测试环境 100% 稳定（无 OOM/fork 失败），生产环境最大化利用 CPU（4 进程并行约 3-4x 加速）

### 模式-002: sklearn 模型持久化选型（pickle vs joblib）

- **技巧名称**: 按数据规模选序列化格式
- **适用场景**: sklearn 模型保存到磁盘
- **使用方法**:
  - 模型 < 10MB 或 DataFrame/ndarray → `pickle`（通用）
  - sklearn 模型（尤其 RF/XGBoost 等大数组） → `joblib`（5-10x 更快）
- **依据**: scikit-learn 官方推荐 joblib；joblib 内部对 numpy array 分块+压缩
- **影响**: 本项目 `src/data/persistence.py`（pickle for 数据）+ `src/models/persistence.py`（joblib for 模型）分离
- **决策记录**: 详见 ADR-005

### 模式-003: CV/Test 差距的诚实记录

- **技巧名称**: 网格搜索时同时记录 CV 评分 + 实际 test 评分
- **适用场景**: GridSearchCV 选出的 best_params 报告
- **问题**: CV 在训练集上评估（数据已见过），test 集才是真实泛化能力
- **使用模板**:

```python
# 训练
best_model, best_params, best_cv_score = grid_search_dt(...)
# CV: 0.99（训练集上）

# 测试
test_metrics = evaluate_model(best_model, X_test, y_test)
# Test: f1=0.78（真实泛化）

# 报告：两个数字都展示
print(f"CV f1={best_cv_score:.4f}, Test f1={test_metrics['f1']:.4f}")
```

- **效果**: 避免 GridSearchCV 的 CV 评分误导；用户能看真实泛化能力

### 模式-004: n_jobs 智能公式（4 核安全并行）

- **技巧名称**: `n_jobs = min(4, max(1, cpu_count // 8))`
- **适用场景**: WSL/Linux 多核机器上跑 sklearn 生产训练
- **公式解读**:
  - `cpu_count // 8`：32 核 → 4，8 核 → 1，4 核 → 0 → max(1, 0) = 1
  - `min(4, ...)`：硬上限 4（避免 32 核 WSL 资源爆炸）
- **验证**: WSL 32 核机器运行 `train_m3.py`（30 组合 × 5 折 RF 网格搜索）13.6s 完成，无 WSL 断开
- **关联**: 踩坑-003

## 文件速查表

| 文件路径 | 用途 | 关键内容 |
|----------|------|----------|
| `readme` | 项目说明文档 | 技术栈、数据集说明、分工建议、实验流程 |
| `dataset/KDDTrain+.txt` | NSL-KDD 训练集 | 125,973 条，41 特征 + 标签 + 难度 |
| `dataset/KDDTest+.txt` | NSL-KDD 测试集 | 22,544 条，41 特征 + 标签 + 难度 |
| `openspec/config.yaml` | OpenSpec 项目配置 | 技术栈、规则、上下文 |
| `/home/daivy/miniconda3/bin/python` | conda 环境 Python | Python 3.13.5，所有依赖已安装 |
| `src/data/loader.py` | M1 数据加载 | KDDTrain+/KDDTest+ → DataFrame，41 列特征名 |
| `src/data/preprocessor.py` | M1 数据预处理 | 编码 + 标准化 + 整合管线 |
| `src/data/outlier.py` | M2 异常值处理 | IQR + clip + log1p |
| `src/data/feature_selector.py` | M2 特征选择 | 方差阈值 + RF Top-K |
| `src/data/persistence.py` | M2 数据持久化 | pickle 序列化（数据/特征） |
| `src/models/decision_tree.py` | M3 决策树 | DT 基线 + 网格搜索（n_jobs=1） |
| `src/models/random_forest.py` | M3 随机森林 | RF 基线 + 网格搜索 + 重要度（n_jobs=1） |
| `src/models/persistence.py` | M3 模型持久化 | joblib 序列化（替代 pickle，numpy 数组效率更高） |
| `scripts/train_m3.py` | M3 训练编排 | 端到端训练（n_jobs=4 智能调节） |
| `tests/conftest.py` | WSL 兼容 | 强制 OMP_NUM_THREADS=1 / MKL=1 / OPENBLAS=1 |
| `outputs/processed/` | M2 输出 | 8 个 .pkl（X_train/test × binary/multi） |
| `outputs/models/` | M3 输出 | 4 个 .joblib（DT/RF × binary/multi） |
| `outputs/figures/` | M2 可视化 | 8 张 PNG（分布/相关/重要度） |
| `docs/eda_report.md` | M2 EDA 报告 | 数据探索全流程与结论 |
| `docs/model_report_dt_rf.md` | M3 训练报告 | DT/RF 10 章节，含多分类局限诚实记录 |
