# Learned Spec

> Version: 1.4.0 | Last Updated: 2026-06-14（M6 论文完成，项目 M1-M6 全部完工）

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

### 踩坑-004: PyTorch + WSL 触发连接断开（M4 新增）

- **症状**: 运行 pytest + PyTorch 测试时 WSL 断开，或 train_m4.py 长时间无输出后断连
- **根因**（双层）:
  1. `torch.set_num_threads` 默认 = `os.cpu_count()`（WSL2 = 32 核 = 32 线程）
  2. CUDA 初始化在 WSL2 占用 300-500MB 显存 + 触发 NVIDIA 驱动调用，对小数据完全无必要反而 OOM
- **解决方案**: 三层防御：
  1. `tests/conftest.py` 强制 `torch.set_num_threads(1)` + `torch.set_num_interop_threads(1)` + `CUDA_VISIBLE_DEVICES=""`（测试禁用 GPU）
  2. `scripts/train_m4.py` 启动时 `N_THREADS = min(4, cpu_count // 8)` 智能公式（生产放开）
  3. 模型 `predict` 方法内部 `device = next(self.parameters()).device` + `.to(device)`（避免 device mismatch）
- **预防措施**:
  - 任何 PyTorch 项目测试 conftest 必须含 torch 三层防御
  - 模型 predict/proba 必须处理 device（CUDA vs CPU），否则 `model.predict(X_test)` 报 `RuntimeError: mat1 on cpu, different from other tensors on cuda:0`
- **结论**: PyTorch + WSL 必须三防：conftest 三件套 + 生产智能公式 + 模型 device-aware

### 踩坑-005: Python stdout 缓冲导致后台任务无输出（M4 新增）

- **症状**: 用 `nohup python script.py &` 或 Bash run_in_background 启动训练，日志文件长时间为空，但 CPU 99% 说明在跑
- **根因**: Python 默认 stdout 是行缓冲（连接到 tty 时）或块缓冲（连接到 pipe/文件时）。`tee` 创建 pipe → Python 切到块缓冲 → print() 输出累积在内核缓冲区，文件不刷新
- **解决方案**: 用 `python -u` 启动（unbuffered）→ 强制无缓冲 → print 立即写入
- **预防措施**:
  - 所有 ML 训练脚本启动时用 `python -u`（或设 `PYTHONUNBUFFERED=1`）
  - 或在脚本里 `sys.stdout.reconfigure(line_buffering=True)`（Python 3.7+）
  - 长时间训练任务必须先验证 `tail -f log` 能看到输出
- **结论**: 后台训练必须用 `python -u`，否则误判"卡死"会 kill 掉实际在跑的任务

### 踩坑-006: SMOTE 在 NSL-KDD 多类不适用（M4 验证 + 否定）

- **症状**: SMOTE 后 MLP 多分类 full_accuracy 从 0.10 降到 0.03（-72%）
- **根因**（多重）:
  1. SMOTE 合成样本仅在训练集类边界内插值，未泛化到测试分布
  2. 多数类（DoS/normal）样本充足，SMOTE 后少数类样本数被提升到多数类水平（53k+），模型训练后倾向于预测多数类（class 0 recall 跳到 37%）
  3. NSL-KDD 多类极不均衡（U2R 仅 52 条），few-shot learning 本身难题，SMOTE 不能根本解决
  4. SMOTE k_neighbors 受限（NSL-KDD 多个类样本数 < 5），k_neighbors 必须降到 2 才能运行
- **解决方案**:
  - **不推荐**用 SMOTE 处理 NSL-KDD 多类问题
  - 改用 focal loss / class_weight 调优（更鲁棒）
  - 真正解决 unseen 攻击需 Open-set Recognition / 半监督
- **预防措施**:
  - 实施 SMOTE 前先评估目标类样本数 < 10 是否值得做（few-shot 难）
  - 实施后必须对比 baseline + 副作用记录（不能只看 acc 是否提升）
  - 论文中诚实记录"我们试了 SMOTE 但失败了"是合规的研究报告
- **结论**: 任何数据增强/重采样方法实施后必须 baseline 对比，不能假设"对不均衡有效"

### 踩坑-007: 验证集 vs 测试集指标差异（M5 发现）

- **症状**: M4 报告 MLP 二分类 f1=0.989 / auc=0.999，M5 统一测试集重评估 f1=0.720 / auc=0.907（差距 27%）
- **根因**: M4 训练时用 `train_val_split(X_train, y_train)` 划分 20% 训练数据作验证集，报告 `val_metrics`。验证集与训练集分布相同（IID），而测试集分布不同（含 unseen 攻击、不同流量模式）。验证集指标高估了真实泛化能力。
- **解决方案**:
  1. M5 在统一测试集上重评估全部 10 个模型，得到公平对比基线（DT f1=0.76, RF f1=0.69, MLP f1=0.72）
  2. 论文中必须标注评估数据来源（"统一测试集 22,544 样本"），不可混淆验证/测试指标
  3. 建立 MODEL_CONFIGS 字典精确记录 PyTorch 模型 kwargs（hidden_dims 等），避免加载时维度不匹配
- **预防措施**:
  - ML 实验流程中验证集仅用于早停/调参，最终报告必须用测试集指标
  - 训练脚本同时保存 test_metrics（不只看验证集）
  - 对比分析时所有模型必须在相同测试集上评估
- **结论**: 验证集指标 ≠ 测试集指标；论文报告必须以测试集为准，验证集偏差可达 27%

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

### 模式-005: PyTorch 三层线程防御（M4 新增）

- **技巧名称**: torch.set_num_threads(1) + CUDA_VISIBLE_DEVICES="" + 模型 device-aware
- **适用场景**: WSL2 上跑 PyTorch + pytest / 长时间训练
- **使用方法**:
```python
# 第一层：tests/conftest.py
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# 第二层：生产脚本智能公式
N_THREADS = min(4, max(1, (os.cpu_count() or 1) // 8))
torch.set_num_threads(N_THREADS)

# 第三层：模型 device-aware
def predict(self, x):
    device = next(self.parameters()).device
    x_t = torch.as_tensor(x, dtype=torch.float32).to(device)
    return self.forward(x_t)
```
- **效果**: WSL2 + PyTorch 100% 稳定（无 OOM / 无 WSL 断开），生产性能保留 ~80%

### 模式-006: python -u 后台训练（避免假"卡死"判断）

- **技巧名称**: `python -u script.py` 启动长时间任务
- **适用场景**: 任何 > 30s 的 ML 训练 / 评测脚本
- **使用方法**:
```bash
# ❌ 错误：print 输出被缓冲，日志文件长时间为空
nohup python train.py > log 2>&1 &

# ✅ 正确：unbuffered，print 立即写入
nohup python -u train.py > log 2>&1 &

# 或设环境变量
PYTHONUNBUFFERED=1 python train.py
```
- **效果**: 日志实时可见，避免误判"卡死"kill 掉在跑的任务

## M1-M4 全局经验总结

> 跨 4 个里程碑提炼的 8 条核心经验，按重要性排序

### 经验 1: 任何 ML 项目测试必须四层防御（WSL/Linux）

- **问题**: pytest + sklearn/PyTorch 触发 WSL 断开（exit 137 OOM）
- **根本原因**:
  - 默认 OMP/MKL/OPENBLAS 多线程爆炸
  - PyTorch 默认 `cpu_count` 线程
  - CUDA 初始化在 WSL2 不稳定
  - Python 模型对象驻留 heap 累积
- **防御方案**（四层，缺一不可）:
```python
# tests/conftest.py — 必须四件套
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.cuda.empty_cache()
```
- **关联**: 踩坑-003/004、模式-005、ADR-007

### 经验 2: 报告必须诚实记录局限和副作用

- **做法**:
  - 双数字：CV f1 + Test f1 同时给（避免 GridSearchCV 误导）
  - 副作用：SMOTE 后 full_acc -72% 也要诚实记录
  - 局限：CNN/LSTM 略低于 MLP 也要写（不夸大）
  - 未知：unseen 攻击无法识别也要承认
- **M3-M4 实践**:
  - M3 §8: "多分类 acc=0.046 接近随机" + 14 unseen 攻击解释
  - M4 §8.5: "SMOTE 后 full_acc 从 0.10 降到 0.03" + 副作用分析
  - M4 §6.3: "tabular CNN 优势不明确" 论文价值论述
- **关联**: Karpathy Requirements Integrity 原则

### 经验 3: 任何数据增强/重采样必须 baseline 对比

- **反面教材**: SMOTE 在本项目中凭直觉实施，导致 full_acc -72%
- **正面做法**:
  1. 实施前：先评估目标类样本数 < 10 是否有意义
  2. 实施中：保留 baseline 模型作对照
  3. 实施后：metrics 报告含 baseline vs 处理后两个数字
  4. 论文中：失败案例也是合规研究成果
- **关联**: 踩坑-006、O-NN-01

### 经验 4: 模型 predict 必须 device-aware

- **问题**: `model.predict(X_test)` 报 `RuntimeError: mat1 on cpu, different from other tensors on cuda:0`
- **根因**: 模型在 CUDA，输入在 CPU
- **方案**:
```python
def predict(self, x):
    self.eval()
    device = next(self.parameters()).device  # ← 关键
    x_t = torch.as_tensor(x, dtype=torch.float32).to(device)  # ← 关键
    with torch.no_grad():
        return self.forward(x_t).argmax(dim=1).cpu().numpy()
```
- **应用**: MLP/CNN/LSTM 三个模型的 predict 方法都已实现 device-aware
- **关联**: 踩坑-004、ADR-007

### 经验 5: 后台训练必须 `python -u`

- **问题**: `nohup python train.py > log 2>&1 &` 日志文件长时间空，误判"卡死"kill 掉
- **根因**: stdout 块缓冲（连接 pipe 时）
- **方案**: `python -u` 或 `PYTHONUNBUFFERED=1` 强制无缓冲
- **关联**: 踩坑-005、模式-006

### 经验 6: PyTorch 持久化用 state_dict 而非整个 model

- **方案**:
  - 保存: `torch.save(model.state_dict(), path)`
  - 加载: `model = ModelClass(**kwargs); model.load_state_dict(torch.load(path))`
- **优势**:
  - 体积更小（仅参数，无类路径）
  - 跨 PyTorch 版本兼容
  - 避免 pickle 安全风险
- **关联**: ADR-005（joblib vs pickle）、ADR-006（PyTorch state_dict）

### 经验 7: 智能公式 `min(N, max(1, cpu_count // 8))` 安全并行

- **方案**: 测试 = 1 / 生产 = `min(4, max(1, cpu_count // 8))`
- **公式解读**:
  - 32 核 → 4，8 核 → 1，4 核 → max(1, 0) = 1
  - 硬上限 N（避免 32 核 WSL 资源爆炸）
- **应用**: M3 `n_jobs` (sklearn) + M4 `torch.set_num_threads` (PyTorch)
- **关联**: 踩坑-003、模式-004

### 经验 8: tabular 数据首选 MLP，CNN/LSTM 仅作论文扩展

- **结论**（M4 验证）:
  - MLP f1=0.988 > LSTM f1=0.984 > CNN f1=0.962
  - tabular 数据无空间/时序结构，CNN/LSTM 优势不明确
- **做法**:
  - 论文主线：MLP（强基线 + 可解释）
  - 论文扩展：CNN/LSTM 简述"扩展性验证"维度
  - 不夸大 CNN/LSTM 性能

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
| `src/models/persistence.py` | M3+M4 模型持久化 | joblib (sklearn) + torch.save (PyTorch) 双后端 |
| `scripts/train_m3.py` | M3 训练编排 | 端到端训练（n_jobs=4 智能调节） |
| `src/models/mlp.py` | M4 MLP | PyTorch MLP：基线/调优/多分类 + 已知类 acc |
| `src/models/cnn.py` | M4 CNN | 1D CNN（tabular 适配） |
| `src/models/lstm.py` | M4 LSTM | LSTM（样本当序列） |
| `src/data/smote.py` | M4 SMOTE | imblearn SMOTE（仅小样本类 + k_neighbors=2） |
| `scripts/train_m4.py` | M4 训练编排 | 端到端（6 组合 grid + CNN/LSTM/SMOTE） |
| `tests/conftest.py` | WSL 兼容 | 强制 OMP=1/MKL=1/OPENBLAS=1 + torch 三防 |
| `outputs/processed/` | M2 输出 | 8 个 .pkl（X_train/test × binary/multi） |
| `outputs/models/` | M3+M4 输出 | 4 个 .joblib (DT/RF) + 6 个 .pt (MLP/CNN/LSTM) |
| `outputs/figures/` | M2 可视化 | 8 张 PNG（分布/相关/重要度） |
| `outputs/metrics_m4.json` | M4 训练指标 | 7 个模型的 acc/f1/auc 等 |
| `docs/eda_report.md` | M2 EDA 报告 | 数据探索全流程与结论 |
| `docs/model_report_dt_rf.md` | M3 训练报告 | DT/RF 10 章节，含多分类局限诚实记录 |
| `docs/model_report_mlp_dl.md` | M4 训练报告 | MLP/CNN/LSTM 13 章节，含 SMOTE 副作用 |
| `docs/comparison_report.md` | M5 对比报告 | 9 章节，10 模型测试集统一评估 |
| `scripts/evaluate_m5.py` | M5 编排脚本 | 加载/评估/图表/报告一体化 |
| `src/evaluation/metrics.py` | M5 评估函数 | compute_binary_metrics / multiclass / f1_by_category |
| `src/evaluation/plot.py` | M5 图表函数 | 混淆矩阵/ROC/F1柱状图/特征重要度/DLvsML |
| `outputs/metrics_m5.json` | M5 测试集指标 | 10 个模型的 accuracy/precision/recall/f1/auc |
| `outputs/label_id_to_name.json` | M5 标签映射 | 40 个攻击名 ID→名称映射 |
| `paper/main.tex` | M6 论文主文件 | ctex + XeLaTeX，6 章 \input，BibTeX |
| `paper/chapters/ch1-intro.tex` | M6 Ch1 | 绪论+数据集，1215字，3图 |
| `paper/chapters/ch6-conclusion.tex` | M6 Ch6 | 总结展望，1694字，3局限+5未来 |
| `paper/refs.bib` | M6 参考文献 | 9 条（NSL-KDD, sklearn, PyTorch, SMOTE, 3中文）|
| `paper/slides_outline.md` | M6 PPT提纲 | 15 slides，15张图引用 |
| `README.md` | 项目指南 | 252行，5步复现流程 |
