# Optimization Spec

> Version: 1.1.0 | Last Updated: 2026-06-14（M3 训练补全，6 条记录已含 M3 优化点 #4 #5 #6）

## Purpose

记录项目中发现的优化点和改进方向，持续提升代码质量和性能。

## Requirements

### Requirement: 优化点记录

发现的性能瓶颈、代码异味、技术债务 SHALL 记录，包含当前影响和建议方案。

#### Scenario: 发现优化机会

- **WHEN** 开发者发现代码中存在性能问题、重复代码、过度复杂设计等
- **THEN** 必须记录到 optimization/spec.md，包含：问题描述、当前影响、建议方案、优先级

#### Scenario: 评估优化价值

- **WHEN** 开发者需要决定是否进行某项优化
- **THEN** 可以参考 optimization/spec.md 中的记录，评估影响范围和收益

### Requirement: 优化完成追踪

已完成的优化 SHALL 记录完成状态，保留历史记录。

#### Scenario: 完成优化

- **WHEN** 开发者完成了某项优化工作
- **THEN** 必须更新 optimization/spec.md，标记为已完成，记录完成日期和实际效果

### Requirement: 优化优先级管理

优化点 SHALL 有优先级排序，合理安排优化顺序。

#### Scenario: 规划优化计划

- **WHEN** 开发者制定优化计划时
- **THEN** 可以参考 optimization/spec.md 中的优先级标注，优先处理高优先级优化点

## 优化记录

| # | 问题描述 | 当前影响 | 建议方案 | 优先级 | 状态 |
|---|----------|----------|----------|--------|------|
| 1 | seaborn/scikit-learn 未安装 | 无法运行 ML 模型训练和高级可视化 | `pip install seaborn scikit-learn` | 高 | ✅ 已完成（2026-06-14，用户在 conda 环境中安装） |
| 2 | 数据集未解压 | 无法开始数据探索 | `unzip dataset/archive.zip -d dataset/` | 高 | ✅ 已完成（2026-06-14，KDDTrain+ 125973条 + KDDTest+ 22544条） |
| 3 | 项目未初始化 git | 无版本控制 | `git init` + 初始提交 | 中 | ✅ 已完成（2026-06-14，main 分支，commit a8d17e0） |
| 4 | DT/RF 多分类泛化能力不足（acc≈0.046） | 多分类几乎无意义，无法识别 14 种训练未见攻击 | M4 范围：one-vs-rest / SMOTE / 半监督 | 高 | 🆕 新增（M3 完成时发现） |
| 5 | 二分类 CV/Test 差距大（CV f1=0.99 vs test f1=0.69） | GridSearchCV 最优参数误导 | 报告 CV + 实际 test 双数字；hold-out 验证 | 中 | 🆕 新增（M3 完成时发现） |
| 6 | sklearn WSL 并行爆炸（踩坑-003） | pytest/sklearn 触发 WSL 崩溃 | conftest.py 强制 OMP=1 + n_jobs=1 默认 | 高 | ✅ 已完成（2026-06-14，tests/conftest.py + random_forest.py n_jobs 参数化） |

## 优化点详情

### O-DT-01 / O-RF-01: 多分类失败 + CV/Test 差距

- **问题**: M3 训练报告显示 DT/RF 多分类 accuracy ≈ 0.046（接近随机猜测 1/23）
- **当前影响**:
  - 二分类 f1=0.69-0.76 仍可用
  - 多分类几乎无意义，无法识别 14 种训练未见攻击
- **根因**:
  - 测试集含 14 种训练集中完全未出现的攻击（apache2/httptunnel/worm 等）
  - class_weight='balanced' 对 unseen class 无效，模型只能归到训练已有高频类
- **建议方案**（M4 范围）:
  - one-vs-rest 拆解为 23 个二分类子问题
  - SMOTE 过采样稀有类
  - 半监督学习处理 unseen 攻击
- **优先级**: 高（M4 启动必读）
- **报告**: `docs/model_report_dt_rf.md` §8 局限

### O-WSL-01: sklearn WSL 并行爆炸（踩坑-003）

- **问题**: pytest + sklearn 触发 WSL 连接断开
- **根因**: 未设 OMP_NUM_THREADS 时，sklearn 的 GridSearchCV（n_jobs=-1）默认启动 OMP 并行，进程数 = 物理核数（WSL 32 逻辑核），加上 pytest 收集 11+ 测试组合爆炸
- **解决方案**（双层防护）:
  1. `tests/conftest.py` 强制 OMP_NUM_THREADS=1 / MKL_NUM_THREADS=1 / OPENBLAS_NUM_THREADS=1
  2. sklearn 模型（RF / GridSearchCV）的 `n_jobs` 默认值改为 1；生产脚本 `train_m3.py` 显式传 `min(4, cpu_count // 8)` = 4
- **预防措施**:
  - ML 项目测试前**必须**确认 conftest.py 存在
  - sklearn 模型函数的 `n_jobs` 默认值应为 1，让生产脚本显式 opt-in 多进程
- **优先级**: 高
- **状态**: ✅ 已完成
