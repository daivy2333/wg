# Optimization Spec

> Version: 1.4.0 | Last Updated: 2026-06-14（M6 论文完成，项目 M1-M6 全部完工）

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
| 7 | PyTorch + WSL 三层防御 | pytest/PyTorch 触发 WSL 断开（exit 137 OOM） | conftest.py 加 torch 线程限制 + CUDA 禁 + gc 清理 | 高 | ✅ 已完成（2026-06-14，O-PYTORCH-WSL-01） |
| 8 | SMOTE 在 NSL-KDD 多类不适用 | SMOTE 后 full_acc -72% | 改用 focal loss / class_weight 调优 | 高 | ✅ 已验证（2026-06-14，O-NN-01） |
| 9 | MLP 多分类 unseen 攻击 | 全量 acc=0.101 但仍有 16 unseen 无法识别 | One-vs-Rest / Open-set Recognition | 中 | ⚠️ 部分缓解（2026-06-14，O-NN-02）|
| 10 | tabular CNN/LSTM 价值有限 | CNN/LSTM 略低于 MLP | 仅作论文扩展性验证 | 低 | ✅ 已验证（2026-06-14，O-NN-03）|
| 11 | MLP 调优边际收益小 | 6 组合 grid 仅 +0.11% f1 提升 | tabular MLP 较简单，可接受 baseline 直接用 | 低 | ✅ 已验证（2026-06-14，O-NN-04）|
| 12 | M4 报告指标基于验证集非测试集 | 论文数据不可直接使用 M4 报告值 | M5 统一测试集重评估已完成，论文引用 M5 指标 | 高 | ✅ 已完成（2026-06-14，O-M5-01）|

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

### O-PYTORCH-WSL-01: PyTorch + WSL 三层防御（M4 新增）

- **问题**: PyTorch 默认占满所有 CPU 核 + CUDA 初始化在 WSL2 不稳定 → pytest 触发 WSL 断开
- **根因**:
  1. `torch.set_num_threads` 默认 = cpu_count（WSL 32 核 = 32 线程）
  2. CUDA 初始化在 WSL2 占用 300-500MB 显存 + 触发 NVIDIA 驱动调用，对小数据完全无必要反而 OOM
- **解决方案**（三层防御）:
  1. `tests/conftest.py` 强制 `torch.set_num_threads(1)` + `torch.set_num_interop_threads(1)` + `CUDA_VISIBLE_DEVICES=""`（测试禁用 GPU）
  2. `scripts/train_m4.py` 启动时 `N_THREADS = min(4, cpu_count // 8)` 智能公式（生产放开）
  3. 模型 `predict` 方法内部 `device = next(self.parameters()).device` + `.to(device)`（避免 device mismatch）
- **预防措施**:
  - 任何 PyTorch 项目测试 conftest 必须含 torch 三层防御
  - 模型 predict/proba 必须处理 device（CUDA vs CPU）
  - 生产脚本用 `python -u`（unbuffered）便于实时监控
- **优先级**: 高
- **状态**: ✅ 已完成

### O-NN-01: SMOTE 在 NSL-KDD 多类不适用（M4 验证 + 否定）

- **问题**: U2R (52 条) / R2L (995 条) 等小样本类 recall 极低（M3 多分类 f1_macro=0.015）
- **原方案**: SMOTE 仅对小样本类过采样（任务 4.6）
- **实际结果**（M4 验证）:
  - Baseline MLP multiclass: full_acc=0.101, f1_macro=0.015
  - SMOTE 后 MLP multiclass: **full_acc=0.028, f1_macro=0.015**（**-72% acc**）
- **根因**:
  1. SMOTE 后模型在测试时过度预测多数类（class 0 recall 从 < 5% 跳到 37%）
  2. 少数类 recall 仍为 0（合成样本未泛化到测试分布）
  3. NSL-KDD 多类不平衡极严重（U2R 仅 52 条），SMOTE 合成样本边界与真实测试分布差距大
- **建议方案**:
  - **不推荐**用 SMOTE 处理 NSL-KDD 多类问题
  - 改用 focal loss / class_weight 调优 / 半监督 / Open-set Recognition
- **优先级**: 高
- **状态**: ✅ 已验证（副作用已诚实记录到 `docs/model_report_mlp_dl.md` §8.5）

### O-NN-02: MLP 多分类 unseen 攻击识别（M4 部分缓解）

- **问题**: M3 报告 16 种训练未见攻击（NSL-KDD 设计特性），多分类 acc=0.046
- **原方案**: 神经网络非线性建模 + 双数字报告
- **实际结果**（M4 验证）:
  - Baseline MLP multiclass: full_acc=0.101, known_acc=0.101（+117% vs M3）
  - 但 unseen 攻击的局限**未根本解决**
- **根因**: 任何监督学习方法都无法识别训练中完全未见的类别
- **建议方案**:
  - One-vs-Rest 拆解为 23 个二分类子问题
  - Open-set Recognition / 异常检测
  - 半监督学习利用 unlabeled 数据
- **优先级**: 中
- **状态**: ⚠️ 部分缓解（双数字报告已实施）

### O-NN-03: tabular CNN/LSTM 价值有限（M4 验证）

- **问题**: CNN 1D / LSTM 应用于 tabular 数据是否有意义
- **实际结果**（M4 验证）:
  - CNN binary f1=0.962 < MLP f1=0.988（-2.6%）
  - LSTM binary f1=0.984 < MLP f1=0.988（-0.4%）
- **根因**:
  - 1D 卷积需要序列/空间结构，tabular 缺乏
  - LSTM 需要时序依赖，NSL-KDD 样本独立
- **建议方案**:
  - tabular 数据首选 MLP（含可解释优势）
  - CNN/LSTM 仅作为论文"扩展性验证"对比维度
  - 报告时简短提及，不夸大
- **优先级**: 低
- **状态**: ✅ 已验证（已诚实记录到 `docs/model_report_mlp_dl.md` §6.3 / §7.3）

### O-NN-04: MLP 调优边际收益小（M4 验证）

- **问题**: 神经网络调优对 tabular 数据有多大提升
- **实际结果**（M4 验证，6 组合网格）:
  - Baseline MLP (128,64)+relu: f1=0.9883, auc=0.9992
  - 调优后 MLP (256,128,64)+relu: f1=0.9894, auc=0.9992（**+0.11%**）
  - 32 组合完整网格（理论）: 估计提升 < 0.3%
- **根因**:
  - 20 维 tabular 数据，MLP 已足够拟合
  - 隐藏层深度对简单数据影响有限
  - 调优空间主要是层数 / dropout，激活函数和 lr 已收敛
- **建议方案**:
  - tabular MLP 任务可接受 baseline 参数（128, 64）+ relu + dropout=0.3 + Adam(lr=1e-3)
  - 完整 32 组合网格搜索耗时 ~30 分钟但收益 < 0.3%
  - 优先时间投入到特征工程 / 多分类策略（unseen 攻击）而非调优
- **优先级**: 低
- **状态**: ✅ 已验证（已诚实记录到 `docs/model_report_mlp_dl.md` §4.4）

### O-M5-01: M4 报告指标基于验证集非测试集（M5 诚实纠正）

- **问题**: M4 报告（`model_report_mlp_dl.md`）中 MLP f1=0.989 / auc=0.999 等指标基于训练集内部的验证划分（`train_val_split` 的 20% holdout），非独立测试集。验证集与训练集 IID，指标高估真实泛化能力。
- **实际结果**（M5 统一测试集重评估）:
  - MLP Tuned: f1 从 0.989 → 0.720（-27%），auc 从 0.999 → 0.907（-9%）
  - CNN: f1 从 0.962 → 0.745（-22%）
  - LSTM: f1 从 0.984 → 0.716（-27%）
  - DT/RF 指标与 M3 报告一致（M3 用测试集评估，从未混淆）
- **根因**: M4 训练流程用 `train_val_split(X_train, y_train)` 划分验证集，报告 `val_metrics`；而测试集分布不同（含 unseen 攻击 + 不同流量模式），验证集指标不反映真实泛化
- **建议方案**:
  - 论文中所有指标统一引用 M5 的测试集重评估值（`outputs/metrics_m5.json`）
  - M4 报告中的验证集指标标注为"训练过程参考值，非最终评估"
  - 未来训练脚本同时输出 test_metrics（不仅 val_metrics）
- **优先级**: 高（直接影响论文数据可信度）
- **状态**: ✅ 已完成（M5 统一测试集重评估产出 `metrics_m5.json` + `comparison_report.md`）

<!-- O-M7-01 -->
### O-M7-01: 实验图表 PNG 转矢量图（投稿要求）

- **问题**: CJC 模板要求"图应为矢量图"，当前 15 张实验图表均为 PNG 栅格图（matplotlib 默认 `savefig(dpi=150)`），不符合期刊投稿规范
- **当前影响**: 论文可编译但图表在放大/印刷时模糊，可能被期刊退稿
- **建议方案**:
  1. 修改所有图表生成脚本（`notebooks/01_data_exploration.py`, `scripts/evaluate_m5.py`），将 `savefig` 参数改为 `format='pdf'` 或 `format='eps'`
  2. 在 `matplotlibrc` 或脚本中设置 `rcParams['savefig.format'] = 'pdf'`
  3. 重新运行 M2+M5 图表生成，产出矢量版本到 `outputs/figures/`
  4. 引用时使用 `.pdf` 扩展名（`xelatex` 原生支持 PDF 插图）
- **优先级**: 中（编译不阻塞，但投稿前必须完成）
- **状态**: 📝 待处理

<!-- O-M7-02 -->
### O-M7-02: cjc.cls 模板补丁需定期同步上游

- **问题**: 为适配课程作业格式，对 `cjc.cls` 做了两处注释修改（去除 CLC/DOI 显示行、去除收稿/修改日期行）。这些修改在模板升级时会被覆盖。
- **当前影响**: 无——当前版本工作正常。但若从 `latex-/` 上游拉取新版模板，需重新应用补丁。
- **建议方案**:
  1. 将补丁记录为 `paper/cjc.patch`（`diff -u latex-/cjc.cls paper/cjc.cls`）
  2. 或在 cjc-main.tex 导言区用 `\patchcmd` 动态打补丁（避免修改 .cls）
- **优先级**: 低
- **状态**: 📝 待处理

<!-- O-M7-03 -->
### O-M7-03: 图宽 `\columnwidth` vs `\textwidth` 在 cjc 模板下的差异

- **问题**: CJC 模板中 `\includegraphics[width=\textwidth]` 导致图片宽度溢出 252pt（约 2x 版心宽），改用 `\columnwidth` 后正常。根因是 CJC 的 `\textwidth`（约 425pt）在 figure 浮体内可能与图片的 DPI 元数据冲突导致 XeTeX 计算异常。
- **当前方案**: 全章节图宽统一使用 `\columnwidth`（已验证编译通过）
- **建议方案**: 若将来升级模板或更换引擎，验证 `\textwidth` vs `\columnwidth` 行为
- **优先级**: 低
- **状态**: ✅ 已修复

<!-- O-M7-04 -->
### O-M7-04: 多分类任务可考虑降维为 5 类以提升可用性

- **问题**: 当前多分类为 40 类细粒度攻击标签，准确率仅 4.6-10.1%。若改为 5 大类（Normal/DoS/Probe/R2L/U2R）分类，可大幅提升准确率且对安全运维更具实际指导意义
- **当前状态**: `label_id_to_category.json` 已包含 40→5 的映射，但未用于训练或评估
- **建议方案**:
  1. 新增 `make_labels(task="multiclass_5cat")` 函数，基于 category 映射生成 5 类标签
  2. 重新训练 5 类多分类模型，预期准确率可达 70-90%
  3. 论文中作为"实际部署建议"提及，不替换现有 40 类分析
- **优先级**: 低（现有 40 类分析已充分说明挑战，5 类改进属于后续工作）
- **状态**: 📝 待处理
