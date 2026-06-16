# SNAPSHOT.md — 项目状态快照

> Last Updated: 2026-06-16
> **M1-M8 全部完成** ✅ 🎉 | **Paper 内部一致性 + 代码 metric 缺口修复**
> - M8：多分类从 40 类攻击标签改为 5 大类（Normal/DoS/Probe/R2L/U2R），重训模型 + 论文更新
> - 修复：`evaluate_m5.py` 配置错误导致 `mlp_multiclass` 数据缺失；10 处 paper 内部不一致（Top-5/章节数/图序号/unseen 数量/衰减数字/per-class F1 精度等）

---

## 项目结构树

```
wg/
├── readme                    # 项目说明文档
├── dataset/
│   └── archive.zip           # NSL-KDD 数据集（待解压）
├── openspec/
│   ├── config.yaml           # OpenSpec 项目配置
│   ├── changes/
│   │   └── archive/          # 已归档变更
│   │       ├── 2026-06-14-m1-infrastructure/
│   │       ├── 2026-06-14-m2-data-exploration-and-preprocessing/
│   │       ├── 2026-06-14-m3-traditional-ml/
│   │       └── 2026-06-14-m4-neural-network/    # ✅ 已归档
│   └── specs/
│       ├── architecture/     # 架构决策记录
│       ├── learned/          # 学习记忆
│       ├── references/       # 外部参考
│       ├── optimization/     # 优化记录
│       ├── cnn-training/     # M4 新增
│       ├── lstm-training/    # M4 新增
│       ├── mlp-training/     # M4 新增
│       ├── nn-model-reporting/  # M4 新增
│       └── smote-sampling/   # M4 新增
├── src/                      # M1+M2+M3+M4 代码
│   ├── data/
│   │   ├── loader.py            # M1 KDDTrain+/KDDTest+ 加载
│   │   ├── preprocessor.py      # M1 编码 + 标准化 + 整合管线
│   │   ├── outlier.py           # M2 IQR + clip + log1p
│   │   ├── feature_selector.py  # M2 方差阈值 + RF Top-K
│   │   ├── persistence.py       # M2 pickle 序列化
│   │   └── smote.py             # M4 SMOTE 样本不均衡
│   ├── models/
│   │   ├── decision_tree.py     # M3 DT 基线 + 网格搜索
│   │   ├── random_forest.py     # M3 RF 基线 + 网格搜索 + 重要度
│   │   ├── persistence.py       # M3 joblib + M4 torch.save 双后端
│   │   ├── mlp.py               # M4 MLP（基线/调优/多分类）
│   │   ├── cnn.py               # M4 1D CNN
│   │   └── lstm.py              # M4 LSTM
│   └── evaluation/           # M5 评估模块
│   │   ├── __init__.py
│   │   ├── metrics.py            # M5 3 个评估函数
│   │   └── plot.py               # M5 5 个图表函数
│   ├── paper/                 # M6 LaTeX 论文 + M7 CJC 模板排版
│   │   ├── main.tex              # M6 ctexart + XeLaTeX 主文件（保留）
│   │   ├── cjc-main.tex          # M7 CJC 模板主文件（新）
│   │   ├── cjc.cls               # M7 CJC 模板类文件（复制自 latex-/）
│   │   ├── cjc.bst               # M7 CJC 参考文献格式（复制自 latex-/）
│   │   ├── latexmkrc             # M7 latexmk 配置
│   │   ├── Makefile              # M7 双模式编译（main + cjc）
│   │   ├── refs.bib              # M6 参考文献（unsrt，保留）
│   │   ├── cjc-refs.bib          # M7 CJC 格式参考文献（19 条：9 原有 + 10 新增）
│   │   ├── chapters/             # 6 个章节 .tex（图1-8/10/13-16 双栏，图9/11/12 单栏）
│   │   ├── background.tex        # M7 英文背景介绍 (~400 词)
│   │   ├── appendix.tex          # M7 附录（超参数/SMOTE/特征排名，引用已硬编码）
│   │   └── slides_outline.md     # 15 页答辩 PPT 提纲
│   └── utils/                # 通用工具
├── tests/                    # M1+M2+M3+M4 pytest 测试
│   ├── conftest.py              # M3 WSL 兼容 + M4 torch 三防
│   ├── test_loader.py        # 12 tests
│   ├── test_preprocessor.py  # 17 tests
│   ├── test_outlier.py       # 11 tests
│   ├── test_feature_selector.py  # 9 tests
│   ├── test_persistence.py   # 8 tests
│   ├── test_decision_tree.py    # M3 11 tests
│   ├── test_random_forest.py    # M3 11 tests
│   ├── test_model_persistence.py # M3 7 tests
│   ├── test_mlp.py              # M4 16 tests
│   ├── test_cnn.py              # M4 6 tests
│   ├── test_lstm.py             # M4 7 tests
│   ├── test_smote.py            # M4 5 tests
│   ├── test_nn_persistence.py   # M4 8 tests
│   ├── test_evaluation_metrics.py  # M5 8 tests
│   ├── test_evaluation_plot.py     # M5 6 tests
├── scripts/
│   ├── train_m3.py              # M3 编排脚本（n_jobs=4 智能调节）
│   ├── train_m4.py              # M4 编排脚本（torch 智能公式）
│   ├── run_smote_only.py        # M4 SMOTE 单独运行
│   └── evaluate_m5.py           # M5 对比分析编排（10 模型 + 7 图 + 报告）
├── notebooks/
│   └── 01_data_exploration.py   # M2 EDA 脚本（14 场景）
├── outputs/                  # M2 + M3 + M4 + M5 输出
│   ├── processed/            # 8 个 .pkl 文件
│   ├── figures/              # 15 个分析图表 PNG（M2: 8, M5: 7）
│   ├── models/               # 4 个 .joblib (M3) + 6 个 .pt (M4)
│   ├── metrics_m4.json        # M4 训练指标
│   ├── metrics_m5.json        # M5 统一测试集指标（10 模型）
│   ├── label_id_to_name.json  # M5 40 个攻击名映射
│   └── label_id_to_category.json  # M5 40 个分类映射
├── docs/
│   ├── eda_report.md
│   ├── model_report_dt_rf.md
│   ├── model_report_mlp_dl.md
│   └── comparison_report.md      # M5 9 章节对比分析报告
├── README.md                 # M6 项目完整复现指南
├── openspec/changes/         # OpenSpec 变更管理
│   ├── archive/
│   │   ├── 2026-06-14-m1-infrastructure/
│   │   ├── 2026-06-14-m2-data-exploration-and-preprocessing/
│   │   ├── 2026-06-14-m3-traditional-ml/
│   │   └── 2026-06-14-m4-neural-network/  # 待归档
│   └── m4-neural-network/    # M4 活跃变更（待归档）
├── requirements.txt          # M1 新增 - 依赖锁定（注：版本与实际 pip 不一致，见技术栈警告）
└── CLAUDE.md                 # 项目入口索引
```

---

## 技术栈

| 类别 | 工具/库 | 版本 | 状态 |
|------|---------|------|------|
| 编程语言 | Python | 3.13.5 | ✅ 已安装（conda） |
| 数据处理 | Pandas | 2.3.3 | ✅ 已安装 |
| 数值计算 | NumPy | 2.2.6 | ✅ 已安装 |
| 可视化 | Matplotlib | 3.10.9 | ✅ 已安装 |
| 机器学习 | scikit-learn | ≥1.8 | ✅ 已安装（M3 RF 训练验证） |
| 统计可视化 | Seaborn | ≥0.13 | ✅ 已安装 |
| 深度学习 | PyTorch | 2.9.0+cu128 | ✅ 已安装（M4 训练验证）|
| 不均衡处理 | imbalanced-learn | 0.14.2 | ✅ 已安装（M4 SMOTE 用）|
| 环境管理 | conda (miniconda) | — | ✅ 可用（which 找不到但 conda activate 正常） |

> ⚠️ 依赖版本说明：`requirements.txt` 锁定的版本（如 pandas 3.0.2、torch 2.9.0）与实际 pip 安装版本不一致，原因待查（可能是 M1 锁定后未同步，或环境升级未更新 requirements.txt）。M1-M3 功能在当前实际环境下全部通过，不影响当前进度。

---

## Git 状态

- **仓库状态**: ✅ 已初始化
- **当前分支**: main
- **项目状态**: M1-M8 全部完成 ✅，本轮修复 paper 内部一致性 + mlp_multiclass 数据缺口

---

## 关键文件

| 文件 | 用途 | 备注 |
|------|------|------|
| `readme` | 项目说明 | 技术栈、数据集、分工、实验流程 |
| `dataset/archive.zip` | NSL-KDD 数据集 | 14MB，待解压 |
| `openspec/config.yaml` | OpenSpec 配置 | schema: spec-driven |
| `openspec/specs/architecture/spec.md` | 架构决策 | 含 5 条 ADR（A01-A05）|
| `openspec/specs/learned/spec.md` | 学习记忆 | 含 13 条踩坑档案 + 6 个技巧模式（M8 新增踩坑-011/012/013）|
| `openspec/specs/references/spec.md` | 外部参考 | 依赖表 + 学术资源 |
| `openspec/specs/optimization/spec.md` | 优化记录 | 12 条优化点（M5 新增 O-M5-01）|
| `scripts/evaluate_m5.py` | M5 编排脚本 | 10 模型加载+评估+图表+报告 |
| `src/evaluation/metrics.py` | M5 评估函数 | 3 个模型无关的指标计算函数 |
| `src/evaluation/plot.py` | M5 可视化函数 | 5 个 matplotlib 图表生成函数 |

---

## 数据集

| 属性 | 值 |
|------|-----|
| 名称 | NSL-KDD |
| 训练集 | 125,973 条记录 |
| 测试集 | 22,544 条记录 |
| 特征数 | 41 |
| 任务类型 | 二分类（normal/anomaly）+ 多分类（23 种攻击类型） |
| 攻击大类 | DoS、Probe、R2L、U2R |
| 文件位置 | dataset/KDDTrain+.txt + KDDTest+.txt（M1 任务 1.1 已解压）|

---

## 项目分工

| 角色 | 负责人 | 职责范围 | 完成情况 |
|------|--------|----------|----------|
| A同学（数据方向） | — | 数据加载与探索、缺失值处理、特征标准化、数据集划分 | M1 + M2 全部完成 |
| B同学（模型方向） | — | 决策树与随机森林模型搭建、模型调优、参数网格搜索 | M3 全部完成（9/9）|
| C同学（对比分析） | — | MLP 神经网络搭建、模型评估、可视化图表、性能对比分析 | M4 + M5 全部完成（8/8 + 7/7）|

---

## 关键产物对比（M5 统一测试集重评估）

| 维度 | 指标 | DT Tuned | RF Tuned | MLP Tuned | CNN | LSTM |
|------|------|----------|----------|-----------|-----|------|
| 二分类 | accuracy | 0.7755 | 0.7234 | 0.7445 | 0.7635 | 0.7427 |
| 二分类 | f1 | 0.7621 | 0.6882 | 0.7202 | 0.7452 | 0.7161 |
| 二分类 | auc | 0.7821 | 0.9150 | 0.9074 | 0.8820 | 0.8832 |
| 多分类 | full_acc | 0.0456 | 0.0464 | **0.1009** | — | — |
| 多分类 | known_acc | 0.0533 | 0.0542 | **0.1179** | — | — |
| 多分类 | f1_macro | 0.0155 | 0.0166 | 0.0146 | — | — |

> ⚠️ **M5 关键发现**：M4 报告中的 MLP f1=0.989 等指标基于训练集验证划分（validation split），M5 在统一测试集上重评估得到更保守的指标（MLP f1≈0.72）。深度学习模型在测试集上与 DT/RF 表现相近，M4 报告的高指标存在验证集偏差。

**M4 关键结论**（M4 报告原始值，基于验证集）：
- 🎯 **二分类 MLP 显著优于 RF**（f1 +30%, auc +8%, recall +47% — 几乎所有异常被捕获）— ⚠️ 此为验证集指标
- 📈 **多分类 MLP 略优于 RF**（+117% acc，但受 unseen 攻击拖累）
- ⚠️ **SMOTE 在本项目中失败**（full_acc -72%，副作用已诚实记录到 O-NN-01）

---

## 关键经验（M1-M5 累计）

| 类别 | 关键经验 | 来源 |
|------|----------|------|
| 环境 | WSL 下 conda 通过 shell 函数初始化，`which conda` 找不到 | 踩坑-001/002 |
| 稳定性 | **ML 项目必须四层防御**（OMP/MKL/torch 线程 + CUDA 禁 + 测试间 gc）| 踩坑-003/004 |
| 稳定性 | **生产脚本必须 `python -u`**，否则 stdout 缓冲误判"卡死" | 踩坑-005 |
| 评估 | **验证集 ≠ 测试集**，M4 报告指标基于验证集，M5 统一测试集重评估发现 DL 模型与 DT 相近 | 踩坑-007 |
| 评估 | **CV + Test 双数字必须同时报告**，避免 GridSearchCV 误导 | 模式-003 |
| 数据 | **SMOTE 不是万能**，多类极不均衡（U2R 52 条）副作用 > 收益 | 踩坑-006 |
| 框架 | **PyTorch 三层线程防御** + 模型 `device-aware` 是 WSL 必做 | 模式-005 |
| 框架 | **PyTorch 模型 kwargs 必须精确匹配 MODEL_CONFIGS**，否则加载失败 | 踩坑-007 |
| 持久化 | **PyTorch save state_dict 而非整个 model**，更轻 + 跨版本兼容 | ADR-005 + 实践 |
| 报告 | **诚实记录局限 + 副作用**是合规研究的关键，0.10 → 0.03 也要写 | M3 §8 + M4 §8.5 |
