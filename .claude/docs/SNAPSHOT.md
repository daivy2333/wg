# SNAPSHOT.md — 项目状态快照

> Last Updated: 2026-06-14
> M1 + M2 + M3 阶段已完成（M1 任务 1.4-1.8 ✅ / M2 任务 2.1-2.10 ✅ / M3 任务 3.1-3.9 ✅）

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
│   │       └── 2026-06-14-m3-traditional-ml/
│   └── specs/
│       ├── architecture/     # 架构决策记录
│       ├── learned/          # 学习记忆
│       ├── references/       # 外部参考
│       └── optimization/     # 优化记录
├── src/                      # M1 新增 - 代码骨架
│   ├── data/
│   │   ├── loader.py            # M1 KDDTrain+/KDDTest+ 加载
│   │   ├── preprocessor.py      # M1 编码 + 标准化 + 整合管线
│   │   ├── outlier.py           # M2 IQR + clip + log1p
│   │   ├── feature_selector.py  # M2 方差阈值 + RF Top-K
│   │   └── persistence.py       # M2 pickle 序列化
│   ├── models/                  # M3 新增
│   │   ├── decision_tree.py     # M3 DT 基线 + 网格搜索
│   │   ├── random_forest.py     # M3 RF 基线 + 网格搜索 + 重要度
│   │   └── persistence.py       # M3 joblib 序列化
│   ├── evaluation/           # M5 占位
│   └── utils/                # 通用工具
├── tests/                    # M1+M2+M3 pytest 测试
│   ├── conftest.py              # M3 WSL 兼容（OMP=1）
│   ├── test_loader.py        # 12 tests
│   ├── test_preprocessor.py  # 17 tests
│   ├── test_outlier.py       # 11 tests
│   ├── test_feature_selector.py  # 9 tests
│   ├── test_persistence.py   # 8 tests
│   ├── test_decision_tree.py    # M3 11 tests
│   ├── test_random_forest.py    # M3 11 tests
│   └── test_model_persistence.py # M3 7 tests
├── scripts/
│   └── train_m3.py              # M3 编排脚本（n_jobs=4 智能调节）
├── notebooks/
│   └── 01_data_exploration.py   # M2 EDA 脚本（14 场景）
├── outputs/                  # M2 + M3 输出
│   ├── processed/            # 8 个 .pkl 文件
│   ├── figures/              # 8 个分析图表 PNG
│   └── models/               # M3 4 个 .joblib
├── docs/
│   ├── eda_report.md             # M2 EDA 报告
│   └── model_report_dt_rf.md     # M3 DT/RF 训练报告
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
| 深度学习 | PyTorch | 2.11.0 | ✅ 已安装（含 torchvision 0.26.0） |
| 环境管理 | conda (miniconda) | — | ✅ 可用（which 找不到但 conda activate 正常） |

> ⚠️ 依赖版本说明：`requirements.txt` 锁定的版本（如 pandas 3.0.2、torch 2.9.0）与实际 pip 安装版本不一致，原因待查（可能是 M1 锁定后未同步，或环境升级未更新 requirements.txt）。M1-M3 功能在当前实际环境下全部通过，不影响当前进度。

---

## Git 状态

- **仓库状态**: ✅ 已初始化
- **当前分支**: main
- **最近提交**: f83ca27 — feat(M3): 传统机器学习模型训练完成
- **提交历史**: a8d17e0 (init) → 2e28300 (M1) → c45b3cc (M2) → f83ca27 (M3)
- **未提交更改**: 无
- **OpenSpec 活跃变更**: 无（M1/M2/M3 均已归档至 `openspec/changes/archive/`）

---

## 关键文件

| 文件 | 用途 | 备注 |
|------|------|------|
| `readme` | 项目说明 | 技术栈、数据集、分工、实验流程 |
| `dataset/archive.zip` | NSL-KDD 数据集 | 14MB，待解压 |
| `openspec/config.yaml` | OpenSpec 配置 | schema: spec-driven |
| `openspec/specs/architecture/spec.md` | 架构决策 | 含 5 条 ADR（A01-A05）|
| `openspec/specs/learned/spec.md` | 学习记忆 | 含 3 条踩坑档案（conda/WSL/sklearn OMP）|
| `openspec/specs/references/spec.md` | 外部参考 | 依赖表 + 学术资源 |
| `openspec/specs/optimization/spec.md` | 优化记录 | 6 条优化点（含 M3 新增 O-DT-01 / O-RF-01 / O-WSL-01）|

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

| 角色 | 负责人 | 职责范围 |
|------|--------|----------|
| A同学（数据方向） | — | 数据加载与探索、缺失值处理、特征标准化、数据集划分 |
| B同学（模型方向） | — | 决策树与随机森林模型搭建、模型调优、参数网格搜索 |
| C同学（对比分析） | — | MLP 神经网络搭建、模型评估、可视化图表、性能对比分析 |
