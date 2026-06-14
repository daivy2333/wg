# SNAPSHOT.md — 项目状态快照

> Last Updated: 2026-06-14
> M1 阶段已完成（任务 1.4-1.8 全 ✅）

---

## 项目结构树

```
wg/
├── readme                    # 项目说明文档
├── dataset/
│   └── archive.zip           # NSL-KDD 数据集（待解压）
├── openspec/
│   ├── config.yaml           # OpenSpec 项目配置
│   ├── changes/              # 变更提案目录
│   │   └── m1-infrastructure/  # M1 变更（待归档）
│   └── specs/
│       ├── architecture/     # 架构决策记录
│       ├── learned/          # 学习记忆
│       ├── references/       # 外部参考
│       └── optimization/     # 优化记录
├── src/                      # M1 新增 - 代码骨架
│   ├── data/
│   │   ├── loader.py         # KDDTrain+/KDDTest+ 加载
│   │   └── preprocessor.py   # 编码 + 标准化 + 整合管线
│   ├── models/               # M3/M4 占位
│   ├── evaluation/           # M5 占位
│   └── utils/                # 通用工具
├── tests/                    # M1 新增 - pytest 测试
│   ├── test_loader.py        # 12 tests
│   └── test_preprocessor.py  # 17 tests
├── notebooks/                # M1 新增 - EDA notebook
├── outputs/                  # 输出目录
├── requirements.txt          # M1 新增 - 依赖锁定
└── CLAUDE.md                 # 项目入口索引
```

---

## 技术栈

| 类别 | 工具/库 | 版本 | 状态 |
|------|---------|------|------|
| 编程语言 | Python | 3.10.12 | ✅ 已安装 |
| 数据处理 | Pandas | 2.3.3 | ✅ 已安装 |
| 数值计算 | NumPy | 2.2.6 | ✅ 已安装 |
| 可视化 | Matplotlib | 3.10.9 | ✅ 已安装 |
| 深度学习 | PyTorch | 2.11.0 | ✅ 已安装 |
| 机器学习 | scikit-learn | — | ❌ 待安装 |
| 统计可视化 | Seaborn | — | ❌ 待安装 |
| 环境管理 | conda (miniconda) | — | ✅ 可用（which 找不到但 conda activate 正常） |

---

## Git 状态

- **仓库状态**: ✅ 已初始化
- **当前分支**: main
- **最近提交**: a8d17e0 — init: 项目初始化 - OpenSpec + 数据集 + 文档体系
- **未提交更改**: 无

---

## 关键文件

| 文件 | 用途 | 备注 |
|------|------|------|
| `readme` | 项目说明 | 技术栈、数据集、分工、实验流程 |
| `dataset/archive.zip` | NSL-KDD 数据集 | 14MB，待解压 |
| `openspec/config.yaml` | OpenSpec 配置 | schema: spec-driven |
| `openspec/specs/architecture/spec.md` | 架构决策 | 含 3 条 ADR |
| `openspec/specs/learned/spec.md` | 学习记忆 | 含 conda 踩坑记录 |
| `openspec/specs/references/spec.md` | 外部参考 | 依赖表 + 学术资源 |
| `openspec/specs/optimization/spec.md` | 优化记录 | 3 条待处理优化 |

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
| 文件位置 | dataset/archive.zip（待解压） |

---

## 项目分工

| 角色 | 负责人 | 职责范围 |
|------|--------|----------|
| A同学（数据方向） | — | 数据加载与探索、缺失值处理、特征标准化、数据集划分 |
| B同学（模型方向） | — | 决策树与随机森林模型搭建、模型调优、参数网格搜索 |
| C同学（对比分析） | — | MLP 神经网络搭建、模型评估、可视化图表、性能对比分析 |
