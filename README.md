# NSL-KDD 网络入侵检测系统

基于 NSL-KDD 数据集的网络入侵检测系统，使用传统机器学习（决策树、随机森林）与深度学习（MLP、CNN、LSTM）多模型对比，覆盖二分类异常检测与多分类攻击识别。代码、测试、实验报告、论文一站式产出，全流程在本地 Python 环境即可复现。

## 环境配置

项目使用本地 conda 环境，Python 3.10+。推荐 Python 3.13.5（与本仓库锁定版本一致）。

### 1. 定位 conda 环境

```bash
which conda  # 找到 miniconda 路径，例如 /home/daivy/miniconda3
```

### 2. 安装依赖

依赖已锁定在 `requirements.txt`，一行命令安装即可：

```bash
pip install -r requirements.txt
```

或手动安装核心包：

```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch torchvision pytest pytest-cov
```

### 3. 验证环境

```bash
python -c "import torch, sklearn, pandas; print('torch', torch.__version__); print('sklearn', sklearn.__version__); print('pandas', pandas.__version__)"
```

本仓库锁定版本：

| 包 | 版本 |
|---|---|
| Python | 3.13.5 |
| PyTorch | 2.9.0 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.2 |
| NumPy | 2.3.3 |

## 数据集

### 下载

NSL-KDD 官方 UNB 网站已停止数据下载，使用 Kaggle 或国内镜像：

| 来源 | 链接 |
|---|---|
| Kaggle（首选） | https://www.kaggle.com/datasets/hassan06/nslkdd |
| 国家基础学科公共科学数据中心 | https://www.nbsdc.cn/general/dataDetail?id=67d50d83195d260905af969e |
| Zenodo V2 扩展版 | https://zenodo.org/records/10141556 |

### 解压

下载 `archive.zip` 后解压到 `dataset/` 目录，确保以下文件存在：

```
dataset/
├── KDDTrain+.txt      # 训练集，125,973 条
├── KDDTest+.txt       # 测试集，22,544 条
└── KDDTrain+_20Percent.txt   # 可选，20% 子集用于快速实验
```

数据集共 41 个特征 + 1 个攻击类型标签 + 1 个难度等级，覆盖 23 种攻击类型（训练集）和 38 种攻击类型（测试集含 15 种训练未见攻击，是多分类的核心挑战）。

## 项目结构

```
wg/
├── CLAUDE.md                    # 项目规则与入口文档
├── README.md                    # 本文件
├── requirements.txt             # 依赖锁定
├── readme                       # 原始任务说明（保留不动）
│
├── dataset/                     # NSL-KDD 原始数据
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
│
├── src/                         # 核心代码库
│   ├── data/                    # M1+M2：数据加载、预处理、特征选择、异常值、SMOTE
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   ├── feature_selector.py
│   │   ├── outlier.py
│   │   ├── persistence.py
│   │   └── smote.py
│   ├── models/                  # M3+M4：DT、RF、MLP、CNN、LSTM
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   ├── mlp.py
│   │   ├── cnn.py
│   │   ├── lstm.py
│   │   └── persistence.py
│   ├── evaluation/              # M5：指标计算、可视化
│   │   ├── metrics.py
│   │   └── plot.py
│   └── utils/
│
├── scripts/                     # 实验入口脚本
│   ├── train_m3.py              # DT/RF 训练
│   ├── train_m4.py              # MLP/CNN/LSTM 训练
│   ├── run_smote_only.py        # SMOTE 过采样实验
│   └── evaluate_m5.py           # 全模型对比评估
│
├── notebooks/
│   └── 01_data_exploration.py   # M2 EDA：分布图、相关性、特征分析
│
├── tests/                       # pytest 测试套件（144 个测试）
│   ├── test_loader.py
│   ├── test_preprocessor.py
│   ├── test_feature_selector.py
│   ├── test_outlier.py
│   ├── test_persistence.py
│   ├── test_decision_tree.py
│   ├── test_random_forest.py
│   ├── test_mlp.py
│   ├── test_cnn.py
│   ├── test_lstm.py
│   ├── test_smote.py
│   ├── test_grid_search_mlp.py
│   ├── test_model_persistence.py
│   ├── test_nn_persistence.py
│   ├── test_evaluation_metrics.py
│   └── test_evaluation_plot.py
│
├── outputs/
│   ├── processed/               # M1+M2 产出的处理后数据（.pkl）
│   ├── models/                  # 10 个持久化模型（4 joblib + 6 torch）
│   ├── figures/                 # 15 张实验图表
│   ├── metrics_m4.json          # M4 训练指标
│   ├── metrics_m5.json          # M5 对比指标
│   └── label_id_to_*.json       # 多分类标签映射
│
├── docs/                        # 实验报告（Markdown）
│   ├── eda_report.md            # M2 数据探索报告
│   ├── model_report_dt_rf.md    # M3 DT/RF 训练报告
│   ├── model_report_mlp_dl.md   # M4 MLP/DL 训练报告
│   └── comparison_report.md     # M5 对比分析报告
│
└── paper/                       # M6 论文产出
    ├── refs.bib                 # BibTeX 参考文献
    ├── chapters/                # 章节 .tex
    │   └── ch1-intro.tex
    └── figures -> ../outputs/figures   # 软链到图表目录
```

## 实验复现

按以下顺序执行，可完整复现全部实验并生成论文所需产物。所有命令在项目根目录运行。

### Step 1：数据准备与探索

```bash
python notebooks/01_data_exploration.py
```

产出：`outputs/figures/01_*.png` 到 `08_*.png`，`outputs/processed/*.pkl`，`docs/eda_report.md`。

### Step 2：传统机器学习训练（DT、RF）

```bash
python scripts/train_m3.py
```

产出：`outputs/models/dt_binary_best.joblib`、`dt_multiclass_best.joblib`、`rf_binary_best.joblib`、`rf_multiclass_best.joblib`，`docs/model_report_dt_rf.md`。

### Step 3：神经网络训练（MLP、CNN、LSTM）

```bash
python scripts/train_m4.py
```

产出：`outputs/models/mlp_*.pt`、`cnn_binary_best.pt`、`lstm_binary_best.pt`、`mlp_multiclass_smote.pt`，`outputs/metrics_m4.json`，`docs/model_report_mlp_dl.md`。

### Step 4：全模型对比分析

```bash
python scripts/evaluate_m5.py
```

产出：`outputs/figures/09_*` 到 `13_*`，`outputs/metrics_m5.json`，`docs/comparison_report.md`。

### Step 5：运行测试套件

```bash
pytest tests/ -v
```

当前测试数：**144 个全部通过**。

## 关键结果

下表汇总在 NSL-KDD 二分类任务上的最佳模型表现（统一测试集，22,544 样本）：

| 模型 | Accuracy | Precision | Recall | F1 | AUC | 推理时间 (s) |
|---|---|---|---|---|---|---|
| **DT** | **0.7755** | 0.9604 | 0.6317 | **0.7621** | 0.7821 | 0.02 |
| RF | 0.7234 | 0.9602 | 0.5363 | 0.6882 | 0.9150 | 0.23 |
| MLP | 0.7423 | 0.9580 | 0.5724 | 0.7166 | 0.9058 | 0.05 |
| MLP (Tuned) | 0.7445 | 0.9560 | 0.5777 | 0.7202 | 0.9074 | 0.10 |
| CNN | 0.7635 | 0.9635 | 0.6075 | 0.7452 | 0.8820 | 0.09 |
| LSTM | 0.7427 | 0.9628 | 0.5700 | 0.7161 | 0.8832 | 0.04 |

每任务最佳模型：

| 任务 | 最佳模型 | F1 / 准确率 | 备注 |
|---|---|---|---|
| 二分类 F1 | **DT** | **F1 = 0.76** | 推理速度最快（0.02s），适合实时场景 |
| 二分类 AUC | **RF** | **AUC = 0.9150** | 概率校准最优，适合阈值敏感业务 |
| 二分类召回率 | **DT** | **Recall = 0.6317** | 异常检出率最高 |
| 多分类（已知类） | MLP | 11.79% | 受限于 15 种训练未见攻击 |

更多细节见 `docs/comparison_report.md`。

## 论文编译

论文使用 XeLaTeX 编译（支持中文），参考文献使用 BibTeX：

```bash
cd paper
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

第一次 `xelatex` 生成 `.aux`，`bibtex` 解析 `refs.bib`，后两次 `xelatex` 解决引用与交叉引用。完整流程产出 `main.pdf`。

章节源文件位于 `paper/chapters/`，图表通过 `paper/figures/` 软链到 `outputs/figures/`。

## 三人分工

本项目按 M1–M6 里程碑推进，对应数据、模型、对比分析三条主线，便于分工分配：

| 角色 | 方向 | 主要产出 | 关键文件 |
|---|---|---|---|
| **A 同学** | 数据方向 | 数据加载、EDA、预处理、特征选择、异常值处理 | `src/data/`, `notebooks/01_data_exploration.py`, `docs/eda_report.md` |
| **B 同学** | 模型方向（传统 ML） | 决策树、随机森林，网格搜索调优，特征重要度 | `src/models/decision_tree.py`, `src/models/random_forest.py`, `scripts/train_m3.py` |
| **C 同学** | 对比分析 + 神经网络 | MLP/CNN/LSTM 训练、SMOTE 实验、指标计算、可视化、论文撰写 | `src/models/mlp.py`, `src/models/cnn.py`, `src/models/lstm.py`, `src/evaluation/`, `scripts/train_m4.py`, `scripts/evaluate_m5.py`, `paper/` |

M1（loader）、M2（EDA + 特征工程）、M3（DT/RF）、M4（MLP/CNN/LSTM + SMOTE）、M5（全模型对比）、M6（论文写作）六个里程碑串联起来正好覆盖三个角色的完整工作链路，每人独立负责自己的脚本与文档，最终在 M5 阶段汇合进行横向对比。

## 注意事项

- OpenCV 等计算机视觉库**不需要**安装，NSL-KDD 是网络流量表格数据，全部任务用 Pandas/NumPy + scikit-learn + Matplotlib + PyTorch 完成。
- 测试集含 15 种训练未见攻击（label 23–37），多分类全量准确率上限约 11%，已知类准确率约 12%，这是 NSL-KDD 的固有设计，不是模型缺陷。
- SMOTE 在本数据集上效果负面（多分类准确率从 10.09% 降到 2.84%），不推荐作为提升手段。
- 论文参考文献见 `paper/refs.bib`，包含 NSL-KDD 原始论文、scikit-learn、PyTorch、KDD Cup 1999、UNB 数据集地址，以及 3 篇中文参考文献。
