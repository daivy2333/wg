## Context

M1 完成数据加载与基础预处理（`src/data/loader.py` + `src/data/preprocessor.py` + 29 tests 全通过）。M2 任务（tasks.md 2.1-2.10）有 4 个与 M1 重叠（2.5/2.6/2.8 已实现，2.1/2.3 notebook 已基本完成），需新增 5 项能力：异常值处理、特征选择、数据持久化、深度 EDA、报告撰写。

**M2 关键约束**：
- 不修改 M1 既有模块（向后兼容）
- 41 个特征中部分长尾分布严重（如 src_bytes max=1.38e9），需 log1p 变换
- NSL-KDD 特征选择文献常用 RF + Variance Threshold 双保险
- pickle 输出固定 Python 3.13.5 环境，跨版本不保证

## Goals / Non-Goals

**Goals:**
- 打通 M2 五项产出（outlier / feature_selector / persistence / eda_report / notebook 扩展）
- 产出 M3/M4 可直接消费的 `.pkl` 数据文件
- TDD Iron Law：outlier/feature_selector/persistence 三个新模块必须有 pytest 测试
- 异常值不丢样本（clip 而非 drop，保留 125973 条全量）

**Non-Goals:**
- 不修改 M1 模块（loader / preprocessor / m1-infrastructure 归档版本）
- 不实现 SMOTE（M4 范围）
- 不实现模型训练（M3/M4 范围）
- 不输出多分类（>=23 类）的特征重要度分析（仅二分类做 RF 训练）

## Decisions

### 决策 1: IQR + clip 异常值处理（不丢样本）

- **方法**：IQR = Q3 - Q1；超限 = Q1 - 1.5*IQR 或 Q3 + 1.5*IQR 之外的值 → clip 到边界
- **原因**：NSL-KDD 攻击样本本就含极端值（如 src_bytes 在 buffer_overflow 攻击下巨大），丢样本会破坏攻击语义
- **替代方案**：
  - Z-score + drop：丢样本（违反不丢样本原则）
  - winsorize（5%/95% 分位数）：与 IQR 类似但分位数固定，IQR 更鲁棒
  - 不处理：保留异常，M3 决策树天然鲁棒但 M4 MLP 会受极端值影响
- **影响**：每列异常值数量记录在 EDA 报告，超限值 clip 到边界（保留行号）

### 决策 2: 特征选择双保险（方差阈值 + RF 重要度）

- **方法**：
  - 步骤 1: VarianceThreshold(threshold=0.01) 过滤近零方差特征
  - 步骤 2: RandomForestClassifier(n_estimators=100, random_state=42) 训练 → feature_importances_ → Top-K
- **原因**：
  - 方差阈值：去掉几乎不变的特征（如 is_host_login 在 NSL-KDD 中几乎全 0）
  - RF 重要度：基于二分类任务，找出对分类贡献最大的 K 个特征
- **替代方案**：
  - 卡方检验：仅适用正特征，与混合 OneHot + Label 数据不兼容
  - L1 正则化：需要训练线性模型，计算成本高
  - 全部 53 列：不选择（影响 M3 训练速度与可解释性）
- **影响**：Top-K 默认 = 20（基于 NSL-KDD 文献经验值）；K 可配置

### 决策 3: log1p 变换长尾特征

- **范围**：`src_bytes`、`dst_bytes`、`count`、`srv_count`（数值跨度 > 6 个数量级）
- **方法**：`np.log1p(x)` = `log(1 + x)`，处理 x=0 的情况
- **原因**：长尾分布导致 StandardScaler 后仍有偏斜，log1p + StandardScaler 双重处理更接近正态
- **替代方案**：
  - Box-Cox：需 x > 0，src_bytes 有 0 值需偏移
  - 不变换：依赖 M3 决策树（鲁棒）+ M4 MLP 加 BatchNorm（M4 范围）
- **影响**：变换后列名加 `_log1p` 后缀，便于追溯

### 决策 4: pickle 持久化（二分类 + 多分类双套）

- **输出**：
  - `outputs/processed/X_train.pkl`、`X_test.pkl`、`y_train.pkl`、`y_test.pkl`（二分类）
  - `outputs/processed/X_train_multi.pkl`、`X_test_multi.pkl`、`y_train_multi.pkl`、`y_test_multi.pkl`（多分类）
- **原因**：
  - 二分类是 M3 决策树/RF 主任务
  - 多分类是 M4 MLP/CNN-LSTM 主任务（也是 M5 对比的关键维度）
  - pickle 保留 pandas DataFrame 元数据（列名 + dtype）+ numpy 数组
- **替代方案**：
  - CSV：丢失 dtype 信息，文件大
  - parquet：需 pyarrow 依赖（conda 已装但增加复杂度）
  - numpy .npz：丢失列名
- **影响**：M3/M4 用 `pickle.load(open(p, 'rb'))` 直接消费

### 决策 5: 模块边界（M2 不破坏 M1）

```
src/data/
├── loader.py             # M1（不动）
├── preprocessor.py       # M1（不动）
├── outlier.py            # M2 新增
├── feature_selector.py   # M2 新增
└── persistence.py        # M2 新增
```

- **原因**：M1 已归档，新模块独立；M3 同时调用 M1 + M2 模块
- **影响**：M3 入口需 import 多个模块（一次性导入成本 < 1ms，可接受）

## Risks / Trade-offs

| 风险 | 影响 | 缓解 |
|------|------|------|
| log1p 改变物理可解释性 | 论文可解释性略降 | 保留原始数值列对照，eda_report.md 标注变换列 |
| RF 训练 125k 行耗时 | notebook 执行慢 | `n_estimators=100`（平衡精度/速度），禁用 verbose |
| pickle 跨 Python 版本 | M3 误用其他 Python 报错 | 在 eda_report.md 顶部固定 Python 3.13.5 版本 |
| Top-20 可能漏掉关键特征 | M3 准确率下降 | eda_report.md 列出 Top-20 + Top-30 + Top-50 三组对比 |
| IQR clip 不适合所有特征 | 部分特征仍偏斜 | 仅对数值长尾特征 clip，OneHot/0-1 特征跳过 |
| 二分类 RF 重要度不能直接套用多分类 | 多分类特征选择缺失 | M2 范围内仅做二分类 RF，多分类在 M4 范围内补 |

## Migration Plan

- **部署步骤**：M2 完成后，M3 启动前由 B 同学 `pickle.load` 4 个文件直接使用
- **回滚策略**：删除 `src/data/outlier.py`、`src/data/feature_selector.py`、`src/data/persistence.py`、`outputs/processed/*.pkl`、`docs/eda_report.md`，notebook 还原到 M1 版本（git revert）
- **依赖顺序**：M1 → M2 → M3 → M4 → M5 → M6（M3/M4 严格依赖 M2 的 pickle 输出）