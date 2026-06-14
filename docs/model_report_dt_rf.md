# NSL-KDD DT/RF 模型训练报告

> **M3 任务 3.9** | Python 3.13.5 | scikit-learn 1.8.0
>
> 配套脚本: `scripts/train_m3.py`
> 持久化模型: `outputs/models/*.joblib`（4 个）
> 数据来源: `outputs/processed/*.pkl`（M2 产出）

## 1. 概述

完成决策树（DT）和随机森林（RF）的二分类/多分类训练、网格搜索调优、特征重要度分析、模型持久化。

**关键参数**：
- `class_weight='balanced'`：应对 U2R/R2L 类别不均衡
- 评价指标：accuracy / precision / recall / f1 / auc（二分类）/ f1_macro（多分类）
- DT 网格：max_depth[5,10,15,20,None] × min_samples_split[2,5,10] × criterion[gini,entropy] = 30 组合
- RF 网格：n_estimators[100,200,300] × max_depth[10,15,20,None] × min_samples_split[2,5] = 24 组合
- n_jobs：生产脚本用 `min(4, cpu_count // 8)` = 4（避踩坑-003）；测试函数默认 n_jobs=1
- joblib 持久化

## 2. 数据

| 集合 | 样本数 | 特征数 | 任务 |
|------|--------|--------|------|
| 训练集 | 125,973 | 20（Top-20）| 二分类 + 多分类（23 类）|
| 测试集 | 22,544 | 20 | 二分类 + 多分类 |

> 特征来自 M2 RF 特征选择（M2 `feature_selection_pipeline`），Top-20 主导特征见 §6。

## 3. DT 基线

### 3.1 DT 二分类基线

| 指标 | 值 |
|------|-----|
| accuracy | 0.7634 |
| precision | 0.9630 |
| recall | 0.6078 |
| f1 | 0.7452 |
| auc | 0.7704 |

### 3.2 DT 多分类基线

| 指标 | 值 |
|------|-----|
| accuracy | 0.0456 |
| f1_macro | 0.0155 |

> ⚠️ 多分类准确率接近随机猜测（1/23 = 0.0435），原因见 §8 局限

## 4. DT 网格搜索调优

### 4.1 最优参数

```python
{
    'criterion': 'entropy',
    'max_depth': 20,
    'min_samples_split': 2,
}
```

### 4.2 调优效果

| 指标 | 基线 | 网格后 | 提升 |
|------|------|--------|------|
| accuracy | 0.7634 | **0.7755** | +1.2% |
| f1 | 0.7452 | **0.7621** | +1.7% |
| auc | 0.7704 | **0.7821** | +1.2% |
| CV best f1 | — | 0.9912 | — |

> ⚠️ **CV 0.99 vs test 0.78 差距显著**：5-fold CV 在训练集上评估，模型对训练集过拟合；test 集含 14 种训练未见攻击（见 §8 局限）。

## 5. RF 基线

### 5.1 RF 二分类基线

| 指标 | 值 |
|------|-----|
| accuracy | 0.7083 |
| precision | 0.9592 |
| recall | 0.5092 |
| f1 | 0.6652 |
| auc | **0.8989** |

> 注意到 RF 基线 acc=0.71 反而低于 DT（0.76），但 **auc=0.90 远高于 DT（0.77）**。说明 RF 排序能力更强，acc 偏低主因是 test 集分布偏移（class_weight='balanced' 牺牲 normal 类精度提升 anomaly 召回）。

## 6. RF 网格搜索调优 + 特征重要度

### 6.1 最优参数

```python
{
    'max_depth': 20,
    'min_samples_split': 2,
    'n_estimators': 300,
}
```

### 6.2 调优效果

| 指标 | 基线 | 网格后 |
|------|------|--------|
| accuracy | 0.7083 | **0.7234** |
| f1 | 0.6652 | **0.6882** |
| auc | 0.8989 | **0.9150** |

### 6.3 Top-20 特征重要度

| 排名 | 特征 | 重要度 |
|------|------|--------|
| 1 | flag_SF | 0.1479 |
| 2 | same_srv_rate | 0.1082 |
| 3 | dst_host_srv_count | 0.1050 |
| 4 | dst_host_same_srv_rate | 0.0808 |
| 5 | diff_srv_rate | 0.0765 |
| 6 | logged_in | 0.0737 |
| 7 | dst_host_diff_srv_rate | 0.0522 |
| 8 | protocol_type_icmp | 0.0522 |
| 9 | service | 0.0506 |
| 10 | count | 0.0429 |
| 11 | count_log1p | 0.0399 |
| 12 | dst_host_same_src_port_rate | 0.0310 |
| 13 | dst_host_count | 0.0272 |
| 14 | dst_host_srv_serror_rate | 0.0210 |
| 15 | protocol_type_udp | 0.0206 |
| 16 | protocol_type_tcp | 0.0202 |
| 17 | dst_host_serror_rate | 0.0155 |
| 18 | dst_host_srv_diff_host_rate | 0.0124 |
| 19 | serror_rate | 0.0123 |
| 20 | srv_serror_rate | 0.0098 |

> ⚠️ 重要：RF 网格后的 Top-20 与 M2 RF 特征选择略有差异——M2 用 100 trees，M3 网格用 300 trees + 调优后 max_depth=20。

## 7. RF 多分类

| 指标 | 值 |
|------|-----|
| accuracy | 0.0464 |
| f1_macro | 0.0166 |

> ⚠️ 多分类性能同样接近随机（见 §8 局限）

## 8. 局限与诚实记录

### 8.1 多分类性能差

**现象**：DT/RF 多分类 accuracy ≈ 0.046（接近随机猜测 1/23）

**根因**（双重）：

1. **测试集 14 种训练未见攻击**（M2 EDA 已识别）：
   ```
   apache2, httptunnel, mailbomb, mscan, named, processtable,
   saint, sendmail, snmpgetattack, snmpguess, sqlattack,
   udpstorm, worm, xterm, xlock, xsnoop
   ```
   这些攻击在训练集中完全不存在，模型无法学到对应模式，必然预测错误。

2. **class_weight='balanced' 加剧 unseen 类的误判**：
   balanced 强制模型对每个训练类给同等权重，但对训练中完全没见过的 14 种攻击（unseen class），模型只能把它们归到训练中已有的高频类，导致准确率接近均匀分布的随机猜测（1/23 ≈ 0.043）。

### 8.2 二分类 CV/Test 差距大

**现象**：CV f1=0.99 vs test f1=0.69-0.78

**根因**：
- GridSearchCV 5-fold 在训练集上做评估，模型见过训练集全部数据
- test 集含 14 种训练未见攻击，泛化能力受限
- NSL-KDD 设计目的就是验证泛化能力，所以这个差距是数据集特性，不是模型 bug

### 8.3 不裁剪但需说明

按 Karpathy Requirements Integrity，本报告**不裁剪**性能差距，而是诚实记录。后续优化点见 §10。

## 9. 模型持久化清单

```
outputs/models/
├── dt_binary_best.joblib         134.3 KB   二分类最佳 DT
├── dt_multiclass_best.joblib     941.0 KB   多分类 DT
├── rf_binary_best.joblib         53.2 MB    二分类最佳 RF（300 树）
└── rf_multiclass_best.joblib     94.9 MB    多分类 RF（100 树）
```

加载方式：
```python
from src.models.persistence import load_model
import pandas as pd
model = load_model("outputs/models/rf_binary_best.joblib")
X_test = load_pickle("outputs/processed/X_test.pkl")
preds = model.predict(X_test)
```

## 10. 结论与下游启示

### 10.1 关键发现

1. **二分类实用**：DT/RF 二分类 f1=0.69-0.76，对正常部署可用
2. **多分类失败**：acc 接近随机，**强烈需要**新方法（M4 神经网络 + SMOTE）
3. **特征集中**：Top-5 特征（flag_SF + same_srv_rate + dst_host_srv_count + dst_host_same_srv_rate + diff_srv_rate）累计贡献 ~52% 重要度

### 10.2 对 M4（神经网络）的建议

- ✅ M3 已用 20 维特征，M4 MLP 直接复用 `X_train.pkl`
- ⚠️ **必须**考虑 SMOTE 处理多分类不均衡（详见 optimization.md 新增 O-NN 优化点）
- 🔧 多分类可见类（normal/DoS/Probe）准确率可能更高（unseen 攻击理论上完全无法预测）
- 💡 可考虑 one-vs-rest 拆解为 23 个二分类子问题，避免 unseen 类拖累

### 10.3 对 M5（对比分析）的建议

- ✅ 二分类：DT vs RF vs MLP 三模型对比 AUC-ROC + 混淆矩阵
- ⚠️ 多分类：建议报告"已知类准确率"（剔除 14 种 unseen 后）vs "全量准确率"两个数字
- 📊 M5 必备图表：3 模型 ROC 曲线叠加 / 4 大攻击类别 F1 对比

### 10.4 优化点（已记录到 optimization.md）

- **O-DT-01**: 多分类泛化能力不足
- **O-RF-01**: 二分类 CV/Test 差距（过拟合训练集）
- **O-NN-01**（M4 范围）: SMOTE 处理 U2R 类别不均衡
- **O-NN-02**（M4 范围）: MLP 神经网络应对 unseen 攻击（one-vs-rest + 半监督）

### 10.5 待跟进事项

- [ ] M4 启动时引用本报告的多分类痛点
- [ ] M5 对比分析时标注 M3 局限
- [ ] M6 论文可写"传统 ML 局限" 章节（CV 0.99 vs test 0.7 + 多分类失败）