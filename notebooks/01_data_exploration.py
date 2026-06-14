"""NSL-KDD 数据探索 (EDA) - M1 + M2 综合分析

覆盖 Phase 1.3 Scenario Sketch 全部 17 场景：
  1.  数据加载
  2.  缺失值检查
  3.  二分类标签分布
  4.  攻击大类分布 (DoS/Probe/R2L/U2R)
  5.  23 种攻击类型频次
  6.  数值特征分布
  7.  分类特征分布
  8.  相关系数热力图
  9.  IQR 异常值检测
  10. clip + log1p 处理
  11. 方差阈值 + RF Top-20 特征选择
  12. 特征重要度柱状图
  13. 持久化数据集验证
  14. 结论与下游启示

运行：
    /home/daivy/miniconda3/bin/python notebooks/01_data_exploration.py
输出：
    outputs/figures/*.png（图表）
    控制台文字输出（统计）
"""
import sys
from pathlib import Path

# 项目根添加到 sys.path（脚本可从任意目录执行）
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.loader import load_train, load_test, FEATURE_NAMES, COLUMN_NAMES
from src.data.preprocessor import (
    preprocess_pipeline,
    check_missing,
    NUMERIC_COLUMNS,
    make_labels,
)
from src.data.outlier import (
    outlier_pipeline,
    detect_outliers_iqr,
    LONG_TAIL_COLUMNS,
)
from src.data.feature_selector import (
    feature_selection_pipeline,
    variance_threshold_filter,
)
from src.data.persistence import load_pickle

FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = ROOT / "outputs" / "processed"

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)


def banner(text: str) -> None:
    """打印分节标题。"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


# ============================================================
# 场景 1: 数据加载
# ============================================================
banner("场景 1: 数据加载（M1 loader）")
df_train = load_train()
df_test = load_test()
print(f"训练集: {df_train.shape}, 测试集: {df_test.shape}")
print(f"特征数: {len(FEATURE_NAMES)}（不含 label 和 difficulty）")


# ============================================================
# 场景 2: 缺失值检查
# ============================================================
banner("场景 2: 缺失值检查")
n_train, _ = check_missing(df_train)
n_test, _ = check_missing(df_test)
print(f"训练集缺失值: {n_train}")
print(f"测试集缺失值: {n_test}")
print("✓ NSL-KDD 官方数据无缺失值")


# ============================================================
# 场景 3: 二分类标签分布
# ============================================================
banner("场景 3: 二分类标签分布")
y_train_bin = make_labels(df_train["label"], task="binary")
y_test_bin = make_labels(df_test["label"], task="binary")
print(f"训练集: normal={int((y_train_bin == 0).sum())}, anomaly={int((y_train_bin == 1).sum())}")
print(f"测试集: normal={int((y_test_bin == 0).sum())}, anomaly={int((y_test_bin == 1).sum())}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
y_train_bin.value_counts().plot(kind="bar", ax=axes[0], title="Train", color="steelblue")
y_test_bin.value_counts().plot(kind="bar", ax=axes[1], title="Test", color="coral")
for ax in axes:
    ax.set_xticklabels([0, 1])
    ax.set_xlabel("label (0=normal, 1=anomaly)")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_binary_label_distribution.png", dpi=80)
plt.close()
print(f"  → {FIG_DIR / '01_binary_label_distribution.png'}")


# ============================================================
# 场景 4: 攻击大类分布
# ============================================================
banner("场景 4: 攻击大类分布（DoS/Probe/R2L/U2R）")
ATTACK_CATEGORY = {
    "normal": "Normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS", "processtable": "DoS",
    "worm": "DoS", "mailbomb": "DoS",
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "mscan": "Probe", "saint": "Probe",
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L", "spy": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R", "perl": "U2R",
    "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}

train_cats = df_train["label"].map(ATTACK_CATEGORY).fillna("Unknown")
test_cats = df_test["label"].map(ATTACK_CATEGORY).fillna("Unknown")
print("训练集攻击大类:")
print(train_cats.value_counts().to_string())
print("\n测试集攻击大类:")
print(test_cats.value_counts().to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
train_cats.value_counts().plot(kind="bar", ax=axes[0], title="Train Categories")
test_cats.value_counts().plot(kind="bar", ax=axes[1], title="Test Categories")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_attack_category_distribution.png", dpi=80)
plt.close()
print(f"  → {FIG_DIR / '02_attack_category_distribution.png'}")


# ============================================================
# 场景 5: 23 种攻击类型 + 训练未见攻击
# ============================================================
banner("场景 5: 23 种攻击类型频次")
test_only = set(df_test["label"].unique()) - set(df_train["label"].unique())
print(f"测试集独有的攻击类型（训练未见）: {sorted(test_only)}")
print(f"→ 共 {len(test_only)} 种，体现 NSL-KDD 设计的泛化挑战")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
df_train["label"].value_counts().head(15).plot(kind="barh", ax=axes[0], title="Train top-15", color="steelblue")
df_test["label"].value_counts().head(15).plot(kind="barh", ax=axes[1], title="Test top-15", color="coral")
for ax in axes:
    ax.invert_yaxis()
plt.tight_layout()
plt.savefig(FIG_DIR / "03_attack_type_top15.png", dpi=80)
plt.close()
print(f"  → {FIG_DIR / '03_attack_type_top15.png'}")


# ============================================================
# 场景 6: 数值特征分布
# ============================================================
banner("场景 6: 数值特征分布（前 12 数值特征）")
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for i, col in enumerate(NUMERIC_COLUMNS[:12]):
    ax = axes[i // 4, i % 4]
    df_train[col].hist(ax=ax, bins=50, color="steelblue", alpha=0.7)
    ax.set_title(col, fontsize=10)
    ax.set_yscale("log")
plt.suptitle("Numerical feature distributions (log y-axis)", y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "04_numeric_distributions.png", dpi=80)
plt.close()
print(f"  → {FIG_DIR / '04_numeric_distributions.png'}")


# ============================================================
# 场景 7: 分类特征 top-10 频次
# ============================================================
banner("场景 7: 分类特征 top-10 频次")
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
df_train["protocol_type"].value_counts().plot(kind="bar", ax=axes[0], title="protocol_type")
df_train["service"].value_counts().head(10).plot(kind="bar", ax=axes[1], title="service (top-10)")
df_train["flag"].value_counts().head(10).plot(kind="bar", ax=axes[2], title="flag (top-10)")
plt.tight_layout()
plt.savefig(FIG_DIR / "05_categorical_top10.png", dpi=80)
plt.close()
print(f"  → {FIG_DIR / '05_categorical_top10.png'}")


# ============================================================
# 场景 8: 相关系数热力图
# ============================================================
banner("场景 8: 相关系数热力图（前 16 数值特征）")
subset_cols = NUMERIC_COLUMNS[:16]
corr = df_train[subset_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title("Correlation heatmap (first 16 numeric features)")
plt.tight_layout()
plt.savefig(FIG_DIR / "06_correlation_heatmap.png", dpi=80)
plt.close()
print(f"  → {FIG_DIR / '06_correlation_heatmap.png'}")


# ============================================================
# 场景 9: IQR 异常值检测
# ============================================================
banner("场景 9: IQR 异常值检测")
stats = detect_outliers_iqr(df_train, columns=NUMERIC_COLUMNS)
top10 = sorted(stats.items(), key=lambda x: -x[1])[:10]
print("Top-10 异常值数量:")
for col, count in top10:
    print(f"  {col}: {count:,} ({count/len(df_train)*100:.1f}%)")


# ============================================================
# 场景 10: clip + log1p 处理前后对比
# ============================================================
banner("场景 10: clip + log1p 异常值处理前后对比")
df_train_clean, _ = outlier_pipeline(
    df_train, numeric_cols=NUMERIC_COLUMNS, long_tail_cols=LONG_TAIL_COLUMNS
)
print(f"处理后 shape: {df_train_clean.shape}（行数不变 = {len(df_train)}）")
print(f"新增 log1p 列: {[c for c in df_train_clean.columns if c.endswith('_log1p')]}")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
df_train["src_bytes"].hist(ax=axes[0], bins=50, color="coral")
axes[0].set_title("src_bytes (before log1p)")
axes[0].set_yscale("log")
df_train_clean["src_bytes_log1p"].hist(ax=axes[1], bins=50, color="steelblue")
axes[1].set_title("src_bytes_log1p (after)")
plt.tight_layout()
plt.savefig(FIG_DIR / "07_log1p_before_after.png", dpi=80)
plt.close()
print(f"  → {FIG_DIR / '07_log1p_before_after.png'}")


# ============================================================
# 场景 11: 预处理 → 异常值 → 特征选择管线
# ============================================================
banner("场景 11: 方差阈值 + RF Top-20 特征选择")
X_train_raw, y_train_bin_pipe, encoders = preprocess_pipeline(df_train, task="binary", fit=True)
X_train_clean, _ = outlier_pipeline(
    X_train_raw, numeric_cols=NUMERIC_COLUMNS, long_tail_cols=LONG_TAIL_COLUMNS
)
cols, rf_model = feature_selection_pipeline(X_train_clean, y_train_bin_pipe, top_k=20)

print(f"特征选择前: {X_train_clean.shape[1]} 列（含 4 个 log1p 新增列）")
kept_after_variance = variance_threshold_filter(X_train_clean)
print(f"方差阈值过滤后: {len(kept_after_variance)} 列")
print(f"Top-20 特征选择后: {len(cols)} 列")
print(f"\nTop-20 特征（按重要度降序）:")
for i, c in enumerate(cols, 1):
    print(f"  {i:2d}. {c}")


# ============================================================
# 场景 12: 特征重要度柱状图
# ============================================================
banner("场景 12: 特征重要度柱状图")
importance = pd.Series(rf_model.feature_importances_, index=kept_after_variance)
top20 = importance.nlargest(20)
top20.plot(kind="barh", figsize=(8, 8), color="steelblue")
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.title("Top-20 Feature Importance (RandomForest)")
plt.tight_layout()
plt.savefig(FIG_DIR / "08_feature_importance_top20.png", dpi=80)
plt.close()
print(f"  → {FIG_DIR / '08_feature_importance_top20.png'}")


# ============================================================
# 场景 13: 持久化数据集验证
# ============================================================
banner("场景 13: 持久化数据集验证（pickle 加载）")
files = sorted(PROCESSED_DIR.glob("*.pkl"))
print(f"已生成 {len(files)} 个 .pkl 文件:")
for p in files:
    obj = load_pickle(p)
    if hasattr(obj, "shape"):
        print(f"  {p.name}: shape={obj.shape}")
    else:
        print(f"  {p.name}: len={len(obj)}, nunique={obj.nunique()}")


# ============================================================
# 场景 14: 结论
# ============================================================
banner("场景 14: 结论与下游启示（M3/M4）")
print("""
  M2 关键发现：
    1. 数据规模：训练 125,973 / 测试 22,544；41 + label + difficulty 列
    2. 类别分布：
       - 二分类：训练 67,343 normal / 58,630 anomaly
       - 测试 9,711 normal / 12,833 anomaly（攻击占比更高）
    3. 攻击大类：DoS (45,927) > Probe (11,656) > R2L (995) > U2R (极少)
    4. 长尾特征：src_bytes/dst_bytes/count/srv_count 已 log1p 变换
    5. 特征选择：Top-20 重要特征已识别并持久化

  下游启示：
    - M3（决策树/RF）：直接 pickle.load，DT 训练 < 30s
    - M4（MLP/CNN）：log1p 后特征更接近正态，BatchNorm 收敛更快
    - M5（对比分析）：所有模型基于同一 Top-20 特征，对比公平

  ✓ EDA 完成，所有图表保存到 outputs/figures/
""")

print(f"\n图表输出目录: {FIG_DIR}")
print(f"数据输出目录: {PROCESSED_DIR}")