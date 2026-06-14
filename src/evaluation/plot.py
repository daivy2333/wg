"""M5 对比可视化模块 — matplotlib 图表生成

提供 5 个纯 matplotlib 静态可视化函数，用于 M5 阶段的对比分析：
- 混淆矩阵热力图
- 多模型 ROC 曲线
- 攻击类别 F1 柱状图
- DT vs RF 特征重要性对比
- 深度学习 vs 传统 ML 关键指标对比

所有函数使用 Agg 后端，统一 dpi=80，调用约定为：
    plt.tight_layout() → 创建父目录 → plt.savefig() → plt.close()
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix


# 统一使用 whitegrid 样式（与 tab10 调色板兼容，matplotlib 3.10 验证通过）
plt.style.use("seaborn-v0_8-whitegrid")

# 图表默认参数
DEFAULT_DPI: int = 80

# tab10 调色板（与 matplotlib 默认 C0..C9 一致；显式列出避免被 style 覆盖）
TAB10_COLORS: list[str] = [
    "#1f77b4",  # C0 — blue   (DT)
    "#ff7f0e",  # C1 — orange (CNN)
    "#2ca02c",  # C2 — green  (RF)
    "#d62728",  # C3 — red    (MLP)
    "#9467bd",  # C4 — purple (LSTM)
    "#8c564b",  # C5 — brown
    "#e377c2",  # C6 — pink
    "#7f7f7f",  # C7 — gray
    "#bcbd22",  # C8 — olive
    "#17becf",  # C9 — cyan
]


def _ensure_parent_dir(save_path) -> None:
    """确保 save_path 的父目录存在（不存在则递归创建）。"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)


def _save_figure(save_path) -> None:
    """统一的保存收尾：tight_layout → 创建父目录 → savefig → close。"""
    plt.tight_layout()
    _ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=DEFAULT_DPI)
    plt.close()


def plot_confusion_matrix_heatmap(
    y_true,
    y_pred,
    class_labels: list[str],
    title: str,
    save_path,
) -> None:
    """绘制二分类混淆矩阵热力图，并在每个单元格内标注计数。

    Args:
        y_true: 一维真实标签数组（元素为 0/1）。
        y_pred: 一维预测标签数组（元素为 0/1）。
        class_labels: 类别名列表，如 ['Normal', 'Anomaly']。
        title: 图表标题。
        save_path: 输出 PNG 文件路径。
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=DEFAULT_DPI)
    ax.imshow(cm, cmap="Blues", interpolation="nearest")

    n_classes = len(class_labels)
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # 单元格内标注（数值大用白字，数值小用黑字以保证可读性）
    threshold = cm.max() / 2 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    _save_figure(save_path)


def plot_roc_curves(roc_data, save_path) -> None:
    """在同一张图上绘制多个模型的 ROC 曲线。

    Args:
        roc_data: (model_name, fpr, tpr, auc) 元组列表。
        save_path: 输出 PNG 文件路径。
    """
    fig, ax = plt.subplots(figsize=(7, 6), dpi=DEFAULT_DPI)

    for idx, (model_name, fpr, tpr, auc) in enumerate(roc_data):
        color = TAB10_COLORS[idx % len(TAB10_COLORS)]
        ax.plot(
            fpr,
            tpr,
            color=color,
            linewidth=2,
            label=f"{model_name} (AUC={auc:.4f})",
        )

    # 随机分类基线
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, linewidth=1)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right", fontsize=9)

    _save_figure(save_path)


def plot_f1_by_category_bars(f1_dicts, save_path) -> None:
    """按攻击类别分组的 F1 柱状图（多模型并列）。

    Args:
        f1_dicts: {model_name: {category: f1_value, ...}, ...}。
                   category 须覆盖 DoS/Probe/R2L/U2R，缺失记为 0。
        save_path: 输出 PNG 文件路径。
    """
    categories: list[str] = ["DoS", "Probe", "R2L", "U2R"]
    n_models = len(f1_dicts)
    bar_width = 0.8 / max(n_models, 1)
    x = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=DEFAULT_DPI)

    for idx, (model_name, f1_by_cat) in enumerate(f1_dicts.items()):
        values = [float(f1_by_cat.get(cat, 0.0)) for cat in categories]
        # 柱位以 0 为中心对称分布
        positions = x + (idx - (n_models - 1) / 2) * bar_width
        color = TAB10_COLORS[idx % len(TAB10_COLORS)]

        bars = ax.bar(positions, values, bar_width, label=model_name, color=color)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score by Attack Category")
    ax.legend(fontsize=9)

    _save_figure(save_path)


def plot_feature_importance_comparison(
    dt_imp,
    rf_imp,
    top_k: int,
    save_path,
) -> None:
    """DT 与 RF Top-K 特征重要性并排水平柱状图。

    Args:
        dt_imp: pd.Series，索引为特征名，值为重要性分数（DecisionTree.feature_importances_）。
        rf_imp: pd.Series，同上（RandomForest.feature_importances_）。
        top_k: 取重要性最大的 K 个特征。
        save_path: 输出 PNG 文件路径。
    """
    # 升序排序后用 barh 绘制，最大的特征自然位于顶端
    dt_top = dt_imp.nlargest(top_k).sort_values(ascending=True)
    rf_top = rf_imp.nlargest(top_k).sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=DEFAULT_DPI, sharey=True)

    axes[0].barh(dt_top.index, dt_top.values, color=TAB10_COLORS[0])
    axes[0].set_title("Decision Tree")
    axes[0].set_xlabel("Importance")

    axes[1].barh(rf_top.index, rf_top.values, color=TAB10_COLORS[2])
    axes[1].set_title("Random Forest")
    axes[1].set_xlabel("Importance")

    fig.suptitle(f"Feature Importance: DT vs RF (Top-{top_k})")
    _save_figure(save_path)


def plot_dl_vs_ml_comparison(metrics_dict, save_path) -> None:
    """深度学习与传统 ML 在 accuracy / f1 / auc 三项指标上的对比柱状图。

    Args:
        metrics_dict: {model_name: {"accuracy": float, "f1": float, "auc": float}, ...}。
        save_path: 输出 PNG 文件路径。
    """
    metric_names: list[str] = ["accuracy", "f1", "auc"]
    n_models = len(metrics_dict)
    bar_width = 0.8 / max(n_models, 1)
    x = np.arange(len(metric_names))

    fig, ax = plt.subplots(figsize=(9, 5), dpi=DEFAULT_DPI)

    for idx, (model_name, metrics) in enumerate(metrics_dict.items()):
        values = [float(metrics.get(m, 0.0)) for m in metric_names]
        positions = x + (idx - (n_models - 1) / 2) * bar_width
        color = TAB10_COLORS[idx % len(TAB10_COLORS)]

        bars = ax.bar(positions, values, bar_width, label=model_name, color=color)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metric_names])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Deep Learning vs Traditional ML: Key Metrics")
    ax.legend(fontsize=8)

    _save_figure(save_path)
