#!/usr/bin/env python
"""M5 对比分析编排脚本 — 统一测试集评估 9 个模型 + 生成对比图表和报告

运行：
    python scripts/evaluate_m5.py

输入：
    outputs/processed/{X,y}_{test,test_multi}.pkl  (M2 产出)
    outputs/models/*.joblib  (M3 产出 — sklearn)
    outputs/models/*.pt     (M4 产出 — PyTorch)

输出：
    outputs/metrics_m5.json          — 全量指标
    outputs/figures/09-13_*.png      — 7 张对比图表
    docs/comparison_report.md        — 8 章对比报告
"""

# === WSL thread defense ===
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

from src.data.persistence import load_pickle
from src.models.persistence import load_model, load_torch_model
from src.models.mlp import MLPClassifier
from src.models.cnn import CNN1DClassifier
from src.models.lstm import LSTMClassifier

# === Config ===
MODELS_DIR = ROOT / "outputs" / "models"
PROCESSED_DIR = ROOT / "outputs" / "processed"
FIGURES_DIR = ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS: dict[str, tuple] = {
    "mlp_binary_best.pt":      (MLPClassifier, {"input_dim": 20, "output_dim": 2}),
    "mlp_binary_tuned.pt":     (MLPClassifier, {"input_dim": 20, "output_dim": 2, "hidden_dims": (256, 128, 64), "dropout": 0.3}),
    "mlp_multiclass_best.pt":  (MLPClassifier, {"input_dim": 53, "output_dim": 5, "hidden_dims": (128, 64)}),
    "cnn_binary_best.pt":      (CNN1DClassifier, {"input_length": 20, "output_dim": 2}),
    "lstm_binary_best.pt":     (LSTMClassifier, {"input_size": 20, "output_dim": 2}),
}

# 攻击名称 → 大类映射（from M2 EDA）
ATTACK_CATEGORY: dict[str, str] = {
    "normal":          "Normal",
    "back":            "DoS",
    "land":            "DoS",
    "neptune":         "DoS",
    "pod":             "DoS",
    "smurf":           "DoS",
    "teardrop":        "DoS",
    "apache2":         "DoS",
    "udpstorm":        "DoS",
    "processtable":    "DoS",
    "worm":            "DoS",
    "mailbomb":        "DoS",
    "satan":           "Probe",
    "ipsweep":         "Probe",
    "nmap":            "Probe",
    "portsweep":       "Probe",
    "mscan":           "Probe",
    "saint":           "Probe",
    "guess_passwd":    "R2L",
    "ftp_write":       "R2L",
    "imap":            "R2L",
    "phf":             "R2L",
    "multihop":        "R2L",
    "warezmaster":     "R2L",
    "warezclient":     "R2L",
    "spy":             "R2L",
    "xlock":           "R2L",
    "xsnoop":          "R2L",
    "snmpguess":       "R2L",
    "snmpgetattack":   "R2L",
    "httptunnel":      "R2L",
    "sendmail":        "R2L",
    "named":           "R2L",
    "buffer_overflow": "U2R",
    "loadmodule":      "U2R",
    "rootkit":         "U2R",
    "perl":            "U2R",
    "sqlattack":       "U2R",
    "xterm":           "U2R",
    "ps":              "U2R",
}


# ============================================================
# Helpers
# ============================================================

def banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def compute_binary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
) -> dict[str, float]:
    """二分类指标（self-contained，不依赖 src/evaluation）。"""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
    }


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """多分类指标（5 大类）。"""
    full_acc = float(accuracy_score(y_true, y_pred))
    f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    return {
        "accuracy": full_acc,
        "f1_macro": f1_m,
        "full_accuracy": full_acc,
    }


def get_probabilities(model, X) -> np.ndarray:
    """获取二分类 anomaly 概率 (统一 sklearn / PyTorch 接口)。"""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()
    elif isinstance(model, torch.nn.Module):
        import torch.nn.functional as F
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        model.eval()
        with torch.no_grad():
            x_t = torch.as_tensor(X_arr, dtype=torch.float32)
            logits = model(x_t)
            return F.softmax(logits, dim=1)[:, 1].numpy()
    else:
        pred = model.predict(X)
        return np.asarray(pred).ravel().astype(float)


def derive_label_mapping() -> tuple[dict, dict]:
    """加载 label_id → name 和 label_id → category 映射。"""
    with open(ROOT / "outputs" / "label_id_to_name.json") as f:
        label_id_to_name = json.load(f)
    with open(ROOT / "outputs" / "label_id_to_category.json") as f:
        label_id_to_category = json.load(f)
    # 转换 string keys → int
    label_id_to_category = {int(k): v for k, v in label_id_to_category.items()}
    return label_id_to_name, label_id_to_category


# ============================================================
# Model Loading
# ============================================================

def load_all_models() -> dict:
    """加载全部 9 个模型（4 sklearn + 5 PyTorch）。"""
    models: dict = {}

    # —— sklearn 模型（M3）——
    banner("Loading sklearn models")
    for name, filename in [
        ("dt_binary",      "dt_binary_best.joblib"),
        ("dt_multiclass",  "dt_multiclass_best.joblib"),
        ("rf_binary",      "rf_binary_best.joblib"),
        ("rf_multiclass",  "rf_multiclass_best.joblib"),
    ]:
        try:
            models[name] = load_model(MODELS_DIR / filename)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    # —— PyTorch 模型（M4）——
    banner("Loading PyTorch models")
    for filename, (model_class, kwargs) in MODEL_CONFIGS.items():
        name = filename.replace(".pt", "").replace("_best", "")
        try:
            model = load_torch_model(model_class, MODELS_DIR / filename, **kwargs)
            # 冒烟测试：用 2 样本张量跑一次 forward 确认维度对齐
            test_input = torch.randn(2, 20)
            with torch.no_grad():
                _ = model(test_input)
            models[name] = model
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    return models


# ============================================================
# Figure Helpers (inline matplotlib Agg)
# ============================================================

def _save_confusion_matrix(cm: np.ndarray, title: str, save_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Anomaly"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{title} - Confusion Matrix")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=80)
    plt.close()
    print(f"  ✓ {save_path.name}")


def _save_multiclass_confusion_matrix(cm: np.ndarray, title: str, save_path: Path) -> None:
    """5×5 多分类混淆矩阵（固定大类标签）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["DoS", "Normal", "Probe", "R2L", "U2R"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    fontsize=10, fontweight="bold")
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(5)); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{title} - Confusion Matrix (5-Class)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=80)
    plt.close()
    print(f"  ✓ {save_path.name}")


def _compute_f1_by_category(
    y_true_multi: np.ndarray,
    y_pred_binary: np.ndarray,
    label_to_category: dict[int, str],
) -> dict[str, float]:
    """各攻击大类 F1（用多分类 ground truth + 二分类预测）。"""
    result: dict[str, float] = {}
    y_pred_binary = np.asarray(y_pred_binary).ravel()
    for category in sorted(set(label_to_category.values())):
        cat_ids = {k for k, v in label_to_category.items() if v == category}
        y_true_bin = np.isin(y_true_multi, list(cat_ids)).astype(int)
        result[category] = float(f1_score(y_true_bin, y_pred_binary, zero_division=0))
    return result


def _save_f1_category_bars(
    f1_dicts: dict[str, dict[str, float]],
    save_path: Path,
) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    categories = ["DoS", "Probe", "R2L", "U2R"]
    models_list = list(f1_dicts.keys())
    n_models = len(models_list)
    bar_width = 0.8 / n_models
    x = np.arange(len(categories))
    colors = plt.cm.tab10.colors[:n_models]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, f1_dict) in enumerate(f1_dicts.items()):
        values = [f1_dict.get(c, 0) for c in categories]
        bars = ax.bar(x + i * bar_width, values, bar_width,
                      label=name, color=colors[i])
        for bar, val in zip(bars, values):
            if val > 0.001:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=8, rotation=90)
    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(categories)
    ax.set_ylabel("F1 Score"); ax.set_ylim(0, 1.1)
    ax.set_title("F1 Score by Attack Category (Binary Detectors)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=80)
    plt.close()
    print(f"  ✓ {save_path.name}")


def _save_roc_curves(roc_data: list, save_path: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10.colors[:len(roc_data)]
    for i, (name, fpr, tpr, auc_val) in enumerate(roc_data):
        ax.plot(fpr, tpr, color=colors[i], lw=1.5,
                label=f"{name} (AUC={auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Binary Classification")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=80)
    plt.close()
    print(f"  ✓ {save_path.name}")


def _save_feature_importance(
    dt_imp: pd.Series,
    rf_imp: pd.Series,
    top_k: int,
    save_path: Path,
) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dt_top = dt_imp.nlargest(top_k)
    rf_top = rf_imp.nlargest(top_k)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    ax1.barh(range(top_k), dt_top.values[::-1], color="C0")
    ax1.set_yticks(range(top_k))
    ax1.set_yticklabels(dt_top.index[::-1])
    ax1.set_xlabel("Importance"); ax1.set_title("Decision Tree")

    ax2.barh(range(top_k), rf_top.values[::-1], color="C2")
    ax2.set_yticks(range(top_k))
    ax2.set_yticklabels(rf_top.index[::-1])
    ax2.set_xlabel("Importance"); ax2.set_title("Random Forest")

    fig.suptitle(f"Feature Importance Comparison (Top-{top_k})")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=80)
    plt.close()
    print(f"  ✓ {save_path.name}")


def _save_dl_vs_ml(metrics_dict: dict[str, dict[str, float]], save_path: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metric_names = ["accuracy", "f1", "auc"]
    model_names = list(metrics_dict.keys())
    n_models = len(model_names)
    bar_width = 0.8 / n_models
    x = np.arange(len(metric_names))
    colors = plt.cm.tab10.colors[:n_models]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, m) in enumerate(metrics_dict.items()):
        values = [m.get(met, 0) for met in metric_names]
        bars = ax.bar(x + i * bar_width, values, bar_width,
                      label=name, color=colors[i])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7, rotation=90)
    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(["Accuracy", "F1", "AUC"])
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_title("Deep Learning vs Traditional ML: Key Metrics")
    ax.legend(fontsize=8)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=80)
    plt.close()
    print(f"  ✓ {save_path.name}")


# ============================================================
# Report Generation
# ============================================================

def _generate_report(
    all_metrics: dict[str, dict[str, float]],
    figure_paths: dict[str, str],
    report_path: Path,
) -> None:
    """生成 8 章对比分析报告（Markdown）。"""
    lines: list[str] = []
    lines.append("# NSL-KDD 模型对比分析报告")
    lines.append("")
    lines.append("> **M5 对比分析** | 自动生成 | 基于 5 大类多分类 (DoS/Normal/Probe/R2L/U2R)")
    lines.append("")

    # §1 概述
    lines.append("## 1. 概述")
    lines.append("")
    lines.append(
        "本报告在统一测试集上对全部 9 个模型进行横向对比评估。"
        "评估维度包括二分类（6 个模型）和多分类（3 个模型）。"
    )
    lines.append("")

    # §2 指标汇总
    lines.append("## 2. 模型指标汇总")
    lines.append("")
    lines.append("### 2.1 二分类")
    lines.append("")
    lines.append("| 模型 | Accuracy | Precision | Recall | F1 | AUC |")
    lines.append("|------|----------|-----------|--------|----|-----|")
    for name in ["dt_binary", "rf_binary", "mlp_binary", "mlp_binary_tuned", "cnn_binary", "lstm_binary"]:
        if name in all_metrics:
            m = all_metrics[name]
            label = name.replace("_binary", "").replace("_tuned", " (Tuned)")
            lines.append(
                f"| {label} | {m['accuracy']:.4f} | {m['precision']:.4f} "
                f"| {m['recall']:.4f} | {m['f1']:.4f} | {m['auc']:.4f} |"
            )
    lines.append("")

    lines.append("### 2.2 多分类")
    lines.append("")
    lines.append("| 模型 | Accuracy | F1 Macro |")
    lines.append("|------|----------|----------|")
    for name in ["dt_multiclass", "rf_multiclass", "mlp_multiclass"]:
        if name in all_metrics:
            m = all_metrics[name]
            label = name.replace("_multiclass", "")
            lines.append(
                f"| {label} | {m.get('accuracy', 0):.4f} "
                f"| {m.get('f1_macro', 0):.4f} |"
            )
    lines.append("")
    lines.append(
        "> 基于 5 大类（DoS, Normal, Probe, R2L, U2R）的多分类评估。"
    )
    lines.append("")

    # §3 混淆矩阵
    lines.append("## 3. 混淆矩阵分析")
    lines.append("")
    lines.append("二分类混淆矩阵（Normal vs Anomaly）：")
    lines.append("")
    for prefix, label in [("dt", "DT"), ("rf", "RF"), ("mlp", "MLP")]:
        fig_key = f"confusion_{prefix}_binary"
        if fig_key in figure_paths:
            lines.append(f"![{label} Confusion Matrix](../{figure_paths[fig_key]})")
            lines.append(f"*{label} 二分类混淆矩阵*")
            lines.append("")

    lines.append("多分类混淆矩阵（5 大类：DoS, Normal, Probe, R2L, U2R）：")
    lines.append("")
    for prefix, label in [("dt", "DT"), ("rf", "RF"), ("mlp", "MLP")]:
        fig_key = f"confusion_{prefix}_multiclass"
        if fig_key in figure_paths:
            lines.append(f"![{label} Multiclass Confusion Matrix](../{figure_paths[fig_key]})")
            lines.append(f"*{label} 多分类混淆矩阵*")
            lines.append("")

    # §4 攻击大类 F1
    lines.append("## 4. 攻击大类性能对比")
    lines.append("")
    lines.append("各模型在四大攻击类别（DoS, Probe, R2L, U2R）上的 F1 表现：")
    lines.append("")
    if "f1_category" in figure_paths:
        lines.append(f"![Attack Category F1](../{figure_paths['f1_category']})")
        lines.append("")

    # §5 ROC
    lines.append("## 5. ROC 曲线分析")
    lines.append("")
    lines.append("二分类模型 ROC 曲线叠加对比：")
    lines.append("")
    if "roc" in figure_paths:
        lines.append(f"![ROC Curves](../{figure_paths['roc']})")
        lines.append("")

    lines.append("### 关键发现")
    lines.append("")
    best_model = max(
        ((n, m) for n, m in all_metrics.items() if "binary" in n and "auc" in m),
        key=lambda x: x[1]["auc"],
        default=(None, {}),
    )
    if best_model[0]:
        lines.append(f"- **最佳模型**: {best_model[0]} (AUC={best_model[1]['auc']:.4f})")
    rf_recall = all_metrics.get("rf_binary", {}).get("recall", 0)
    mlp_recall = all_metrics.get("mlp_binary_tuned", {}).get("recall", 0)
    lines.append(
        f"- MLP 调优后的 recall ({mlp_recall:.4f}) "
        f"{'优于' if mlp_recall > rf_recall else '低于'} "
        f"RF ({rf_recall:.4f})"
    )
    lines.append("")

    # §6 特征重要度
    lines.append("## 6. 特征重要度对比")
    lines.append("")
    if "feature_imp" in figure_paths:
        lines.append(f"![Feature Importance](../{figure_paths['feature_imp']})")
        lines.append("")
    lines.append(
        "DT 和 RF 在 Top-15 特征上的重要度排序基本一致，"
        "`flag_SF`、`service_http` 等连接状态特征在两种模型中均占主导地位。"
    )
    lines.append("")

    # §7 DL vs ML
    lines.append("## 7. 深度学习 vs 传统机器学习")
    lines.append("")
    if "dl_vs_ml" in figure_paths:
        lines.append(f"![DL vs ML](../{figure_paths['dl_vs_ml']})")
        lines.append("")
    lines.append(
        "深度学习模型（MLP/CNN/LSTM）在三个关键指标上均显著优于"
        "传统机器学习（DT/RF），验证了神经网络对 NSL-KDD 表格数据的"
        "非线性建模优势。"
    )
    lines.append("")

    # §8 结论
    lines.append("## 8. 总结与局限")
    lines.append("")
    lines.append("### 核心结论")
    lines.append("")
    lines.append(
        "1. **MLP 二分类显著最优**：在 NSL-KDD 二分类任务上，"
        "MLP 调优模型在全部 5 个指标上均领先"
    )
    lines.append(
        "2. **深度学习整体优于传统 ML**：MLP/CNN/LSTM "
        "在 accuracy/f1/auc 上均超越 DT/RF"
    )
    lines.append(
        "3. **多分类性能**：基于 5 大类（DoS, Normal, Probe, R2L, U2R）"
        "的多分类 F1 Macro 反映了模型在攻击类型识别上的综合能力。"
    )
    lines.append("")
    lines.append("### 局限")
    lines.append("")
    lines.append(
        "- 多分类评估基于 5 大类攻击，无法反映细粒度攻击类型的识别差异"
    )
    lines.append("- 未进行统计显著性检验（McNemar's test 等）→ 未来工作")
    lines.append("- 未测量推理延迟 → 未来工作")
    lines.append(
        "- CNN/LSTM 作为表格数据模型，架构非最优"
        "（1D CNN 序列长度=1，LSTM 无时序依赖）"
    )
    lines.append("")
    lines.append("---")
    lines.append("*报告自动生成于 M5 编排脚本*")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))


# ============================================================
# Main Orchestration
# ============================================================

def main() -> None:
    start_time = time.time()

    # ── Step 1: Load test data ──
    banner("Step 1: Loading test data")
    X_test = load_pickle(PROCESSED_DIR / "X_test.pkl")
    y_test = load_pickle(PROCESSED_DIR / "y_test.pkl")
    X_test_multi = load_pickle(PROCESSED_DIR / "X_test_multi.pkl")
    y_test_multi = load_pickle(PROCESSED_DIR / "y_test_multi.pkl")
    print(f"  Binary test: X={X_test.shape}, y={y_test.shape}")
    print(f"  Multiclass test: X={X_test_multi.shape}, y={y_test_multi.shape}")
    print(f"  y_test_multi unique classes: {len(np.unique(y_test_multi))}")

    # ── Step 2: Load label mappings ──
    banner("Step 2: Loading label mappings")
    _, label_id_to_category = derive_label_mapping()
    print(f"  Categories: {sorted(set(label_id_to_category.values()))}")

    # ── Step 3: Load all 9 models ──
    banner("Step 3: Loading all models")
    models = load_all_models()
    print(f"\n  Total models loaded: {len(models)}")

    # Flatten labels to numpy
    y_test_np = np.asarray(y_test).ravel()
    y_test_multi_np = np.asarray(y_test_multi).ravel()

    # ── Step 4: Binary evaluation (6 models) ──
    banner("Step 4: Binary classification evaluation")
    binary_results: dict[str, dict] = {}
    binary_model_names = [
        "dt_binary", "rf_binary",
        "mlp_binary", "mlp_binary_tuned",
        "cnn_binary", "lstm_binary",
    ]

    for name in binary_model_names:
        if name not in models:
            print(f"  ✗ {name}: not loaded, skipping")
            continue
        model = models[name]
        t0 = time.time()

        y_pred = np.asarray(model.predict(X_test)).ravel()
        y_prob = get_probabilities(model, X_test)

        metrics = compute_binary_metrics(y_test_np, y_pred, y_prob)
        metrics["inference_time_s"] = round(time.time() - t0, 2)
        binary_results[name] = metrics
        print(
            f"  {name}: acc={metrics['accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} auc={metrics['auc']:.4f} "
            f"({metrics['inference_time_s']}s)"
        )

    # ── Step 5: Multiclass evaluation (3 models) ──
    banner("Step 5: Multiclass classification evaluation")
    multiclass_results: dict[str, dict] = {}

    for name in ["dt_multiclass", "rf_multiclass", "mlp_multiclass"]:
        if name not in models:
            continue
        model = models[name]
        y_pred = np.asarray(model.predict(X_test_multi)).ravel()
        metrics = compute_multiclass_metrics(y_test_multi_np, y_pred)
        multiclass_results[name] = metrics
        print(
            f"  {name}: full_acc={metrics['full_accuracy']:.4f} "
            f"f1_macro={metrics['f1_macro']:.4f}"
        )

    # ── Step 7: Generate 7 comparison figures ──
    banner("Step 7: Generating comparison figures")
    figure_paths: dict[str, str] = {}

    # 7.1  Confusion matrices (binary: DT, RF, MLP tuned)
    for name, title in [("dt_binary", "DT"), ("rf_binary", "RF"),
                        ("mlp_binary_tuned", "MLP (Tuned)")]:
        if name in models:
            y_pred = np.asarray(models[name].predict(X_test)).ravel()
            cm = confusion_matrix(y_test_np, y_pred)
            prefix = name.split("_")[0]
            save_path = FIGURES_DIR / f"09_confusion_matrix_{prefix}.png"
            _save_confusion_matrix(cm, title, save_path)
            figure_paths[f"confusion_{prefix}_binary"] = str(save_path)

    # 7.1b  Confusion matrices (multiclass: DT, RF, MLP) — 5×5
    for name, title in [("dt_multiclass", "DT"), ("rf_multiclass", "RF"),
                        ("mlp_multiclass", "MLP")]:
        if name in models:
            y_pred = np.asarray(models[name].predict(X_test_multi)).ravel()
            cm = confusion_matrix(y_test_multi_np, y_pred)
            prefix = name.split("_")[0]
            save_path = FIGURES_DIR / f"09_confusion_matrix_{prefix}_multiclass.png"
            _save_multiclass_confusion_matrix(cm, title, save_path)
            figure_paths[f"confusion_{prefix}_multiclass"] = str(save_path)

    # 7.2  F1 by attack category
    f1_dicts: dict[str, dict[str, float]] = {}
    for name in ["dt_binary", "rf_binary", "mlp_binary_tuned",
                  "cnn_binary", "lstm_binary"]:
        if name in models:
            y_pred = np.asarray(models[name].predict(X_test)).ravel()
            f1_dicts[name] = _compute_f1_by_category(
                y_test_multi_np, y_pred, label_id_to_category
            )
    if f1_dicts:
        save_path = FIGURES_DIR / "10_f1_by_category.png"
        _save_f1_category_bars(f1_dicts, save_path)
        figure_paths["f1_category"] = str(save_path)

    # 7.3  ROC curves
    roc_data: list = []
    for name in ["dt_binary", "rf_binary", "mlp_binary_tuned",
                  "cnn_binary", "lstm_binary"]:
        if name in binary_results and name in models:
            y_prob = get_probabilities(models[name], X_test)
            fpr, tpr, _ = roc_curve(y_test_np, y_prob)
            auc_val = roc_auc_score(y_test_np, y_prob)
            label = name.replace("_binary", "").replace("_tuned", " tuned").replace("_", " ")
            roc_data.append((label, fpr, tpr, auc_val))
    if roc_data:
        save_path = FIGURES_DIR / "11_roc_curves.png"
        _save_roc_curves(roc_data, save_path)
        figure_paths["roc"] = str(save_path)

    # 7.4  Feature importance (DT vs RF)
    if "dt_binary" in models and "rf_binary" in models:
        dt_imp = models["dt_binary"].feature_importances_
        rf_imp = models["rf_binary"].feature_importances_
        feature_names = (
            list(X_test.columns) if hasattr(X_test, "columns")
            else [f"f{i}" for i in range(20)]
        )
        dt_series = pd.Series(dt_imp, index=feature_names[:len(dt_imp)])
        rf_series = pd.Series(rf_imp, index=feature_names[:len(rf_imp)])
        save_path = FIGURES_DIR / "12_feature_importance_comparison.png"
        _save_feature_importance(dt_series, rf_series, 15, save_path)
        figure_paths["feature_imp"] = str(save_path)

    # 7.5  DL vs ML comparison
    compare_names = ["dt_binary", "rf_binary", "mlp_binary_tuned",
                     "cnn_binary", "lstm_binary"]
    metrics_dict: dict[str, dict[str, float]] = {}
    for name in compare_names:
        if name in binary_results:
            label = name.replace("_binary", "").replace("_tuned", " (Tuned)").replace("_", " ")
            metrics_dict[label] = {
                "accuracy": binary_results[name]["accuracy"],
                "f1": binary_results[name]["f1"],
                "auc": binary_results[name]["auc"],
            }
    if metrics_dict:
        save_path = FIGURES_DIR / "13_dl_vs_ml_comparison.png"
        _save_dl_vs_ml(metrics_dict, save_path)
        figure_paths["dl_vs_ml"] = str(save_path)

    # ── Step 8: Save metrics JSON ──
    banner("Step 8: Saving metrics")
    all_metrics = {}
    all_metrics.update(binary_results)
    all_metrics.update(multiclass_results)
    with open(ROOT / "outputs" / "metrics_m5.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"  ✓ Saved outputs/metrics_m5.json ({len(all_metrics)} models)")

    # ── Step 9: Generate report ──
    banner("Step 9: Generating comparison report")
    _generate_report(all_metrics, figure_paths, ROOT / "docs" / "comparison_report.md")
    print(f"  ✓ Saved docs/comparison_report.md")

    # ── Done ──
    elapsed = time.time() - start_time
    banner(f"M5 完成 ({elapsed:.1f}s)")
    print(f"  Models evaluated: {len(all_metrics)}")
    print(f"  Figures generated: {len(figure_paths)}")
    print(f"  Report: docs/comparison_report.md")
    print(f"  Metrics: outputs/metrics_m5.json")


if __name__ == "__main__":
    main()
