#!/usr/bin/env python
"""一次性补丁：补 metrics_m5.json 中的 mlp_multiclass 数据 + 生成缺失的混淆矩阵图。

根因：scripts/evaluate_m5.py 的 MODEL_CONFIGS 中 mlp_multiclass_best.pt 的 kwargs 配置错误
（input_dim=20/hidden_dims=(128,128)），但实际模型是用 53 维 + (128, 64) 训练的。

本脚本直接以正确 kwargs 评估 MLP 多分类，把结果合并到 metrics_m5.json，并生成
09_confusion_matrix_mlp_multiclass.png。
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.persistence import load_pickle
from src.models.persistence import load_torch_model
from src.models.mlp import MLPClassifier

MODELS_DIR = ROOT / "outputs" / "models"
PROCESSED_DIR = ROOT / "outputs" / "processed"
FIGURES_DIR = ROOT / "outputs" / "figures"
METRICS_PATH = ROOT / "outputs" / "metrics_m5.json"
FIG_PATH = FIGURES_DIR / "09_confusion_matrix_mlp_multiclass.png"


def compute_multiclass_metrics(y_true, y_pred):
    full_acc = float(accuracy_score(y_true, y_pred))
    f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_class_f1 = {}
    for cls in sorted(set(np.unique(y_true))):
        cls = int(cls)
        per_class_f1[str(cls)] = float(
            f1_score((y_true == cls).astype(int), (y_pred == cls).astype(int), zero_division=0)
        )
    return {
        "accuracy": full_acc,
        "f1_macro": f1_m,
        "full_accuracy": full_acc,
        "per_class_f1": per_class_f1,
    }


def save_confusion_matrix(cm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["DoS", "Normal", "Probe", "R2L", "U2R"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(5)); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("MLP - Confusion Matrix (5-Class)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_PATH, dpi=80)
    plt.close()


def main():
    print("=" * 60)
    print("  Patching mlp_multiclass into metrics_m5.json")
    print("=" * 60)

    X_test_multi = load_pickle(PROCESSED_DIR / "X_test_multi.pkl")
    y_test_multi = load_pickle(PROCESSED_DIR / "y_test_multi.pkl")
    print(f"  X_test_multi: {X_test_multi.shape}")

    model = load_torch_model(
        MLPClassifier,
        MODELS_DIR / "mlp_multiclass_best.pt",
        input_dim=53, output_dim=5, hidden_dims=(128, 64),
    )
    model.eval()

    X_arr = np.asarray(X_test_multi).copy()
    with torch.no_grad():
        logits = model(torch.as_tensor(X_arr, dtype=torch.float32))
        y_pred = logits.argmax(dim=1).numpy()

    metrics = compute_multiclass_metrics(y_test_multi, y_pred)
    print(f"  accuracy:   {metrics['accuracy']:.4f}")
    print(f"  f1_macro:   {metrics['f1_macro']:.4f}")
    print(f"  per_class_f1: {metrics['per_class_f1']}")

    cm = confusion_matrix(y_test_multi, y_pred)
    save_confusion_matrix(cm)
    print(f"  ✓ Saved {FIG_PATH.name}")

    with open(METRICS_PATH, "r") as f:
        all_metrics = json.load(f)
    all_metrics["mlp_multiclass"] = metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"  ✓ Updated {METRICS_PATH.name} ({len(all_metrics)} keys)")


if __name__ == "__main__":
    main()