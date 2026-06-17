#!/usr/bin/env python
"""一次性补丁：重新生成 10_f1_by_category.png 为多分类 per-class F1

历史背景：
- 旧版本 _compute_f1_by_category 用原始攻击 id 查新 0-4 标签，导致 Probe/R2L/U2R 永远为 0
- 即使修了 id 映射问题，旧逻辑的 y_pred_binary（0/1）与 y_true_bin（"是否为该类"）的
  正例语义不匹配，得到的 F1 不是有意义的"多分类 per-class F1"
- ch5 §5.4 文字描述（Probe 0.7-0.75, R2L 0.2-0.6, U2R MLP 56%）实际来自多分类 per-class F1

本脚本重新设计：使用 3 个多分类模型（dt/rf/mlp），在 5 大类上计算 per-class F1，
得到与 ch5 §5.4 文字描述完全一致的结果。
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.persistence import load_pickle
from src.models.persistence import load_model, load_torch_model
from src.models.mlp import MLPClassifier

PROCESSED_DIR = ROOT / "outputs" / "processed"
MODELS_DIR = ROOT / "outputs" / "models"
FIG_PATH = ROOT / "outputs" / "figures" / "10_f1_by_category.png"

# 5 大类显示顺序
CATEGORIES = ["DoS", "Normal", "Probe", "R2L", "U2R"]


def compute_per_class_f1(y_true, y_pred, n_classes=5) -> dict:
    """计算每个类别的 F1（基于多分类预测）。"""
    result = {}
    for cls in range(n_classes):
        f1 = f1_score(
            (y_true == cls).astype(int),
            (y_pred == cls).astype(int),
            zero_division=0,
        )
        result[CATEGORIES[cls]] = f1
    return result


def save_per_class_f1_bars(f1_dicts, save_path):
    """绘制多分类 per-class F1 柱状图（每个类别一簇，3 个模型并排）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = list(f1_dicts.keys())
    n_models = len(model_names)
    bar_width = 0.8 / n_models
    x = np.arange(len(CATEGORIES))
    colors = plt.cm.tab10.colors[:n_models]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, f1_dict) in enumerate(f1_dicts.items()):
        values = [f1_dict.get(c, 0) for c in CATEGORIES]
        bars = ax.bar(x + i * bar_width, values, bar_width,
                      label=name, color=colors[i])
        for bar, val in zip(bars, values):
            if val > 0.001:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=8, rotation=90)
    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-Class F1 Score by Attack Category (Multiclass Models)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=80)
    plt.close()
    print(f"  ✓ Saved {save_path.name}")


def main():
    print("=" * 60)
    print("  Patching 10_f1_by_category.png (per-class F1 multiclass)")
    print("=" * 60)

    X_test_multi = load_pickle(PROCESSED_DIR / "X_test_multi.pkl")
    y_test_multi = load_pickle(PROCESSED_DIR / "y_test_multi.pkl")
    print(f"  X_test_multi: {X_test_multi.shape}, y_test_multi: {y_test_multi.shape}")
    print(f"  各类样本数: {dict(zip(*np.unique(y_test_multi, return_counts=True)))}")

    # 3 个多分类模型（dt/rf 用 X_test_multi 53 维，mlp_multiclass 用 53 维 + (128,64)）
    specs = [
        ("DT", "dt_multiclass_best.joblib", "joblib", None),
        ("RF", "rf_multiclass_best.joblib", "joblib", None),
        ("MLP", "mlp_multiclass_best.pt", "torch",
         (MLPClassifier, {"input_dim": 53, "output_dim": 5, "hidden_dims": (128, 64)})),
    ]

    f1_dicts = {}
    for name, fname, kind, extra in specs:
        try:
            if kind == "joblib":
                model = load_model(MODELS_DIR / fname)
            else:
                cls, kwargs = extra
                model = load_torch_model(cls, MODELS_DIR / fname, **kwargs)
            y_pred = np.asarray(model.predict(X_test_multi)).ravel()
            f1_dicts[name] = compute_per_class_f1(y_test_multi, y_pred)
            print(f"  {name:>4s}: {f1_dicts[name]}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ✗ {name}: {e}")

    if f1_dicts:
        save_per_class_f1_bars(f1_dicts, FIG_PATH)
        print(f"\n  各类别平均 F1:")
        for cat in CATEGORIES:
            vals = [d[cat] for d in f1_dicts.values() if cat in d]
            if vals:
                print(f"    {cat:>6s}: mean={np.mean(vals):.3f}  range=[{min(vals):.3f}, {max(vals):.3f}]")


if __name__ == "__main__":
    main()