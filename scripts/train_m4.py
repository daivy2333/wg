"""M4 神经网络模型训练编排脚本

端到端流程：
  1. 加载 M2 产出数据
  2. 训练 MLP 二分类基线
  3. 训练 MLP 二分类调优（网格搜索）
  4. 训练 MLP 多分类（含全量 + 已知类 acc 双数字）
  5. 训练 CNN 二分类
  6. 训练 LSTM 二分类
  7. SMOTE 仅 U2R/R2L → 训练 MLP 多分类（对比实验）
  8. 持久化所有模型（torch.save state_dict）
  9. 输出 JSON 指标汇总（供报告使用）

运行：
  python scripts/train_m4.py [--epochs N] [--tune] [--smote]

负责人：C同学（神经网络方向）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 必须在 import torch 之前设置
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# 添加项目根到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.data.persistence import load_pickle
from src.models.cnn import train_cnn_binary
from src.models.lstm import train_lstm_binary
from src.models.mlp import (
    UNSEEN_ATTACKS_IN_TEST,
    compute_known_class_accuracy,
    grid_search_mlp,
    train_mlp_binary,
    train_mlp_multiclass,
)
from src.models.persistence import save_best_nn_models


# ============================================================
# 智能公式（沿用模式-004 + 踩坑-003）
# ============================================================


def get_optimal_threads() -> int:
    """生产环境线程数（智能公式）。"""
    return min(4, max(1, (os.cpu_count() or 1) // 8))


# ============================================================
# 数据加载
# ============================================================


def load_data(data_dir: str = "outputs/processed") -> dict:
    """加载 M2 产出的 pickle 数据。"""
    data_dir = Path(data_dir)
    return {
        "X_train": load_pickle(data_dir / "X_train.pkl"),
        "X_test": load_pickle(data_dir / "X_test.pkl"),
        "y_train_binary": load_pickle(data_dir / "y_train.pkl"),
        "y_test_binary": load_pickle(data_dir / "y_test.pkl"),
        "y_train_multi": load_pickle(data_dir / "y_train_multi.pkl"),
        "y_test_multi": load_pickle(data_dir / "y_test_multi.pkl"),
    }


def train_val_split(
    X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """训练/验证集划分（8:2）。"""
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=val_ratio, random_state=random_state, stratify=y)


# ============================================================
# 指标计算
# ============================================================


def multiclass_metrics_with_known_class(
    model, X_test: np.ndarray, y_test: np.ndarray, num_classes: int = 5
) -> dict:
    """多分类全量 + 已知类 acc 双数字。

    注：训练集中没有的 16 种攻击（M2 EDA 识别），其 label id 在训练 y 中根本不存在。
    测试集中如果出现这些攻击的 label id（被映射到最相似训练类），它们被预测正确
    的概率低，导致全量 acc 偏低。

    已知类 acc：剔除测试集中所有出现频率极低（< 5 次）的类后计算。
    """
    from sklearn.metrics import accuracy_score, f1_score

    preds = model.predict(X_test)
    full_accuracy = float(accuracy_score(y_test, preds))
    f1_macro = float(f1_score(y_test, preds, average="macro", zero_division=0))

    # 已知类：剔除测试集中样本数 < 5 的类（这些类在训练中可能也很稀少）
    from collections import Counter

    class_counts = Counter(y_test.tolist())
    rare_classes = {cls for cls, count in class_counts.items() if count < 5}
    if rare_classes:
        mask = ~np.isin(y_test, list(rare_classes))
        known_accuracy = float(accuracy_score(y_test[mask], preds[mask]))
    else:
        known_accuracy = full_accuracy

    return {
        "full_accuracy": full_accuracy,
        "known_class_accuracy": known_accuracy,
        "f1_macro": f1_macro,
        "num_test_samples": len(y_test),
        "rare_classes_filtered": sorted(rare_classes),
    }


# ============================================================
# 主流程
# ============================================================


def main(epochs: int = 50, do_tune: bool = True, do_smote: bool = True, output_dir: str = "outputs/models") -> None:
    """M4 端到端训练。

    Args:
        epochs: 默认训练 epoch 数
        do_tune: 是否运行网格搜索调优
        do_smote: 是否运行 SMOTE 对比实验
        output_dir: 模型输出目录
    """
    n_threads = get_optimal_threads()
    torch.set_num_threads(n_threads)
    print(f"[setup] PyTorch threads: {n_threads} (cpu_count={os.cpu_count()})")

    print("\n[1/8] 加载数据...")
    data = load_data()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train_bin, y_test_bin = data["y_train_binary"], data["y_test_binary"]
    y_train_multi, y_test_multi = data["y_train_multi"], data["y_test_multi"]
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  binary label distribution: train {np.bincount(y_train_bin)}, test {np.bincount(y_test_bin)}")
    print(f"  multi label classes: train {len(np.unique(y_train_multi))}, test {len(np.unique(y_test_multi))}")

    # 训练/验证划分
    X_tr, X_val, y_tr_bin, y_val_bin = train_val_split(X_train, y_train_bin)
    _, _, y_tr_multi, y_val_multi = train_val_split(X_train, y_train_multi)

    metrics_all: dict[str, dict] = {}

    # ----- 2. MLP 二分类基线 -----
    print(f"\n[2/8] MLP 二分类基线（epochs={epochs}）...")
    t0 = time.time()
    result_mlp_bin = train_mlp_binary(
        X_tr, y_tr_bin, X_val, y_val_bin,
        epochs=epochs, batch_size=256, verbose=False,
    )
    metrics_all["mlp_binary_baseline"] = result_mlp_bin.val_metrics
    print(f"  val_metrics: {result_mlp_bin.val_metrics}")
    print(f"  耗时: {time.time() - t0:.1f}s")

    # ----- 3. MLP 调优（可选） -----
    if do_tune:
        print(f"\n[3/8] MLP 网格搜索调优（epochs={epochs // 2} 加速）...")
        t0 = time.time()
        try:
            best_model_tuned, best_params, summary = grid_search_mlp(
                X_tr, y_tr_bin, X_val, y_val_bin,
                epochs=epochs // 2, cv_folds=3, verbose=True,
            )
            # 评估调优后的最佳模型
            from src.models.mlp import _evaluate_binary
            from src.models.mlp import make_dataloader
            val_loader = make_dataloader(X_val, y_val_bin, batch_size=256, shuffle=False)
            device = next(best_model_tuned.parameters()).device
            tuned_metrics = _evaluate_binary(best_model_tuned, val_loader, device)
            metrics_all["mlp_binary_tuned"] = tuned_metrics
            metrics_all["mlp_binary_tuned"]["best_params"] = best_params
            metrics_all["mlp_binary_tuned"]["cv_f1"] = summary[
                max(summary, key=lambda k: summary[k]["val_f1"])
            ]["cv_f1"]
            print(f"  best_params: {best_params}")
            print(f"  cv_f1={metrics_all['mlp_binary_tuned']['cv_f1']:.4f}, val_f1={tuned_metrics['f1']:.4f}")
        except Exception as e:
            print(f"  [WARN] 调优失败: {e}，跳过")
            metrics_all["mlp_binary_tuned"] = {"error": str(e)}
        print(f"  耗时: {time.time() - t0:.1f}s")
    else:
        print("\n[3/8] 跳过调优（--no-tune）")
        best_model_tuned = None

    # ----- 4. MLP 多分类 -----
    print(f"\n[4/8] MLP 多分类（epochs={epochs}）...")
    t0 = time.time()
    result_mlp_multi = train_mlp_multiclass(
        X_tr, y_tr_multi, X_val, y_val_multi,
        num_classes=5, epochs=epochs, verbose=False,
    )
    multi_metrics = multiclass_metrics_with_known_class(
        result_mlp_multi.model, X_test, y_test_multi, num_classes=5
    )
    metrics_all["mlp_multiclass"] = {**result_mlp_multi.val_metrics, **multi_metrics}
    print(f"  full_accuracy: {multi_metrics['full_accuracy']:.4f}")
    print(f"  known_class_accuracy: {multi_metrics['known_class_accuracy']:.4f}")
    print(f"  f1_macro: {multi_metrics['f1_macro']:.4f}")
    print(f"  耗时: {time.time() - t0:.1f}s")

    # ----- 5. CNN 二分类 -----
    print(f"\n[5/8] CNN 二分类（epochs={epochs}）...")
    t0 = time.time()
    result_cnn = train_cnn_binary(
        X_tr, y_tr_bin, X_val, y_val_bin, epochs=epochs, verbose=False,
    )
    metrics_all["cnn_binary"] = result_cnn.val_metrics
    print(f"  val_metrics: {result_cnn.val_metrics}")
    print(f"  耗时: {time.time() - t0:.1f}s")

    # ----- 6. LSTM 二分类 -----
    print(f"\n[6/8] LSTM 二分类（epochs={epochs}）...")
    t0 = time.time()
    result_lstm = train_lstm_binary(
        X_tr, y_tr_bin, X_val, y_val_bin, epochs=epochs, verbose=False,
    )
    metrics_all["lstm_binary"] = result_lstm.val_metrics
    print(f"  val_metrics: {result_lstm.val_metrics}")
    print(f"  耗时: {time.time() - t0:.1f}s")

    # ----- 7. SMOTE 对比（可选） -----
    mlp_smote = None
    if do_smote:
        print(f"\n[7/8] SMOTE 对比实验（仅 U2R/R2L）...")
        t0 = time.time()
        try:
            from src.data.smote import apply_smote_to_minority_classes, get_minority_class_ids
            minority_ids_all = get_minority_class_ids(y_tr_multi, threshold=1000)
            # 过滤掉样本数 < 4 的类（SMOTE 最低要求 k_neighbors=1 即需 2 样本；用 k_neighbors=2 更稳）
            minority_ids = [cls for cls in minority_ids_all if (y_tr_multi == cls).sum() >= 3]
            skipped_ids = [cls for cls in minority_ids_all if cls not in minority_ids]
            print(f"  识别小样本类: {minority_ids_all}（样本数 < 1000）")
            print(f"  过滤 < 3 样本的类: {skipped_ids}")
            print(f"  实际 SMOTE 目标: {minority_ids}")
            if minority_ids:
                k_neighbors = 2  # 最低 2，避免大 k_neighbors 导致内存错误
                X_smote, y_smote = apply_smote_to_minority_classes(
                    X_tr, y_tr_multi, target_classes=minority_ids, k_neighbors=k_neighbors
                )
                print(f"  SMOTE 后样本: {X_smote.shape[0]}（原 {X_tr.shape[0]}）")
                result_mlp_smote = train_mlp_multiclass(
                    X_smote, y_smote, X_val, y_val_multi,
                    num_classes=5, epochs=epochs, verbose=False,
                )
                smote_metrics = multiclass_metrics_with_known_class(
                    result_mlp_smote.model, X_test, y_test_multi, num_classes=5
                )
                # 提取目标类的 recall
                from sklearn.metrics import recall_score
                preds_smote = result_mlp_smote.model.predict(X_test)
                per_class_recall = recall_score(
                    y_test_multi, preds_smote, average=None,
                    labels=list(range(5)), zero_division=0
                )
                smote_metrics["per_class_recall"] = {
                    int(i): float(per_class_recall[i]) for i in range(5)
                }
                smote_metrics["minority_classes"] = minority_ids
                smote_metrics["skipped_classes"] = skipped_ids
                metrics_all["mlp_multiclass_smote"] = smote_metrics
                print(f"  SMOTE 后 full_accuracy: {smote_metrics['full_accuracy']:.4f}")
                print(f"  SMOTE 后 known_class_accuracy: {smote_metrics['known_class_accuracy']:.4f}")
                print(f"  小样本类 recall:")
                for cls in minority_ids:
                    print(f"    类 {cls}: {per_class_recall[cls]:.4f}")
                mlp_smote = result_mlp_smote.model
            else:
                print("  [WARN] 未识别到小样本类（过滤后为空），跳过 SMOTE")
                metrics_all["mlp_multiclass_smote"] = {"skipped": "no_minority_classes"}
        except Exception as e:
            print(f"  [WARN] SMOTE 失败: {e}，跳过")
            metrics_all["mlp_multiclass_smote"] = {"error": str(e)}
        print(f"  耗时: {time.time() - t0:.1f}s")
    else:
        print("\n[7/8] 跳过 SMOTE（--no-smote）")

    # ----- 8. 持久化 -----
    print(f"\n[8/8] 模型持久化 → {output_dir}...")
    saved_paths = save_best_nn_models(
        output_dir,
        mlp_binary=result_mlp_bin.model,
        mlp_binary_tuned=best_model_tuned,
        mlp_multiclass=result_mlp_multi.model,
        cnn_binary=result_cnn.model,
        lstm_binary=result_lstm.model,
        mlp_multiclass_smote=mlp_smote,
    )
    print(f"  保存 {len(saved_paths)} 个 .pt 文件")

    # ----- 写出指标 JSON -----
    metrics_path = Path(output_dir).parent / "metrics_m4.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_all, f, indent=2, default=str)
    print(f"\n[done] 指标已保存到 {metrics_path}")
    print(f"[done] 模型已保存到 {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M4 神经网络训练编排")
    parser.add_argument("--epochs", type=int, default=50, help="训练 epoch 数（默认 50）")
    parser.add_argument("--tune", action="store_true", default=True, help="运行网格搜索调优（默认 True）")
    parser.add_argument("--no-tune", dest="tune", action="store_false", help="跳过调优")
    parser.add_argument("--smote", action="store_true", default=True, help="运行 SMOTE 对比（默认 True）")
    parser.add_argument("--no-smote", dest="smote", action="store_false", help="跳过 SMOTE")
    parser.add_argument("--output-dir", type=str, default="outputs/models", help="模型输出目录")
    args = parser.parse_args()
    main(epochs=args.epochs, do_tune=args.tune, do_smote=args.smote, output_dir=args.output_dir)
