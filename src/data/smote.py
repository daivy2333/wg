"""SMOTE 样本不均衡处理（M4 任务 4.6）

针对 M3 多分类失败痛点（U2R 仅 52 条训练样本），仅对训练集中的小样本类
（U2R / R2L）做 SMOTE 过采样，DoS / Probe / normal 等样本充足的类保持原样。

负责人：C同学（神经网络方向）
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


# NSL-KDD 训练集中标签 id → 攻击名称的映射（需要根据 M2 实际标签编码确定）
# 此处使用 M2 标准映射：前 22 类是训练中存在的 22 种攻击，类 22 是 "no_label" / 未知
# 实际 U2R 和 R2L 包含多种攻击类型，需要根据 y_train 实际分布确定
DEFAULT_MINORITY_TARGETS = ("U2R", "R2L")  # 仅文档化，调用时需要传具体 label id


def apply_smote_to_minority_classes(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    target_classes: Sequence[int],
    sampling_strategy: str | dict[int, int] = "auto",
    k_neighbors: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """对指定小样本类做 SMOTE 过采样。

    Args:
        X_train: 训练特征 (N, 20)
        y_train: 训练标签 (N,)
        target_classes: 需要过采样的标签 id 列表（如 [u2r_label_id, r2l_label_id_1, ...]）
            注意：因为 R2L/U2R 在数据中可能对应多个 label id（不同攻击类型），
            需要根据 M2 标签编码确定具体 id。
        sampling_strategy: 'auto' 表示提升到多数类样本数；或 dict 指定具体目标数
        k_neighbors: SMOTE 的 k 近邻数（U2R 仅 52 条，建议 k_neighbors=3 避免内存错误）
        random_state: 随机种子

    Returns:
        (X_resampled, y_resampled) — SMOTE 后的训练集
    """
    if not IMBLEARN_AVAILABLE:
        raise ImportError(
            "imbalanced-learn 未安装。请先执行 `pip install imbalanced-learn`（任务 1.1）"
        )

    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values

    y_train = np.asarray(y_train)
    target_classes_list = list(target_classes)

    # 验证 target_classes 中每个类至少有 k_neighbors+1 个样本（SMOTE 要求）
    for cls in target_classes_list:
        n_cls = int((y_train == cls).sum())
        if n_cls < k_neighbors + 1:
            raise ValueError(
                f"类 {cls} 仅 {n_cls} 个样本，少于 SMOTE 要求的 k_neighbors+1={k_neighbors + 1}。"
                f"请降低 k_neighbors（如 3）或跳过此类。"
            )

    # 构造 sampling_strategy：只对 target_classes 过采样
    if sampling_strategy == "auto":
        # 'auto' 默认提升到多数类样本数
        class_counts = np.bincount(y_train)
        majority_count = int(class_counts.max())
        sampling_strategy = {cls: majority_count for cls in target_classes_list}

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def get_minority_class_ids(
    y_train: np.ndarray | pd.Series,
    threshold: int = 1000,
) -> list[int]:
    """识别训练集中的小样本类（样本数 < threshold）。

    默认 threshold=1000：
      - U2R (52 条) / R2L (995 条) 会被识别
      - normal (67343) / DoS (45927) / Probe (11656) 不会
    """
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    y_train = np.asarray(y_train)
    class_counts = np.bincount(y_train)
    return [int(cls) for cls, count in enumerate(class_counts) if 0 < count < threshold]


def compare_with_without_smote(
    train_fn,
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    target_classes: Sequence[int],
    k_neighbors: int = 3,
) -> dict:
    """对比 SMOTE 前后的多分类 per-class recall。

    Args:
        train_fn: 训练函数签名 `(X_train, y_train, X_val, y_val) -> (model, metrics)`
        target_classes: 需要过采样的类
        k_neighbors: SMOTE k_neighbors

    Returns:
        {
            "baseline": {"per_class_recall": {0: 0.xx, ...}, "accuracy": 0.xx},
            "smote": {"per_class_recall": {0: 0.xx, ...}, "accuracy": 0.xx},
            "target_classes": [list of target class ids]
        }
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import recall_score, accuracy_score

    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    # 划分训练/验证集
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # 1. Baseline 训练
    model_baseline, _ = train_fn(X_tr, y_tr, X_val, y_val)
    preds_baseline = model_baseline.predict(X_test)
    per_class_recall_baseline = recall_score(
        y_test, preds_baseline, average=None, labels=list(range(23)), zero_division=0
    )
    acc_baseline = float(accuracy_score(y_test, preds_baseline))

    # 2. SMOTE 训练
    X_smote, y_smote = apply_smote_to_minority_classes(
        X_tr, y_tr, target_classes=target_classes, k_neighbors=k_neighbors
    )
    model_smote, _ = train_fn(X_smote, y_smote, X_val, y_val)
    preds_smote = model_smote.predict(X_test)
    per_class_recall_smote = recall_score(
        y_test, preds_smote, average=None, labels=list(range(23)), zero_division=0
    )
    acc_smote = float(accuracy_score(y_test, preds_smote))

    return {
        "baseline": {
            "per_class_recall": {int(i): float(per_class_recall_baseline[i]) for i in range(23)},
            "accuracy": acc_baseline,
        },
        "smote": {
            "per_class_recall": {int(i): float(per_class_recall_smote[i]) for i in range(23)},
            "accuracy": acc_smote,
        },
        "target_classes": list(target_classes),
    }


def save_smote_data(
    X_smote: np.ndarray, y_smote: np.ndarray, output_path: str
) -> None:
    """保存 SMOTE 生成的合成训练集到磁盘。"""
    from src.data.persistence import save_pickle

    save_pickle({"X": X_smote, "y": y_smote}, output_path)


def load_smote_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    """加载 SMOTE 数据。"""
    from src.data.persistence import load_pickle

    data = load_pickle(path)
    return data["X"], data["y"]
