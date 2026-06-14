"""评估指标计算函数（纯函数，无模型依赖）。"""

from __future__ import annotations

from typing import Hashable

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_prob: ArrayLike,
) -> dict[str, float]:
    """计算二分类任务的 5 项核心指标。

    Parameters
    ----------
    y_true : ArrayLike
        真实标签（0/1）。
    y_pred : ArrayLike
        预测标签（0/1）。
    y_prob : ArrayLike
        预测为正类的概率（用于 AUC）。

    Returns
    -------
    dict[str, float]
        包含 ``accuracy``、``precision``、``recall``、``f1``、``auc`` 五个键的字典。
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
    }


def compute_multiclass_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    unseen_ids: tuple = (),
) -> dict[str, float]:
    """计算多分类任务指标，包含对未见类别的过滤评估。

    Parameters
    ----------
    y_true : ArrayLike
        真实标签。
    y_pred : ArrayLike
        预测标签。
    unseen_ids : tuple, default ()
        测试集中出现但训练时未见过的标签 ID。这些样本在
        ``known_class_accuracy`` 中会被剔除。

    Returns
    -------
    dict[str, float]
        包含 ``accuracy``（全量）、``known_class_accuracy``、``f1_macro`` 的字典。
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    full_accuracy = float(accuracy_score(y_true_arr, y_pred_arr))

    if unseen_ids:
        seen_mask = ~np.isin(y_true_arr, list(unseen_ids))
        known_accuracy = float(
            accuracy_score(y_true_arr[seen_mask], y_pred_arr[seen_mask])
        )
    else:
        known_accuracy = full_accuracy

    f1_macro = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

    return {
        "accuracy": full_accuracy,
        "known_class_accuracy": known_accuracy,
        "f1_macro": f1_macro,
    }


def compute_f1_by_category(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    label_to_category_map: dict[Hashable, str],
) -> dict[str, float]:
    """按攻击类别分组计算 F1 分数。

    将细粒度标签映射到高层类别（如 DoS / Probe / R2L / U2R / Normal），
    对每个类别构造二分类掩码并计算 F1。

    Parameters
    ----------
    y_true : ArrayLike
        真实标签（细粒度）。
    y_pred : ArrayLike
        预测标签（细粒度）。
    label_to_category_map : dict
        标签 ID 到类别名的映射，例如 ``{0: "DoS", 1: "Probe"}``。

    Returns
    -------
    dict[str, float]
        类别名到 F1 分数的映射。
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    y_true_cat = np.array(
        [label_to_category_map[int(label)] for label in y_true_arr]
    )
    y_pred_cat = np.array(
        [label_to_category_map.get(int(label), "unknown") for label in y_pred_arr]
    )

    categories = sorted(set(label_to_category_map.values()))
    return {
        category: float(
            f1_score(
                (y_true_cat == category).astype(int),
                (y_pred_cat == category).astype(int),
                zero_division=0,
            )
        )
        for category in categories
    }
