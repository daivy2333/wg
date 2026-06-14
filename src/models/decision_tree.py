"""决策树训练模块（M3 任务 3.1-3.3）

基线 + GridSearchCV 调优 + 5 指标评估
"""
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


# 默认参数（M3 BDD 默认假设）
DEFAULT_RANDOM_STATE: int = 42
DEFAULT_CV: int = 5

# DT 基线验收阈值（spec R1/R2）
DT_BINARY_THRESHOLD: float = 0.80
DT_MULTICLASS_THRESHOLD: float = 0.75

# GridSearchCV 网格（30 组合）
DT_PARAM_GRID: dict = {
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"],
}


def train_dt_binary(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> DecisionTreeClassifier:
    """DT 二分类基线训练。

    Args:
        X_train: 训练特征
        y_train: 训练标签（0=normal, 1=anomaly）
        random_state: 随机种子

    Returns:
        训练好的 DecisionTreeClassifier，class_weight='balanced'
    """
    model = DecisionTreeClassifier(
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def train_dt_multiclass(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> DecisionTreeClassifier:
    """DT 多分类基线训练。

    Args:
        X_train: 训练特征
        y_train: 训练标签（23 类：0-22）
        random_state: 随机种子

    Returns:
        训练好的 DecisionTreeClassifier，class_weight='balanced'
    """
    model = DecisionTreeClassifier(
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: Literal["binary", "multiclass"] = "binary",
) -> dict[str, float]:
    """评估模型：5 指标（accuracy/precision/recall/f1/auc）。

    Args:
        model: 训练好的 sklearn 模型
        X_test: 测试特征
        y_test: 测试标签
        task: 'binary' 或 'multiclass'

    Returns:
        dict 含指标名 → 值
    """
    y_pred = model.predict(X_test)
    metrics: dict[str, float] = {}

    if task == "binary":
        avg = "binary"
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["precision"] = float(precision_score(y_test, y_pred, average=avg, zero_division=0))
        metrics["recall"] = float(recall_score(y_test, y_pred, average=avg, zero_division=0))
        metrics["f1"] = float(f1_score(y_test, y_pred, average=avg, zero_division=0))
        # AUC 需要 predict_proba
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            try:
                metrics["auc"] = float(roc_auc_score(y_test, y_proba))
            except ValueError:
                metrics["auc"] = float("nan")
    else:  # multiclass
        avg = "macro"
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["precision_macro"] = float(precision_score(y_test, y_pred, average=avg, zero_division=0))
        metrics["recall_macro"] = float(recall_score(y_test, y_pred, average=avg, zero_division=0))
        metrics["f1_macro"] = float(f1_score(y_test, y_pred, average=avg, zero_division=0))

    return metrics


def grid_search_dt(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task: Literal["binary", "multiclass"] = "binary",
    cv: int = DEFAULT_CV,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[DecisionTreeClassifier, dict, float]:
    """DT 超参数网格搜索（30 组合 × cv 折）。

    Args:
        X_train: 训练特征
        y_train: 训练标签
        task: 'binary' 或 'multiclass'
        cv: 交叉验证折数
        random_state: 随机种子

    Returns:
        (best_model, best_params, best_score)
    """
    scoring = "f1" if task == "binary" else "f1_macro"
    base = DecisionTreeClassifier(
        random_state=random_state,
        class_weight="balanced",
    )
    grid = GridSearchCV(
        base,
        param_grid=DT_PARAM_GRID,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, float(grid.best_score_)