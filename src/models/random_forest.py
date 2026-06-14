"""随机森林训练模块（M3 任务 3.4-3.7）

基线 + GridSearchCV 调优 + 特征重要度 + 4 大攻击类别 F1
"""
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# 默认参数
DEFAULT_RANDOM_STATE: int = 42
DEFAULT_CV: int = 5
DEFAULT_N_ESTIMATORS: int = 100  # 基线树数
DEFAULT_N_JOBS: int = 1  # 默认单进程（WSL/sklearn 兼容，避免 OOM）

# RF 基线验收阈值（spec R5/R6）
RF_BINARY_THRESHOLD: float = 0.85
RF_MULTICLASS_THRESHOLD: float = 0.78

# GridSearchCV 网格（24 组合）
RF_PARAM_GRID: dict = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 15, 20, None],
    "min_samples_split": [2, 5],
}


def train_rf_binary(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = DEFAULT_N_JOBS,
) -> RandomForestClassifier:
    """RF 二分类基线训练。

    Args:
        X_train: 训练特征
        y_train: 训练标签
        n_estimators: 树数量（基线默认 100）
        random_state: 随机种子
        n_jobs: 并行度（默认 1；生产环境可传 -1 加速）

    Returns:
        训练好的 RandomForestClassifier，class_weight='balanced'
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    return model


def train_rf_multiclass(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = DEFAULT_N_JOBS,
) -> RandomForestClassifier:
    """RF 多分类基线训练。

    Args:
        X_train: 训练特征
        y_train: 训练标签（23 类）
        n_estimators: 树数量
        random_state: 随机种子
        n_jobs: 并行度（默认 1）

    Returns:
        训练好的 RandomForestClassifier
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    return model


def grid_search_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task: Literal["binary", "multiclass"] = "binary",
    cv: int = DEFAULT_CV,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = DEFAULT_N_JOBS,
) -> Tuple[RandomForestClassifier, dict, float]:
    """RF 超参数网格搜索（24 组合 × cv 折）。

    Args:
        X_train: 训练特征
        y_train: 训练标签
        task: 'binary' 或 'multiclass'
        cv: 交叉验证折数
        random_state: 随机种子
        n_jobs: 并行度（默认 1；生产可传 -1）

    Returns:
        (best_model, best_params, best_score)
    """
    scoring = "f1" if task == "binary" else "f1_macro"
    base = RandomForestClassifier(
        random_state=random_state,
        class_weight="balanced",
        n_jobs=n_jobs,
    )
    grid = GridSearchCV(
        base,
        param_grid=RF_PARAM_GRID,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, float(grid.best_score_)


def get_top_k_features(model: RandomForestClassifier, k: int = 20) -> pd.Series:
    """提取 Top-K 重要特征。

    Args:
        model: 已 fit 的 RF 模型
        k: 保留数量

    Returns:
        pd.Series，索引为列名（feature），按 feature_importances_ 降序
    """
    importance = pd.Series(
        model.feature_importances_,
        index=range(model.n_features_in_),
    )
    # 用 getattr 拿到 feature 名（RF 在 sklearn 1.x 才有 feature_names_in_）
    if hasattr(model, "feature_names_in_"):
        importance.index = model.feature_names_in_
    k = min(k, len(importance))
    return importance.nlargest(k)


def f1_by_category(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label_mapping: dict,
) -> dict[str, float]:
    """按攻击大类拆解的 F1 评估。

    Args:
        model: 已训练的分类器
        X_test: 测试特征
        y_test: 测试标签（整数）
        label_mapping: {整数标签: 大类名}，如 {0:'DoS', 1:'Probe', 2:'R2L', 3:'U2R'}

    Returns:
        dict{大类名: f1_score}
    """
    from sklearn.metrics import f1_score

    y_pred = model.predict(X_test)
    f1_dict: dict[str, float] = {}
    for label_val, category in label_mapping.items():
        # 把 y_test/y_pred 转为"是否属于该大类"的二分类
        y_true_bin = (y_test == label_val).astype(int)
        y_pred_bin = (y_pred == label_val).astype(int)
        f1_dict[category] = float(
            f1_score(y_true_bin, y_pred_bin, zero_division=0)
        )
    return f1_dict