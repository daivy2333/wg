"""特征选择模块（M2 任务 2.7）

双保险策略：VarianceThreshold + RandomForest Top-K
"""
from typing import Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold


# 默认参数（M2 BDD 默认假设）
DEFAULT_VARIANCE_THRESHOLD: float = 0.01
DEFAULT_TOP_K: int = 20
DEFAULT_N_ESTIMATORS: int = 100
DEFAULT_RANDOM_STATE: int = 42


def variance_threshold_filter(
    X: pd.DataFrame,
    threshold: float = DEFAULT_VARIANCE_THRESHOLD,
) -> list[str]:
    """方差阈值过滤：去除方差 < threshold 的列。

    Args:
        X: 输入特征 DataFrame
        threshold: 方差阈值（默认 0.01）

    Returns:
        保留的列名 list
    """
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    return X.columns[selector.get_support()].tolist()


def rf_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.Series:
    """随机森林特征重要度评分。

    Args:
        X: 输入特征 DataFrame
        y: 二分类标签 Series
        n_estimators: 树数量（默认 100）
        random_state: 随机种子

    Returns:
        pd.Series，索引为列名，值为 feature_importances_
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns)


def select_top_k_features(importance: pd.Series, k: int) -> list[str]:
    """选 Top-K 重要特征。

    Args:
        importance: rf_feature_importance 返回的 Series
        k: 保留数量（k >= len(importance) 时返回全部）

    Returns:
        按重要度降序的列名 list
    """
    k = min(k, len(importance))
    return importance.nlargest(k).index.tolist()


def feature_selection_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: int = DEFAULT_TOP_K,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[list[str], RandomForestClassifier]:
    """双保险特征选择整合管线。

    流程：
        1. VarianceThreshold 过滤低方差列
        2. 在过滤后的特征上训练 RF
        3. 取 Top-K 重要特征

    Args:
        X_train: 训练特征
        y_train: 训练标签
        top_k: 保留数量（默认 20）
        variance_threshold: 方差阈值（默认 0.01）
        random_state: 随机种子

    Returns:
        (保留列名 list, 训练好的 RF 模型)
        测试集用 `X_test[cols]` 应用相同列。
    """
    # 1. 方差阈值
    kept_cols = variance_threshold_filter(X_train, threshold=variance_threshold)
    X_filtered = X_train[kept_cols]

    # 2. RF 训练
    rf = RandomForestClassifier(
        n_estimators=DEFAULT_N_ESTIMATORS,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_filtered, y_train)

    # 3. Top-K
    importance = pd.Series(rf.feature_importances_, index=kept_cols)
    final_cols = select_top_k_features(importance, k=top_k)

    return final_cols, rf