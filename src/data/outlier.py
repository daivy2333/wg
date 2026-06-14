"""异常值处理模块（M2 任务 2.4）

默认策略：IQR + clip + log1p（不丢样本）
"""
from typing import Iterable

import numpy as np
import pandas as pd


# NSL-KDD 长尾数值特征（M2 BDD 默认假设）
LONG_TAIL_COLUMNS: list[str] = [
    "src_bytes",
    "dst_bytes",
    "count",
    "srv_count",
]

# IQR 系数（标准 1.5，Tukey's fence）
IQR_K: float = 1.5


def _safe_numeric(df: pd.DataFrame, column: str) -> pd.Series | None:
    """返回数值列；非数值列返回 None。"""
    if column not in df.columns:
        return None
    s = df[column]
    if not pd.api.types.is_numeric_dtype(s):
        return None
    return s


def detect_outliers_iqr(df: pd.DataFrame, columns: Iterable[str]) -> dict[str, int]:
    """IQR 异常值检测：每列超限值数量（Q1 - 1.5*IQR 之外 或 Q3 + 1.5*IQR 之外）。

    Args:
        df: 输入 DataFrame
        columns: 要检测的列名列表

    Returns:
        dict[列名, 超限值数量]。非数值列或不存在的列返回 0。
    """
    result: dict[str, int] = {}
    for col in columns:
        s = _safe_numeric(df, col)
        if s is None:
            result[col] = 0
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - IQR_K * iqr
        upper = q3 + IQR_K * iqr
        result[col] = int(((s < lower) | (s > upper)).sum())
    return result


def clip_outliers(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """IQR + clip：超限值替换为边界值（Q1 - 1.5*IQR 或 Q3 + 1.5*IQR）。

    不丢样本（行数不变）。clip 不引入 NaN。

    Args:
        df: 输入 DataFrame
        columns: 要 clip 的列名列表

    Returns:
        clip 后的新 DataFrame（原 df 不修改）
    """
    df = df.copy()
    for col in columns:
        s = _safe_numeric(df, col)
        if s is None:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - IQR_K * iqr
        upper = q3 + IQR_K * iqr
        df[col] = s.clip(lower=lower, upper=upper)
    return df


def log1p_transform(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """log1p 长尾变换：新列追加（原列保留）。

    log1p(0) = 0，不产生 -inf 或 NaN。

    Args:
        df: 输入 DataFrame
        columns: 要变换的列名列表

    Returns:
        含原列 + `{col}_log1p` 新列的 DataFrame
    """
    df = df.copy()
    for col in columns:
        s = _safe_numeric(df, col)
        if s is None:
            continue
        df[f"{col}_log1p"] = np.log1p(s)
    return df


def outlier_pipeline(
    df: pd.DataFrame,
    numeric_cols: Iterable[str],
    long_tail_cols: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """异常值处理整合管线。

    流程：
        1. IQR 检测（统计每列超限数量）
        2. clip 替换超限值（保留行）
        3. log1p 变换长尾特征

    Args:
        df: 输入 DataFrame
        numeric_cols: 参与 IQR clip 的数值列
        long_tail_cols: 参与 log1p 变换的列；默认 = LONG_TAIL_COLUMNS ∩ numeric_cols

    Returns:
        (处理后 DataFrame, 异常值统计 dict)
    """
    long_tail_cols = list(long_tail_cols) if long_tail_cols is not None else LONG_TAIL_COLUMNS

    # 1. 检测
    stats = detect_outliers_iqr(df, numeric_cols)

    # 2. clip
    df = clip_outliers(df, numeric_cols)

    # 3. log1p
    long_tail_in_df = [c for c in long_tail_cols if c in df.columns]
    df = log1p_transform(df, long_tail_in_df)

    return df, stats