"""NSL-KDD 数据预处理模块

默认假设（NSL-KDD 学术惯例 + M1 BDD 决策）：
- 分类编码：protocol_type/flag → OneHot；service → LabelEncoder
- 标准化：StandardScaler
- 数据集划分：保留 KDDTrain+/KDDTest+ 原始划分
- 难度列：丢弃
"""
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .loader import COLUMN_NAMES, FEATURE_NAMES


# 数值特征列（41 特征中除去 3 个分类列）
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
NUMERIC_COLUMNS = [c for c in FEATURE_NAMES if c not in CATEGORICAL_COLUMNS]
DROPPABLE_COLUMNS = ["difficulty"]


def check_missing(df: pd.DataFrame) -> tuple[int, list[str]]:
    """检查缺失值。

    NSL-KDD 官方数据无缺失值，但保留检测能力供其他数据集复用。
    本 M1 阶段**不自动填充**，仅返回计数（Spec 简化 S1）。

    Args:
        df: 输入 DataFrame

    Returns:
        (n_missing, list_of_columns_with_missing)
    """
    n_missing = int(df.isnull().sum().sum())
    cols_with_missing = df.columns[df.isnull().any()].tolist()
    if n_missing > 0:
        import warnings
        warnings.warn(
            f"检测到 {n_missing} 个缺失值，列：{cols_with_missing}。"
            f"M1 阶段不自动填充，请调用方决定处理策略。",
            UserWarning,
            stacklevel=2,
        )
    return n_missing, cols_with_missing


def encode_categorical(
    df: pd.DataFrame,
    service_le: Optional[LabelEncoder] = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, LabelEncoder]:
    """分类特征编码（OneHot + Label 混合策略）。

    - protocol_type (3 种) → OneHot
    - flag (11 种) → OneHot
    - service (70+ 种高基数) → LabelEncoder

    Args:
        df: 输入 DataFrame（含 protocol_type/service/flag 列）
        service_le: 已 fit 的 LabelEncoder（测试集 transform 时传入）
        fit: True=训练模式（fit + transform），False=推理模式（仅 transform）

    Returns:
        (encoded_df, fitted_label_encoder)
    """
    df = df.copy()

    # 1. LabelEncoder for service（高基数，避免 OneHot 维度爆炸）
    if fit:
        service_le = LabelEncoder()
        df["service"] = service_le.fit_transform(df["service"])
    else:
        # 测试集可能含训练集未见过的 service 值，扩展 encoder
        assert service_le is not None, "推理模式必须传入已 fit 的 service_le"
        known = set(service_le.classes_)
        new_values = set(df["service"].unique()) - known
        if new_values:
            # 追加新类别到 encoder
            service_le.classes_ = np.concatenate([service_le.classes_, sorted(new_values)])
        df["service"] = service_le.transform(df["service"])

    # 2. OneHot for protocol_type + flag（低基数）
    df = pd.get_dummies(df, columns=["protocol_type", "flag"], prefix=["protocol_type", "flag"])

    # 确保 OneHot 列只含 0/1 整数（避免 bool 类型在某些 sklearn 模型报错）
    onehot_cols = [c for c in df.columns if c.startswith(("protocol_type_", "flag_"))]
    for col in onehot_cols:
        df[col] = df[col].astype(np.int8)

    return df, service_le


def standardize(
    X: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    """数值特征标准化（StandardScaler）。

    Args:
        X: 数值特征 DataFrame
        scaler: 已 fit 的 StandardScaler（测试集 transform 时传入）
        fit: True=训练模式（fit + transform），False=推理模式（仅 transform）

    Returns:
        (scaled_df, fitted_scaler)
    """
    if fit:
        scaler = StandardScaler()
        values = scaler.fit_transform(X)
    else:
        assert scaler is not None, "推理模式必须传入已 fit 的 scaler"
        values = scaler.transform(X)

    return pd.DataFrame(values, columns=X.columns, index=X.index), scaler


def make_labels(
    labels: pd.Series,
    task: Literal["binary", "multiclass"] = "binary",
) -> pd.Series:
    """生成二分类或多分类标签。

    Args:
        labels: 原始 label 列（string 类型：normal / 各种攻击名）
        task: 'binary' → 0=normal, 1=anomaly
              'multiclass' → 0..N-1（LabelEncoder 整数映射）

    Returns:
        整数标签 Series
    """
    if task == "binary":
        return (labels != "normal").astype(np.int8)
    elif task == "multiclass":
        le = LabelEncoder()
        return pd.Series(le.fit_transform(labels), index=labels.index, dtype=np.int32)
    else:
        raise ValueError(f"未知 task: {task!r}（仅支持 'binary' 或 'multiclass'）")


def preprocess_pipeline(
    df: pd.DataFrame,
    task: Literal["binary", "multiclass"] = "binary",
    service_le: Optional[LabelEncoder] = None,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """预处理整合管线（fit 训练集 / transform 测试集统一接口）。

    流程：
        1. 缺失值检查（warn 但不自动填充，Spec S1）
        2. 标签提取 + 二分类/多分类转换
        3. 分类特征编码（OneHot + Label 混合）
        4. 难度列丢弃
        5. 数值特征标准化（StandardScaler）

    Args:
        df: 输入 DataFrame（loader.load_train() 或 load_test() 的输出）
        task: 'binary' 或 'multiclass'
        service_le: 已 fit 的 service LabelEncoder（测试集 transform 时传入）
        scaler: 已 fit 的 StandardScaler（测试集 transform 时传入）
        fit: True=训练，False=推理

    Returns:
        (X, y, encoders)
        - X: 预处理后特征 DataFrame（不含 difficulty 和 label）
        - y: 标签 Series
        - encoders: dict，含 service_le / scaler（供 M3/M4 复用）
    """
    # 1. 缺失值检查
    check_missing(df)

    # 2. 标签提取（在编码前，避免 label 被转换）
    y = make_labels(df["label"], task=task)

    # 3. 分离特征
    X = df.drop(columns=["label"])

    # 4. 分类特征编码
    X, service_le = encode_categorical(X, service_le=service_le, fit=fit)

    # 5. 丢弃 difficulty
    X = X.drop(columns=DROPPABLE_COLUMNS)

    # 6. 数值特征标准化
    X, scaler = standardize(X, scaler=scaler, fit=fit)

    encoders = {"service_le": service_le, "scaler": scaler}
    return X, y, encoders