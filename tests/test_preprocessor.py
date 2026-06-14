"""Tests for src.data.preprocessor (M1 task 1.7).

TDD Iron Law: tests written BEFORE implementation.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.data.preprocessor import (
    check_missing,
    encode_categorical,
    make_labels,
    preprocess_pipeline,
    standardize,
)


# ============================================================
# 测试辅助 fixture：小型合成 DataFrame
# ============================================================

@pytest.fixture
def sample_df():
    """8 行小型合成 NSL-KDD 数据（含分类 + 数值 + label + difficulty）"""
    return pd.DataFrame({
        "duration": [0, 1, 2, 3, 4, 5, 6, 7],
        "protocol_type": ["tcp", "udp", "icmp", "tcp", "udp", "icmp", "tcp", "udp"],
        "service": ["http", "telnet", "ftp", "http", "smtp", "domain_u", "http", "telnet"],
        "flag": ["SF", "S0", "REJ", "SF", "RSTR", "S0", "SF", "RSTO"],
        "src_bytes": [100, 200, 0, 50, 300, 0, 150, 250],
        "dst_bytes": [0, 500, 0, 100, 200, 0, 50, 300],
        "label": ["normal", "neptune", "normal", "satan", "neptune", "normal", "portsweep", "neptune"],
        "difficulty": [21, 21, 21, 21, 21, 21, 21, 21],
    })


# ============================================================
# R11: 缺失值处理
# ============================================================

def test_check_missing_no_missing(sample_df):
    """无缺失值时返回 (0, 0)"""
    n_missing, _ = check_missing(sample_df)
    assert n_missing == 0


def test_check_missing_detects_missing(sample_df):
    """检测到缺失值时正确计数"""
    sample_df.loc[0, "src_bytes"] = np.nan
    n_missing, columns = check_missing(sample_df)
    assert n_missing == 1
    assert "src_bytes" in columns


# ============================================================
# R12: 分类特征编码（混合策略）
# ============================================================

def test_encode_categorical_drops_onehot_original_columns(sample_df):
    """OneHot 处理后，protocol_type 和 flag 原列被删除（pd.get_dummies 行为）"""
    result, _ = encode_categorical(sample_df)
    assert "protocol_type" not in result.columns
    assert "flag" not in result.columns


def test_encode_categorical_creates_onehot_columns(sample_df):
    """OneHot 列被创建（protocol_type + flag）"""
    result, _ = encode_categorical(sample_df)
    # protocol_type: 3 种 → 3 个 OneHot 列
    onehot_pt = [c for c in result.columns if c.startswith("protocol_type_")]
    assert len(onehot_pt) == 3
    # flag: fixture 中 5 种唯一值（SF/S0/REJ/RSTR/RSTO）→ 5 个 OneHot 列
    onehot_flag = [c for c in result.columns if c.startswith("flag_")]
    assert len(onehot_flag) == 5


def test_encode_categorical_service_label_encoded(sample_df):
    """service 列被 LabelEncoder 编码为整数"""
    result, _ = encode_categorical(sample_df)
    assert result["service"].dtype in (np.int64, np.int32, int)
    assert result["service"].min() >= 0
    assert result["service"].max() < len(sample_df["service"].unique())


def test_encode_categorical_returns_consistent_encoder_for_test(sample_df):
    """对测试集使用相同 encoder，避免 unseen label"""
    _, service_le = encode_categorical(sample_df, fit=True)
    # 测试集有新的 service 值，应返回 None（视为 anomaly）或抛错 → 当前实现接受并追加
    test_df = sample_df.copy()
    test_df.loc[0, "service"] = "new_service_unseen"
    result, _ = encode_categorical(test_df, service_le=service_le, fit=False)
    # 训练时未见过的 service 应被赋予新整数
    assert result.loc[0, "service"] >= len(sample_df["service"].unique())


# ============================================================
# R13: 数值特征标准化
# ============================================================

def test_standardize_returns_dataframe():
    """标准化返回 DataFrame 而非 ndarray"""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
    result, _ = standardize(X)
    assert isinstance(result, pd.DataFrame)


def test_standardize_mean_zero():
    """标准化后均值 ≈ 0"""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
    result, _ = standardize(X)
    assert abs(result["a"].mean()) < 1e-6
    assert abs(result["b"].mean()) < 1e-6


def test_standardize_std_one():
    """标准化后标准差 ≈ 1（用 ddof=0 与 StandardScaler 一致）"""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
    result, _ = standardize(X)
    # pandas std() 默认 ddof=1，StandardScaler 用 ddof=0，统一用 ddof=0
    assert abs(result["a"].std(ddof=0) - 1.0) < 1e-6
    assert abs(result["b"].std(ddof=0) - 1.0) < 1e-6


def test_standardize_fit_transform_consistency():
    """训练集 fit + 测试集 transform 使用同一 scaler"""
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    X_test = pd.DataFrame({"a": [5.0, 6.0]})
    _, scaler = standardize(X_train, fit=True)
    result_test, _ = standardize(X_test, scaler=scaler, fit=False)
    # 测试集均值约 5.5，标准化后应为正数
    assert result_test["a"].mean() > 0


# ============================================================
# R14: 难度列丢弃
# ============================================================

def test_preprocess_pipeline_drops_difficulty(sample_df):
    """预处理管线丢弃 difficulty 列"""
    X, y, _ = preprocess_pipeline(sample_df, task="binary")
    assert "difficulty" not in X.columns


# ============================================================
# R15: 数据集划分（保留原始）
# ============================================================

def test_preprocess_pipeline_preserves_sample_count(sample_df):
    """保留原始样本数（不做随机划分）"""
    X, y, _ = preprocess_pipeline(sample_df, task="binary")
    assert len(X) == len(sample_df)


# ============================================================
# R16: 二分类与多分类标签
# ============================================================

def test_make_labels_binary():
    """二分类标签：0=normal, 1=anomaly"""
    s = pd.Series(["normal", "neptune", "satan", "normal", "portsweep"])
    result = make_labels(s, task="binary")
    assert set(result.unique()) == {0, 1}
    assert (result == 0).sum() == 2  # 2 个 normal
    assert (result == 1).sum() == 3  # 3 个 anomaly


def test_make_labels_multiclass():
    """多分类标签：保留 23 个唯一值"""
    s = pd.Series(["normal", "neptune", "satan", "normal", "portsweep", "smurf"])
    result = make_labels(s, task="multiclass")
    assert result.nunique() == 5
    # 标签值范围应映射为 0-N
    assert result.min() == 0


# ============================================================
# R17: 预处理管线整合
# ============================================================

def test_preprocess_pipeline_returns_X_y_encoders(sample_df):
    """整合函数返回 (X, y, encoders) 三元组"""
    X, y, encoders = preprocess_pipeline(sample_df, task="binary")
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(encoders, dict)
    assert "service_le" in encoders
    assert "scaler" in encoders


def test_preprocess_pipeline_binary_label_only(sample_df):
    """二分类任务 y 仅含 0/1"""
    X, y, _ = preprocess_pipeline(sample_df, task="binary")
    assert set(y.unique()) == {0, 1}


def test_preprocess_pipeline_no_difficulty_no_label_in_X(sample_df):
    """X 不含 difficulty 和 label"""
    X, y, _ = preprocess_pipeline(sample_df, task="binary")
    assert "difficulty" not in X.columns
    assert "label" not in X.columns