"""Tests for src.data.feature_selector (M2 task 2.7).

TDD Iron Law: tests written BEFORE implementation.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.data.feature_selector import (
    variance_threshold_filter,
    rf_feature_importance,
    select_top_k_features,
    feature_selection_pipeline,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def X_with_low_variance():
    """含近零方差列 + 数值列"""
    np.random.seed(42)
    return pd.DataFrame({
        "useful_a": np.random.randn(100),
        "useful_b": np.random.randn(100),
        "useful_c": np.random.randn(100),
        "useful_d": np.random.randn(100),
        "constant": [0.5] * 100,           # 零方差
        "near_constant": [0] * 99 + [1],   # 近零方差
    })


@pytest.fixture
def y_binary():
    np.random.seed(42)
    return pd.Series(np.random.randint(0, 2, size=100))


# ============================================================
# R5: 方差阈值
# ============================================================

def test_variance_threshold_removes_constant_column(X_with_low_variance):
    """过滤零方差列"""
    kept = variance_threshold_filter(X_with_low_variance, threshold=0.01)
    assert "constant" not in kept
    assert "useful_a" in kept


def test_variance_threshold_keeps_high_variance_columns(X_with_low_variance):
    """保留高方差列"""
    kept = variance_threshold_filter(X_with_low_variance, threshold=0.01)
    assert "useful_a" in kept
    assert "useful_b" in kept


def test_variance_threshold_threshold_parameter(X_with_low_variance):
    """threshold 参数化生效"""
    # 低 threshold 保留更多列
    kept_low = variance_threshold_filter(X_with_low_variance, threshold=0.0001)
    kept_high = variance_threshold_filter(X_with_low_variance, threshold=0.01)
    assert len(kept_low) >= len(kept_high)


# ============================================================
# R6: RF 特征重要度
# ============================================================

def test_rf_feature_importance_returns_series(X_with_low_variance, y_binary):
    """返回 pd.Series"""
    importance = rf_feature_importance(X_with_low_variance, y_binary, random_state=42)
    assert isinstance(importance, pd.Series)
    assert len(importance) == X_with_low_variance.shape[1]


def test_rf_feature_importance_values_sum_to_one(X_with_low_variance, y_binary):
    """feature_importances_ 之和 ≈ 1"""
    importance = rf_feature_importance(X_with_low_variance, y_binary, random_state=42)
    assert abs(importance.sum() - 1.0) < 1e-6


# ============================================================
# R7: Top-K 选择 + 双保险管线
# ============================================================

def test_select_top_k_features():
    """Top-K 返回最重要的 K 个特征"""
    importance = pd.Series({
        "a": 0.5, "b": 0.3, "c": 0.15, "d": 0.04, "e": 0.01
    })
    top_2 = select_top_k_features(importance, k=2)
    assert list(top_2) == ["a", "b"]


def test_select_top_k_k_equals_all():
    """k >= len 时返回全部"""
    importance = pd.Series({"a": 0.5, "b": 0.3})
    top_all = select_top_k_features(importance, k=5)
    assert set(top_all) == {"a", "b"}


def test_feature_selection_pipeline_returns_columns_and_model(X_with_low_variance, y_binary):
    """整合函数返回 (列名 list, RF 模型)"""
    cols, model = feature_selection_pipeline(
        X_with_low_variance, y_binary, top_k=3, random_state=42
    )
    assert isinstance(cols, list)
    assert isinstance(model, RandomForestClassifier)
    assert len(cols) == 3


def test_feature_selection_pipeline_test_transform(X_with_low_variance, y_binary):
    """训练 fit + 测试 transform 列名一致"""
    X_test = X_with_low_variance.copy()
    cols, _ = feature_selection_pipeline(
        X_with_low_variance, y_binary, top_k=2, random_state=42
    )
    # 测试集 transform 应用训练集保留的列
    X_test_filtered = X_test[cols]
    assert list(X_test_filtered.columns) == cols