"""Tests for src.models.decision_tree (M3 task 3.1/3.2/3.3).

TDD Iron Law: tests written BEFORE implementation.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.models.decision_tree import (
    train_dt_binary,
    train_dt_multiclass,
    grid_search_dt,
    evaluate_model,
    DT_BINARY_THRESHOLD,
    DT_MULTICLASS_THRESHOLD,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def small_binary():
    """小二分类合成数据"""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, size=100))
    return X, y


@pytest.fixture
def small_multiclass():
    """小多分类合成数据（3 类）"""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(150, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.random.randint(0, 3, size=150))
    return X, y


# ============================================================
# R1: DT 二分类基线
# ============================================================

def test_train_dt_binary_returns_model(small_binary):
    """返回 DecisionTreeClassifier"""
    X, y = small_binary
    model = train_dt_binary(X, y, random_state=42)
    assert isinstance(model, DecisionTreeClassifier)


def test_train_dt_binary_uses_balanced(small_binary):
    """class_weight='balanced'"""
    X, y = small_binary
    model = train_dt_binary(X, y, random_state=42)
    assert model.class_weight == "balanced"


def test_train_dt_binary_predict_works(small_binary):
    """predict 返回 0/1"""
    X, y = small_binary
    model = train_dt_binary(X, y, random_state=42)
    preds = model.predict(X)
    assert set(preds).issubset({0, 1})


# ============================================================
# R2: DT 多分类
# ============================================================

def test_train_dt_multiclass_returns_model(small_multiclass):
    """多分类返回 DecisionTreeClassifier"""
    X, y = small_multiclass
    model = train_dt_multiclass(X, y, random_state=42)
    assert isinstance(model, DecisionTreeClassifier)


def test_train_dt_multiclass_predict_works(small_multiclass):
    """多分类 predict 返回 0/1/2"""
    X, y = small_multiclass
    model = train_dt_multiclass(X, y, random_state=42)
    preds = model.predict(X)
    assert set(preds).issubset({0, 1, 2})


# ============================================================
# R3: 评估函数
# ============================================================

def test_evaluate_model_returns_metrics_dict(small_binary):
    """evaluate_model 返回 dict 含 5 指标"""
    X, y = small_binary
    model = train_dt_binary(X, y, random_state=42)
    metrics = evaluate_model(model, X, y)
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics


def test_evaluate_model_metrics_in_range(small_binary):
    """所有指标在 [0, 1]"""
    X, y = small_binary
    model = train_dt_binary(X, y, random_state=42)
    metrics = evaluate_model(model, X, y)
    for v in metrics.values():
        if isinstance(v, (int, float)):
            assert 0.0 <= v <= 1.0


def test_evaluate_model_multiclass_uses_macro(small_multiclass):
    """多分类评估含 f1_macro"""
    X, y = small_multiclass
    model = train_dt_multiclass(X, y, random_state=42)
    metrics = evaluate_model(model, X, y, task="multiclass")
    assert "f1_macro" in metrics


# ============================================================
# R3: GridSearchCV 调优
# ============================================================

def test_grid_search_dt_returns_tuple(small_binary):
    """返回 (best_model, best_params, best_score)"""
    X, y = small_binary
    result = grid_search_dt(X, y, task="binary", cv=3, random_state=42)
    assert len(result) == 3
    best_model, best_params, best_score = result
    assert isinstance(best_model, DecisionTreeClassifier)
    assert isinstance(best_params, dict)
    assert isinstance(best_score, float)


def test_grid_search_dt_params_cover_hyperparameters(small_binary):
    """best_params 包含 3 个超参数"""
    X, y = small_binary
    _, best_params, _ = grid_search_dt(X, y, task="binary", cv=3, random_state=42)
    assert "max_depth" in best_params
    assert "min_samples_split" in best_params
    assert "criterion" in best_params


# ============================================================
# 常量定义
# ============================================================

def test_threshold_constants_defined():
    """DT_BINARY_THRESHOLD / DT_MULTICLASS_THRESHOLD 已定义"""
    assert DT_BINARY_THRESHOLD > 0
    assert DT_MULTICLASS_THRESHOLD > 0