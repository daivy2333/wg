"""Tests for src.models.random_forest (M3 task 3.4/3.5/3.6/3.7).

TDD Iron Law: tests written BEFORE implementation.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.models.random_forest import (
    train_rf_binary,
    train_rf_multiclass,
    grid_search_rf,
    get_top_k_features,
    f1_by_category,
    RF_BINARY_THRESHOLD,
    RF_MULTICLASS_THRESHOLD,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def small_binary():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(200, 8), columns=[f"f{i}" for i in range(8)])
    y = pd.Series(np.random.randint(0, 2, size=200))
    return X, y


@pytest.fixture
def small_multiclass():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(300, 8), columns=[f"f{i}" for i in range(8)])
    y = pd.Series(np.random.randint(0, 4, size=300))
    return X, y


# ============================================================
# R5: RF 二分类基线
# ============================================================

def test_train_rf_binary_returns_model(small_binary):
    X, y = small_binary
    model = train_rf_binary(X, y, random_state=42)
    assert isinstance(model, RandomForestClassifier)


def test_train_rf_binary_n_estimators(small_binary):
    X, y = small_binary
    model = train_rf_binary(X, y, random_state=42)
    assert model.n_estimators >= 100


def test_train_rf_binary_uses_balanced(small_binary):
    X, y = small_binary
    model = train_rf_binary(X, y, random_state=42)
    assert model.class_weight == "balanced"


# ============================================================
# R6: RF 多分类
# ============================================================

def test_train_rf_multiclass_returns_model(small_multiclass):
    X, y = small_multiclass
    model = train_rf_multiclass(X, y, random_state=42)
    assert isinstance(model, RandomForestClassifier)
    preds = model.predict(X)
    assert set(preds).issubset({0, 1, 2, 3})


# ============================================================
# R7: RF 网格搜索
# ============================================================

def test_grid_search_rf_returns_tuple(small_binary):
    X, y = small_binary
    best_model, best_params, best_score = grid_search_rf(
        X, y, task="binary", cv=3, random_state=42
    )
    assert isinstance(best_model, RandomForestClassifier)
    assert isinstance(best_params, dict)
    assert isinstance(best_score, float)


def test_grid_search_rf_params_cover_hyperparameters(small_binary):
    X, y = small_binary
    _, best_params, _ = grid_search_rf(X, y, task="binary", cv=3, random_state=42)
    assert "n_estimators" in best_params
    assert "max_depth" in best_params
    assert "min_samples_split" in best_params


# ============================================================
# R8: RF 特征重要度
# ============================================================

def test_get_top_k_features_returns_series(small_binary):
    X, y = small_binary
    model = train_rf_binary(X, y, random_state=42)
    top = get_top_k_features(model, k=5)
    assert isinstance(top, pd.Series)
    assert len(top) == 5


def test_get_top_k_features_sorted_descending(small_binary):
    X, y = small_binary
    model = train_rf_binary(X, y, random_state=42)
    top = get_top_k_features(model, k=5)
    # 降序
    assert (top.values[:-1] >= top.values[1:]).all()


def test_get_top_k_features_importance_sums_to_one(small_binary):
    X, y = small_binary
    model = train_rf_binary(X, y, random_state=42)
    full_importance = pd.Series(model.feature_importances_, index=X.columns)
    assert abs(full_importance.sum() - 1.0) < 1e-6


# ============================================================
# R9: 4 大攻击类别 F1
# ============================================================

def test_f1_by_category_returns_dict(small_multiclass):
    X, y = small_multiclass
    model = train_rf_multiclass(X, y, random_state=42)
    # 4 大类映射：0=DoS, 1=Probe, 2=R2L, 3=U2R
    label_mapping = {0: "DoS", 1: "Probe", 2: "R2L", 3: "U2R"}
    f1_dict = f1_by_category(model, X, y, label_mapping)
    assert isinstance(f1_dict, dict)
    assert "DoS" in f1_dict
    assert "Probe" in f1_dict
    assert "R2L" in f1_dict
    assert "U2R" in f1_dict


# ============================================================
# 常量
# ============================================================

def test_threshold_constants_defined():
    assert RF_BINARY_THRESHOLD > 0
    assert RF_MULTICLASS_THRESHOLD > 0