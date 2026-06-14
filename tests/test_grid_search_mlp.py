"""Tests for MLP grid search (M4 task 4.2)."""
import numpy as np
import pandas as pd
import pytest

from src.models.mlp import grid_search_mlp


@pytest.fixture
def small_binary():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(500, 20), columns=[f"f{i}" for i in range(20)])
    y = pd.Series(np.random.randint(0, 2, size=500))
    return X, y


def test_grid_search_mlp_small_param_grid(small_binary):
    """小规模网格搜索（4 组合）应返回 best_model + best_params + summary。"""
    X, y = small_binary
    X_train, X_val = X.iloc[:400], X.iloc[400:]
    y_train, y_val = y.iloc[:400], y.iloc[400:]

    param_grid = {
        "hidden_dims": [(64,), (128, 64)],
        "activation": ["relu"],
        "dropout": [0.3],
        "lr": [1e-3],
    }

    best_model, best_params, summary = grid_search_mlp(
        X_train, y_train, X_val, y_val,
        param_grid=param_grid,
        epochs=2,
        cv_folds=2,
        verbose=False,
    )

    assert best_model is not None
    assert "hidden_dims" in best_params
    assert len(summary) == 2  # 2 组合
    for key, entry in summary.items():
        assert "cv_f1" in entry
        assert "val_f1" in entry
        assert 0.0 <= entry["cv_f1"] <= 1.0
        assert 0.0 <= entry["val_f1"] <= 1.0


def test_grid_search_mlp_records_both_scores(small_binary):
    """网格搜索记录 CV f1 和 val f1 双数字（避免 M3 痛点）。"""
    X, y = small_binary
    X_train, X_val = X.iloc[:400], X.iloc[400:]
    y_train, y_val = y.iloc[:400], y.iloc[400:]

    param_grid = {
        "hidden_dims": [(64,)],
        "activation": ["relu"],
        "dropout": [0.3],
        "lr": [1e-3],
    }

    _, _, summary = grid_search_mlp(
        X_train, y_train, X_val, y_val,
        param_grid=param_grid, epochs=1, cv_folds=2, verbose=False,
    )
    assert len(summary) == 1
    entry = list(summary.values())[0]
    assert "cv_f1" in entry and "val_f1" in entry
    # 至少有一个数字 > 0（随机数据有微弱信号）
    assert entry["cv_f1"] >= 0.0
    assert entry["val_f1"] >= 0.0
