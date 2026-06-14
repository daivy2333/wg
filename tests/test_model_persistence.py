"""Tests for src.models.persistence (M3 task 3.8).

TDD Iron Law: tests written BEFORE implementation.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.models.persistence import save_model, load_model, save_best_models


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def dummy_rf():
    """小型 RF 模型用于测试"""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 4), columns=[f"f{i}" for i in range(4)])
    y = pd.Series(np.random.randint(0, 2, size=50))
    model = RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=1)
    model.fit(X, y)
    return model


@pytest.fixture
def dummy_dt():
    X = pd.DataFrame(np.random.randn(50, 4), columns=[f"f{i}" for i in range(4)])
    y = pd.Series(np.random.randint(0, 2, size=50))
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model


# ============================================================
# R10: joblib save/load
# ============================================================

def test_save_model_creates_file(dummy_rf, tmp_path):
    """save_model 创建 .joblib 文件"""
    path = tmp_path / "test.joblib"
    save_model(dummy_rf, path)
    assert path.exists()


def test_load_model_roundtrip(dummy_rf, tmp_path):
    """save + load 还原模型"""
    path = tmp_path / "test.joblib"
    save_model(dummy_rf, path)
    loaded = load_model(path)
    assert isinstance(loaded, RandomForestClassifier)


def test_load_model_predict_works(dummy_rf, tmp_path):
    """加载后的模型可 predict"""
    path = tmp_path / "test.joblib"
    save_model(dummy_rf, path)
    loaded = load_model(path)
    X = pd.DataFrame(np.random.randn(3, 4), columns=[f"f{i}" for i in range(4)])
    preds = loaded.predict(X)
    assert len(preds) == 3


def test_load_model_missing_file_raises(tmp_path):
    """文件不存在抛 FileNotFoundError"""
    with pytest.raises(FileNotFoundError, match="missing.joblib"):
        load_model(tmp_path / "missing.joblib")


def test_save_model_creates_parent_dirs(dummy_rf, tmp_path):
    """自动创建中间目录"""
    nested = tmp_path / "a" / "b" / "model.joblib"
    save_model(dummy_rf, nested)
    assert nested.exists()


# ============================================================
# R11: 4 模型批量保存
# ============================================================

def test_save_best_models_creates_4_files(dummy_dt, dummy_rf, tmp_path):
    """save_best_models 生成 4 个文件"""
    save_best_models(
        dt_binary=dummy_dt,
        rf_binary=dummy_rf,
        dt_multiclass=dummy_dt,
        rf_multiclass=dummy_rf,
        output_dir=tmp_path,
    )
    assert (tmp_path / "dt_binary_best.joblib").exists()
    assert (tmp_path / "rf_binary_best.joblib").exists()
    assert (tmp_path / "dt_multiclass_best.joblib").exists()
    assert (tmp_path / "rf_multiclass_best.joblib").exists()


def test_save_best_models_loads_correctly(dummy_dt, dummy_rf, tmp_path):
    """保存后可加载"""
    save_best_models(
        dt_binary=dummy_dt,
        rf_binary=dummy_rf,
        dt_multiclass=dummy_dt,
        rf_multiclass=dummy_rf,
        output_dir=tmp_path,
    )
    loaded = load_model(tmp_path / "rf_binary_best.joblib")
    assert isinstance(loaded, RandomForestClassifier)
    loaded_dt = load_model(tmp_path / "dt_binary_best.joblib")
    assert isinstance(loaded_dt, DecisionTreeClassifier)