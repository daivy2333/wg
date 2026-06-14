"""Tests for src.data.smote (M4 task 4.6)."""
import numpy as np
import pandas as pd
import pytest

from src.data.smote import (
    apply_smote_to_minority_classes,
    get_minority_class_ids,
)


@pytest.fixture
def imbalanced_data():
    """构造不均衡数据：1000 样本，3 类（多数 800 + 少数 100 + 极少数 30）。"""
    np.random.seed(42)
    X = np.random.randn(1000, 20).astype(np.float32)
    y = np.array([0] * 800 + [1] * 100 + [2] * 30 + [3] * 70)
    return X, y


def test_get_minority_class_ids():
    """识别小样本类。"""
    y = np.array([0] * 800 + [1] * 100 + [2] * 30 + [3] * 70)
    minority = get_minority_class_ids(y, threshold=200)
    # 1 (100), 2 (30), 3 (70) 都是 < 200
    assert set(minority) == {1, 2, 3}


def test_apply_smote_to_target_classes(imbalanced_data):
    """对 target_classes 做 SMOTE。"""
    X, y = imbalanced_data
    X_res, y_res = apply_smote_to_minority_classes(
        X, y, target_classes=[1, 2, 3], k_neighbors=3
    )
    # SMOTE 后样本数应增加
    assert X_res.shape[0] > X.shape[0]
    # 特征维度不变
    assert X_res.shape[1] == 20
    # 多数类（0）样本数不变
    assert (y_res == 0).sum() == 800
    # 少数类样本数提升到多数类水平（800）
    for cls in [1, 2, 3]:
        assert (y_res == cls).sum() == 800


def test_apply_smote_insufficient_samples_raises():
    """目标类样本数 < k_neighbors+1 时抛 ValueError。"""
    X = np.random.randn(50, 20).astype(np.float32)
    y = np.array([0] * 45 + [1] * 5)  # 类 1 仅 5 个样本
    with pytest.raises(ValueError, match="少于 SMOTE 要求"):
        apply_smote_to_minority_classes(X, y, target_classes=[1], k_neighbors=5)


def test_apply_smote_dataframe_input(imbalanced_data):
    """支持 DataFrame 输入。"""
    X, y = imbalanced_data
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
    y_series = pd.Series(y)
    X_res, y_res = apply_smote_to_minority_classes(
        X_df, y_series, target_classes=[1, 2], k_neighbors=3
    )
    assert X_res.shape[0] > X.shape[0]


def test_imblearn_missing_raises(monkeypatch):
    """imblearn 缺失时抛 ImportError。"""
    import src.data.smote as smote_mod

    monkeypatch.setattr(smote_mod, "IMBLEARN_AVAILABLE", False)
    with pytest.raises(ImportError, match="imbalanced-learn 未安装"):
        apply_smote_to_minority_classes(
            np.random.randn(100, 20).astype(np.float32),
            np.array([0] * 90 + [1] * 10),
            target_classes=[1],
            k_neighbors=3,
        )
