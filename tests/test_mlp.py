"""Tests for src.models.mlp (M4 task 4.1/4.2/4.3).

TDD Iron Law: tests written BEFORE extensive use.
"""
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.models.mlp import (
    MLPClassifier,
    NSLKDDDataset,
    UNSEEN_ATTACKS_IN_TEST,
    compute_known_class_accuracy,
    make_dataloader,
    train_mlp_binary,
    train_mlp_multiclass,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def small_binary():
    """小规模二分类数据：1000 样本，20 维。"""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(1000, 20), columns=[f"f{i}" for i in range(20)])
    y = pd.Series(np.random.randint(0, 2, size=1000))
    return X, y


@pytest.fixture
def small_multiclass():
    """小规模多分类数据：1500 样本，20 维，5 类。"""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(1500, 20), columns=[f"f{i}" for i in range(20)])
    y = pd.Series(np.random.randint(0, 5, size=1500))
    return X, y


# ============================================================
# R-mlp-1: MLP 模型定义
# ============================================================


def test_mlp_default_architecture():
    """默认架构：(20)→(128,64)→(2)，含 ReLU + Dropout 0.3。"""
    model = MLPClassifier(input_dim=20, output_dim=2)
    assert isinstance(model, nn.Module)
    assert model.input_dim == 20
    assert model.output_dim == 2
    assert model.hidden_dims == (128, 64)
    assert model.dropout == 0.3
    assert model.activation == "relu"


def test_mlp_custom_hidden_dims():
    """自定义隐藏层：(20)→(256,128,64)→(10)。"""
    model = MLPClassifier(input_dim=20, output_dim=10, hidden_dims=(256, 128, 64), dropout=0.5)
    assert model.hidden_dims == (256, 128, 64)
    assert model.dropout == 0.5


def test_mlp_invalid_activation():
    """非法 activation 应抛 ValueError。"""
    with pytest.raises(ValueError, match="activation 必须是"):
        MLPClassifier(input_dim=20, output_dim=2, activation="sigmoid")


def test_mlp_forward_shape_binary():
    """前向传播：输入 (4, 20) → 输出 (4, 2)。"""
    model = MLPClassifier(input_dim=20, output_dim=2)
    x = torch.randn(4, 20)
    y = model(x)
    assert y.shape == (4, 2)


def test_mlp_forward_shape_multiclass():
    """前向传播：输入 (8, 20) → 输出 (8, 5)。"""
    model = MLPClassifier(input_dim=20, output_dim=5)
    x = torch.randn(8, 20)
    y = model(x)
    assert y.shape == (8, 5)


def test_mlp_predict_returns_ndarray():
    """predict 返回 numpy 数组。"""
    model = MLPClassifier(input_dim=20, output_dim=2)
    model.eval()
    X = np.random.randn(10, 20)
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (10,)
    assert set(preds.tolist()).issubset({0, 1})


def test_mlp_predict_proba_sums_to_one():
    """predict_proba 每行和为 1（softmax）。"""
    model = MLPClassifier(input_dim=20, output_dim=2)
    X = np.random.randn(5, 20)
    proba = model.predict_proba(X)
    assert proba.shape == (5, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ============================================================
# R-mlp-5: Dataset / DataLoader
# ============================================================


def test_nslkdd_dataset_len_and_getitem(small_binary):
    """NSLKDDDataset __len__ 和 __getitem__。"""
    X, y = small_binary
    dataset = NSLKDDDataset(X, y)
    assert len(dataset) == 1000
    x0, y0 = dataset[0]
    assert x0.shape == (20,)
    assert y0.ndim == 0  # scalar tensor


def test_make_dataloader_batch_shape(small_binary):
    """DataLoader batch 形状：(256, 20) + (256,)。"""
    X, y = small_binary
    loader = make_dataloader(X, y, batch_size=256, shuffle=False)
    batch = next(iter(loader))
    assert len(batch) == 2
    xb, yb = batch
    assert xb.shape == (256, 20)
    assert yb.shape == (256,)


# ============================================================
# R-mlp-2: MLP 二分类训练
# ============================================================


def test_train_mlp_binary_returns_tuple(small_binary):
    """train_mlp_binary 返回 TrainResult。"""
    X, y = small_binary
    X_train, X_val = X.iloc[:800], X.iloc[800:]
    y_train, y_val = y.iloc[:800], y.iloc[800:]

    result = train_mlp_binary(X_train, y_train, X_val, y_val, epochs=2, verbose=False)
    assert hasattr(result, "model")
    assert hasattr(result, "val_metrics")
    assert hasattr(result, "train_history")
    assert isinstance(result.model, MLPClassifier)
    assert "f1" in result.val_metrics
    assert "accuracy" in result.val_metrics
    assert "auc" in result.val_metrics


def test_train_mlp_binary_metrics_in_range(small_binary):
    """指标在 [0, 1] 范围。"""
    X, y = small_binary
    X_train, X_val = X.iloc[:800], X.iloc[800:]
    y_train, y_val = y.iloc[:800], y.iloc[800:]

    result = train_mlp_binary(X_train, y_train, X_val, y_val, epochs=3, verbose=False)
    for key in ["accuracy", "precision", "recall", "f1", "auc"]:
        assert 0.0 <= result.val_metrics[key] <= 1.0, f"{key} out of range: {result.val_metrics[key]}"


def test_train_mlp_binary_early_stopping(small_binary):
    """早停触发时 epochs 可能 < 配置。"""
    X, y = small_binary
    X_train, X_val = X.iloc[:800], X.iloc[800:]
    y_train, y_val = y.iloc[:800], y.iloc[800:]

    result = train_mlp_binary(
        X_train, y_train, X_val, y_val, epochs=50, patience=2, verbose=False
    )
    # 早停可能减少实际 epoch
    actual_epochs = len(result.train_history["train_loss"])
    assert actual_epochs <= 50


# ============================================================
# R-mlp-4: MLP 多分类训练
# ============================================================


def test_train_mlp_multiclass_returns_metrics(small_multiclass):
    """多分类训练返回 f1_macro + per_class_f1。"""
    X, y = small_multiclass
    X_train, X_val = X.iloc[:1200], X.iloc[1200:]
    y_train, y_val = y.iloc[:1200], y.iloc[1200:]

    result = train_mlp_multiclass(
        X_train, y_train, X_val, y_val, num_classes=5, epochs=2, verbose=False
    )
    assert "f1_macro" in result.val_metrics
    assert "accuracy" in result.val_metrics
    assert "per_class_f1" in result.val_metrics
    assert len(result.val_metrics["per_class_f1"]) == 5


def test_unseen_attacks_constant():
    """UNSEEN_ATTACKS_IN_TEST 常量含 16 种攻击（M2 EDA 识别）。"""
    assert len(UNSEEN_ATTACKS_IN_TEST) == 16
    assert "apache2" in UNSEEN_ATTACKS_IN_TEST
    assert "worm" in UNSEEN_ATTACKS_IN_TEST


def test_compute_known_class_accuracy_no_unseen(small_multiclass):
    """unseen_label_ids 为空时 = 全量 acc。"""
    X, y = small_multiclass
    X_train, X_val = X.iloc[:1200], X.iloc[1200:]
    y_train, y_val = y.iloc[:1200], y.iloc[1200:]

    result = train_mlp_multiclass(
        X_train, y_train, X_val, y_val, num_classes=5, epochs=2, verbose=False
    )
    acc = compute_known_class_accuracy(result.model, X_val, y_val)
    # 应等于 val_metrics.accuracy
    np.testing.assert_almost_equal(acc, result.val_metrics["accuracy"], decimal=5)


def test_compute_known_class_accuracy_filter_unseen(small_multiclass):
    """剔除 unseen 类后只算剩余样本。"""
    X, y = small_multiclass
    X_train, X_val = X.iloc[:1200], X.iloc[1200:]
    y_train, y_val = y.iloc[:1200], y.iloc[1200:]

    result = train_mlp_multiclass(
        X_train, y_train, X_val, y_val, num_classes=5, epochs=2, verbose=False
    )
    # 剔除类 0, 1 后计算 acc
    acc_filtered = compute_known_class_accuracy(result.model, X_val, y_val, unseen_label_ids=[0, 1])
    # acc_filtered 应该与全量 acc 不同（除非全量样本本来就不含 0, 1）
    assert 0.0 <= acc_filtered <= 1.0
