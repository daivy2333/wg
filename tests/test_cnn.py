"""Tests for src.models.cnn (M4 task 4.4)."""
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.models.cnn import CNN1DClassifier, train_cnn_binary


@pytest.fixture
def small_binary():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(1000, 20), columns=[f"f{i}" for i in range(20)])
    y = pd.Series(np.random.randint(0, 2, size=1000))
    return X, y


def test_cnn_default_architecture():
    model = CNN1DClassifier(input_length=20, output_dim=2)
    assert isinstance(model, nn.Module)
    assert model.input_length == 20
    assert model.output_dim == 2
    assert model.conv_channels == (16,)
    assert model.kernel_size == 3


def test_cnn_invalid_input_length():
    with pytest.raises(ValueError, match="input_length=20"):
        CNN1DClassifier(input_length=41, output_dim=2)


def test_cnn_forward_shape():
    model = CNN1DClassifier(input_length=20, output_dim=2)
    x = torch.randn(4, 20)
    y = model(x)
    assert y.shape == (4, 2)


def test_cnn_custom_conv_channels():
    model = CNN1DClassifier(input_length=20, output_dim=2, conv_channels=(16, 32), kernel_size=5)
    assert model.conv_channels == (16, 32)
    assert model.kernel_size == 5
    x = torch.randn(4, 20)
    y = model(x)
    assert y.shape == (4, 2)


def test_cnn_predict_returns_ndarray():
    model = CNN1DClassifier(input_length=20, output_dim=2)
    model.eval()
    X = np.random.randn(10, 20)
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (10,)
    assert set(preds.tolist()).issubset({0, 1})


def test_train_cnn_binary_returns_metrics(small_binary):
    X, y = small_binary
    X_train, X_val = X.iloc[:800], X.iloc[800:]
    y_train, y_val = y.iloc[:800], y.iloc[800:]

    result = train_cnn_binary(X_train, y_train, X_val, y_val, epochs=2, verbose=False)
    assert "f1" in result.val_metrics
    assert "accuracy" in result.val_metrics
    assert "auc" in result.val_metrics
    for key in ["accuracy", "precision", "recall", "f1", "auc"]:
        assert 0.0 <= result.val_metrics[key] <= 1.0
