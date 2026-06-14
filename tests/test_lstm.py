"""Tests for src.models.lstm (M4 task 4.5)."""
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.models.lstm import LSTMClassifier, train_lstm_binary


@pytest.fixture
def small_binary():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(1000, 20), columns=[f"f{i}" for i in range(20)])
    y = pd.Series(np.random.randint(0, 2, size=1000))
    return X, y


def test_lstm_default_architecture():
    model = LSTMClassifier(input_size=20, output_dim=2)
    assert isinstance(model, nn.Module)
    assert model.input_size == 20
    assert model.output_dim == 2
    assert model.hidden_size == 32
    assert model.num_layers == 1


def test_lstm_invalid_input_size():
    with pytest.raises(ValueError, match="input_size=20"):
        LSTMClassifier(input_size=41, output_dim=2)


def test_lstm_forward_shape():
    model = LSTMClassifier(input_size=20, output_dim=2)
    x = torch.randn(4, 20)  # (batch, features) — forward 会自动加 seq_len=1
    y = model(x)
    assert y.shape == (4, 2)


def test_lstm_invalid_seq_len():
    model = LSTMClassifier(input_size=20, output_dim=2)
    # seq_len=5 不被允许（应抛 ValueError）
    x = torch.randn(4, 5, 20)
    with pytest.raises(ValueError, match="seq_len=1"):
        model(x)


def test_lstm_custom_hidden_size():
    model = LSTMClassifier(input_size=20, output_dim=2, hidden_size=64, num_layers=2)
    assert model.hidden_size == 64
    assert model.num_layers == 2
    x = torch.randn(4, 20)
    y = model(x)
    assert y.shape == (4, 2)


def test_lstm_predict_returns_ndarray():
    model = LSTMClassifier(input_size=20, output_dim=2)
    model.eval()
    X = np.random.randn(10, 20)
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (10,)


def test_train_lstm_binary_returns_metrics(small_binary):
    X, y = small_binary
    X_train, X_val = X.iloc[:800], X.iloc[800:]
    y_train, y_val = y.iloc[:800], y.iloc[800:]

    result = train_lstm_binary(X_train, y_train, X_val, y_val, epochs=2, verbose=False)
    assert "f1" in result.val_metrics
    assert "accuracy" in result.val_metrics
    for key in ["accuracy", "precision", "recall", "f1", "auc"]:
        assert 0.0 <= result.val_metrics[key] <= 1.0
