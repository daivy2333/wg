"""Tests for src.models.persistence torch.save/load (M4 task 4.7)."""
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.cnn import CNN1DClassifier
from src.models.lstm import LSTMClassifier
from src.models.mlp import MLPClassifier
from src.models.persistence import (
    load_torch_model,
    save_best_nn_models,
    save_torch_model,
)


@pytest.fixture
def tmp_models_dir(tmp_path):
    return tmp_path / "models"


def test_save_torch_model_creates_file(tmp_models_dir):
    """save_torch_model 创建 .pt 文件。"""
    model = MLPClassifier(input_dim=20, output_dim=2)
    save_torch_model(model, tmp_models_dir / "test_model.pt")
    assert (tmp_models_dir / "test_model.pt").exists()
    assert (tmp_models_dir / "test_model.pt").stat().st_size > 0


def test_save_load_roundtrip_consistency(tmp_models_dir):
    """save → load 后预测结果一致。"""
    model = MLPClassifier(input_dim=20, output_dim=2)
    model.eval()
    X = np.random.randn(5, 20).astype(np.float32)
    preds_before = model.predict(X)

    save_torch_model(model, tmp_models_dir / "mlp_test.pt")
    loaded = load_torch_model(
        MLPClassifier, tmp_models_dir / "mlp_test.pt", input_dim=20, output_dim=2
    )
    preds_after = loaded.predict(X)
    np.testing.assert_array_equal(preds_before, preds_after)


def test_load_torch_model_file_not_found():
    """加载不存在的文件抛 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError, match="模型文件不存在"):
        load_torch_model(MLPClassifier, "/nonexistent/path/model.pt", input_dim=20, output_dim=2)


def test_load_torch_model_architecture_mismatch(tmp_models_dir):
    """架构不匹配抛 RuntimeError。"""
    model = MLPClassifier(input_dim=20, output_dim=2)
    save_torch_model(model, tmp_models_dir / "mismatch.pt")
    # 尝试用 output_dim=23 加载（输出维度不匹配）
    with pytest.raises(RuntimeError, match="加载 state_dict 失败"):
        load_torch_model(
            MLPClassifier, tmp_models_dir / "mismatch.pt", input_dim=20, output_dim=23
        )


def test_save_load_cnn(tmp_models_dir):
    """CNN save/load 一致性。"""
    model = CNN1DClassifier(input_length=20, output_dim=2)
    model.eval()
    X = np.random.randn(5, 20).astype(np.float32)
    preds_before = model.predict(X)

    save_torch_model(model, tmp_models_dir / "cnn_test.pt")
    loaded = load_torch_model(
        CNN1DClassifier, tmp_models_dir / "cnn_test.pt", input_length=20, output_dim=2
    )
    preds_after = loaded.predict(X)
    np.testing.assert_array_equal(preds_before, preds_after)


def test_save_load_lstm(tmp_models_dir):
    """LSTM save/load 一致性。"""
    model = LSTMClassifier(input_size=20, output_dim=2)
    model.eval()
    X = np.random.randn(5, 20).astype(np.float32)
    preds_before = model.predict(X)

    save_torch_model(model, tmp_models_dir / "lstm_test.pt")
    loaded = load_torch_model(
        LSTMClassifier, tmp_models_dir / "lstm_test.pt", input_size=20, output_dim=2
    )
    preds_after = loaded.predict(X)
    np.testing.assert_array_equal(preds_before, preds_after)


def test_save_best_nn_models_creates_dir(tmp_models_dir):
    """save_best_nn_models 自动创建目录。"""
    assert not tmp_models_dir.exists()
    save_best_nn_models(tmp_models_dir)  # 全部 None，应该只创建目录
    assert tmp_models_dir.exists()
    # 目录下没有 .pt 文件（全部 None）
    assert list(tmp_models_dir.glob("*.pt")) == []


def test_save_best_nn_models_saves_provided(tmp_models_dir, capsys):
    """只保存提供的模型，未提供的跳过。"""
    mlp_bin = MLPClassifier(input_dim=20, output_dim=2)
    cnn_bin = CNN1DClassifier(input_length=20, output_dim=2)
    save_best_nn_models(
        tmp_models_dir,
        mlp_binary=mlp_bin,
        cnn_binary=cnn_bin,
    )
    saved = list(tmp_models_dir.glob("*.pt"))
    assert len(saved) == 2
    assert any(p.name == "mlp_binary_best.pt" for p in saved)
    assert any(p.name == "cnn_binary_best.pt" for p in saved)
    # 未提供的会打印 skip
    captured = capsys.readouterr()
    assert "skip" in captured.out
