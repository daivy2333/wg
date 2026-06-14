"""Tests for src.data.persistence (M2 task 2.9).

TDD Iron Law: tests written BEFORE implementation.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.persistence import (
    save_pickle,
    load_pickle,
    save_m2_datasets,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})


@pytest.fixture
def sample_series():
    return pd.Series([0, 1, 0, 1], name="label")


# ============================================================
# R8: pickle 持久化
# ============================================================

def test_save_pickle_creates_file(sample_df, tmp_path):
    """save_pickle 创建文件"""
    path = tmp_path / "test.pkl"
    save_pickle(sample_df, path)
    assert path.exists()


def test_load_pickle_roundtrip(sample_df, tmp_path):
    """save + load 还原对象"""
    path = tmp_path / "test.pkl"
    save_pickle(sample_df, path)
    loaded = load_pickle(path)
    pd.testing.assert_frame_equal(loaded, sample_df)


def test_load_pickle_missing_file_raises(tmp_path):
    """文件不存在抛 FileNotFoundError"""
    with pytest.raises(FileNotFoundError, match="missing.pkl"):
        load_pickle(tmp_path / "missing.pkl")


def test_save_pickle_creates_parent_dirs(sample_df, tmp_path):
    """自动创建中间目录"""
    nested_path = tmp_path / "a" / "b" / "c" / "test.pkl"
    save_pickle(sample_df, nested_path)
    assert nested_path.exists()


# ============================================================
# R9: M2 数据集持久化（双套）
# ============================================================

def test_save_m2_datasets_binary(sample_df, sample_series, tmp_path):
    """二分类任务保存 4 个文件（不带 _multi 后缀）"""
    save_m2_datasets(
        sample_df, sample_df, sample_series, sample_series,
        output_dir=tmp_path, task="binary",
    )
    assert (tmp_path / "X_train.pkl").exists()
    assert (tmp_path / "X_test.pkl").exists()
    assert (tmp_path / "y_train.pkl").exists()
    assert (tmp_path / "y_test.pkl").exists()


def test_save_m2_datasets_multiclass(sample_df, sample_series, tmp_path):
    """多分类任务保存 4 个文件（带 _multi 后缀）"""
    save_m2_datasets(
        sample_df, sample_df, sample_series, sample_series,
        output_dir=tmp_path, task="multiclass",
    )
    assert (tmp_path / "X_train_multi.pkl").exists()
    assert (tmp_path / "X_test_multi.pkl").exists()
    assert (tmp_path / "y_train_multi.pkl").exists()
    assert (tmp_path / "y_test_multi.pkl").exists()


def test_save_m2_datasets_loads_correctly(sample_df, sample_series, tmp_path):
    """保存后可加载还原"""
    save_m2_datasets(
        sample_df, sample_df, sample_series, sample_series,
        output_dir=tmp_path, task="binary",
    )
    X_train_loaded = load_pickle(tmp_path / "X_train.pkl")
    y_train_loaded = load_pickle(tmp_path / "y_train.pkl")
    pd.testing.assert_frame_equal(X_train_loaded, sample_df)
    pd.testing.assert_series_equal(y_train_loaded, sample_series)


def test_save_m2_datasets_invalid_task_raises(sample_df, sample_series, tmp_path):
    """未知 task 抛 ValueError"""
    with pytest.raises(ValueError, match="未知 task"):
        save_m2_datasets(
            sample_df, sample_df, sample_series, sample_series,
            output_dir=tmp_path, task="invalid",
        )