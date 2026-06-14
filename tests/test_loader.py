"""Tests for src.data.loader (M1 task 1.6).

TDD Iron Law: these tests are written BEFORE the implementation.
They should FAIL when first run, then PASS after implementation.
"""
import pytest

from src.data import loader as loader_module
from src.data.loader import COLUMN_NAMES, FEATURE_NAMES, load_test, load_train


# ============================================================
# R3: NSL-KDD 41 特征名常量
# ============================================================

def test_feature_names_count():
    """41 个 NSL-KDD 特征名（不含 label 和 difficulty）"""
    assert len(FEATURE_NAMES) == 41


def test_feature_names_contain_classification_columns():
    """包含 3 个分类特征：protocol_type / service / flag"""
    assert "protocol_type" in FEATURE_NAMES
    assert "service" in FEATURE_NAMES
    assert "flag" in FEATURE_NAMES


def test_column_names_count():
    """43 列 = 41 特征 + label + difficulty"""
    assert len(COLUMN_NAMES) == 43


def test_column_names_contain_label_and_difficulty():
    """末尾两列为 label 和 difficulty"""
    assert "label" in COLUMN_NAMES
    assert "difficulty" in COLUMN_NAMES
    # label 和 difficulty 在最后两列（loader 设计）
    assert COLUMN_NAMES[-2] == "label"
    assert COLUMN_NAMES[-1] == "difficulty"


# ============================================================
# R4/R7: Happy Path - 加载成功
# ============================================================

def test_load_train_shape():
    """加载训练集，shape 必须为 (125973, 43)"""
    df = load_train()
    assert df.shape == (125973, 43)


def test_load_test_shape():
    """加载测试集，shape 必须为 (22544, 43)"""
    df = load_test()
    assert df.shape == (22544, 43)


def test_load_train_columns_match():
    """加载后列名匹配 COLUMN_NAMES"""
    df = load_train()
    assert list(df.columns) == COLUMN_NAMES


def test_load_test_columns_match():
    df = load_test()
    assert list(df.columns) == COLUMN_NAMES


# ============================================================
# R10: 无 header 自动命名
# ============================================================

def test_load_train_no_unnamed_columns():
    """无 header 文件不应出现 Unnamed: 0 类默认列名"""
    df = load_train()
    assert not any(col.startswith("Unnamed:") for col in df.columns)


# ============================================================
# R5/R8: Sad Path - 文件不存在
# ============================================================

def test_load_train_missing_file_raises(monkeypatch, tmp_path):
    """当 KDDTrain+.txt 不存在时，抛出 FileNotFoundError"""
    monkeypatch.setattr(loader_module, "DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="KDDTrain\\+\\.txt"):
        load_train()


def test_load_test_missing_file_raises(monkeypatch, tmp_path):
    """当 KDDTest+.txt 不存在时，抛出 FileNotFoundError"""
    monkeypatch.setattr(loader_module, "DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="KDDTest\\+\\.txt"):
        load_test()


# ============================================================
# R9: Latin-1 编码（隐式 - 通过成功加载验证）
# ============================================================

def test_load_train_returns_dataframe():
    """加载成功且返回 DataFrame 类型"""
    df = load_train()
    assert isinstance(df, type(__import__("pandas").DataFrame()))
    assert len(df) > 0