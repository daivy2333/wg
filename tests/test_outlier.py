"""Tests for src.data.outlier (M2 task 2.4).

TDD Iron Law: tests written BEFORE implementation.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.outlier import (
    detect_outliers_iqr,
    clip_outliers,
    log1p_transform,
    outlier_pipeline,
    LONG_TAIL_COLUMNS,
)


# ============================================================
# 测试 fixture
# ============================================================

@pytest.fixture
def sample_df_with_outliers():
    """含明显异常值的合成数据"""
    return pd.DataFrame({
        "normal_col": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],          # 无异常
        "with_outlier_low": [1, 2, 3, 4, 5, 6, 7, 8, 9, -100],   # 低端异常
        "with_outlier_high": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000],  # 高端异常
        "onehot_col": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],            # 0/1 特征
    })


# ============================================================
# R1: IQR 异常值检测
# ============================================================

def test_detect_outliers_iqr_normal_column(sample_df_with_outliers):
    """无异常值的列返回 0"""
    result = detect_outliers_iqr(sample_df_with_outliers, columns=["normal_col"])
    assert result["normal_col"] == 0


def test_detect_outliers_iqr_detects_high_outlier(sample_df_with_outliers):
    """检测高端异常值"""
    result = detect_outliers_iqr(sample_df_with_outliers, columns=["with_outlier_high"])
    assert result["with_outlier_high"] >= 1


def test_detect_outliers_iqr_skips_non_numeric():
    """非数值列跳过（不抛错）"""
    df = pd.DataFrame({"str_col": ["a", "b", "c"]})
    result = detect_outliers_iqr(df, columns=["str_col"])
    assert result["str_col"] == 0  # 跳过 = 0


# ============================================================
# R2: clip 替换
# ============================================================

def test_clip_outliers_replaces_extreme_values(sample_df_with_outliers):
    """clip 替换超限值"""
    result = clip_outliers(sample_df_with_outliers, columns=["with_outlier_high"])
    # 原 1000 应被 clip 到上界（小于 1000）
    assert result["with_outlier_high"].max() < 1000


def test_clip_outliers_preserves_row_count(sample_df_with_outliers):
    """clip 不丢样本（行数不变）"""
    result = clip_outliers(sample_df_with_outliers, columns=["with_outlier_high"])
    assert len(result) == len(sample_df_with_outliers)


def test_clip_outliers_no_nan_introduced(sample_df_with_outliers):
    """clip 不引入 NaN"""
    result = clip_outliers(sample_df_with_outliers, columns=["with_outlier_high"])
    assert not result.isnull().any().any()


# ============================================================
# R3: log1p 变换
# ============================================================

def test_log1p_transform_adds_log_suffix():
    """log1p 变换后列名加 _log1p 后缀"""
    df = pd.DataFrame({"src_bytes": [0, 100, 10000, 1000000]})
    result = log1p_transform(df, columns=["src_bytes"])
    assert "src_bytes_log1p" in result.columns
    assert "src_bytes" in result.columns  # 原列保留


def test_log1p_transform_handles_zero():
    """log1p(0) = 0，不产生 -inf 或 NaN"""
    df = pd.DataFrame({"src_bytes": [0, 100, 10000]})
    result = log1p_transform(df, columns=["src_bytes"])
    assert result["src_bytes_log1p"].iloc[0] == 0
    assert not result["src_bytes_log1p"].isnull().any()


def test_log1p_transform_default_long_tail_columns():
    """默认 LONG_TAIL_COLUMNS 包含 src_bytes/dst_bytes/count/srv_count"""
    assert "src_bytes" in LONG_TAIL_COLUMNS
    assert "dst_bytes" in LONG_TAIL_COLUMNS
    assert "count" in LONG_TAIL_COLUMNS
    assert "srv_count" in LONG_TAIL_COLUMNS


# ============================================================
# R4: 整合管线
# ============================================================

def test_outlier_pipeline_returns_processed_df_and_stats(sample_df_with_outliers):
    """整合函数返回 (DataFrame, stats_dict)"""
    result_df, stats = outlier_pipeline(
        sample_df_with_outliers,
        numeric_cols=["normal_col", "with_outlier_high"],
        long_tail_cols=["with_outlier_high"],
    )
    assert isinstance(result_df, pd.DataFrame)
    assert isinstance(stats, dict)
    assert "with_outlier_high" in stats  # IQR 异常值数
    assert "with_outlier_high_log1p" in result_df.columns  # log1p 新列


def test_outlier_pipeline_preserves_rows(sample_df_with_outliers):
    """管线不丢样本"""
    result_df, _ = outlier_pipeline(
        sample_df_with_outliers,
        numeric_cols=["normal_col"],
        long_tail_cols=[],
    )
    assert len(result_df) == len(sample_df_with_outliers)