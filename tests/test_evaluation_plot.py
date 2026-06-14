"""src/evaluation/plot.py smoke tests — verify PNG generation."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.plot import (
    plot_confusion_matrix_heatmap,
    plot_roc_curves,
    plot_f1_by_category_bars,
    plot_feature_importance_comparison,
    plot_dl_vs_ml_comparison,
)


@pytest.fixture
def tmp_dir():
    import shutil
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def binary_data():
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    return y_true, y_pred


class TestConfusionMatrixHeatmap:
    def test_generates_png(self, binary_data, tmp_dir):
        y_true, y_pred = binary_data
        path = tmp_dir / "test_cm.png"
        plot_confusion_matrix_heatmap(
            y_true, y_pred, ["Normal", "Anomaly"], "Test CM", str(path)
        )
        assert path.exists()
        assert path.stat().st_size > 500  # non-trivial PNG


class TestROCCurves:
    def test_generates_png(self, tmp_dir):
        roc_data = [
            ("Model A", np.linspace(0, 1, 10), np.linspace(0, 1, 10), 0.85),
            ("Model B", np.linspace(0, 1, 10), np.linspace(0, 0.8, 10), 0.72),
        ]
        path = tmp_dir / "test_roc.png"
        plot_roc_curves(roc_data, str(path))
        assert path.exists()
        assert path.stat().st_size > 500


class TestF1ByCategoryBars:
    def test_generates_png(self, tmp_dir):
        f1_dicts = {
            "Model A": {"DoS": 0.9, "Probe": 0.7, "R2L": 0.3, "U2R": 0.1},
            "Model B": {"DoS": 0.8, "Probe": 0.6, "R2L": 0.2, "U2R": 0.05},
        }
        path = tmp_dir / "test_f1bars.png"
        plot_f1_by_category_bars(f1_dicts, str(path))
        assert path.exists()
        assert path.stat().st_size > 500


class TestFeatureImportanceComparison:
    def test_generates_png(self, tmp_dir):
        import pandas as pd
        dt_imp = pd.Series(
            [0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04],
            index=[f"feat_{i}" for i in range(8)],
        )
        rf_imp = pd.Series(
            [0.14, 0.13, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04],
            index=[f"feat_{i}" for i in range(8)],
        )
        path = tmp_dir / "test_featimp.png"
        plot_feature_importance_comparison(dt_imp, rf_imp, top_k=5, save_path=str(path))
        assert path.exists()
        assert path.stat().st_size > 500


class TestDLvsMLComparison:
    def test_generates_png(self, tmp_dir):
        metrics_dict = {
            "Model A": {"accuracy": 0.95, "f1": 0.94, "auc": 0.98},
            "Model B": {"accuracy": 0.85, "f1": 0.82, "auc": 0.88},
        }
        path = tmp_dir / "test_dlvsml.png"
        plot_dl_vs_ml_comparison(metrics_dict, str(path))
        assert path.exists()
        assert path.stat().st_size > 500

    def test_nested_directory_creation(self, tmp_dir):
        """Verify auto-creation of nested directories."""
        path = tmp_dir / "deep" / "nested" / "dir" / "test.png"
        metrics_dict = {"M": {"accuracy": 0.9, "f1": 0.9, "auc": 0.9}}
        plot_dl_vs_ml_comparison(metrics_dict, str(path))
        assert path.exists()
        assert path.stat().st_size > 500
