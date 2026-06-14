"""src/evaluation/metrics.py 单元测试"""

import numpy as np
import pytest
from src.evaluation.metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    compute_f1_by_category,
)


class TestBinaryMetrics:
    def test_perfect_prediction(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8])
        m = compute_binary_metrics(y_true, y_pred, y_prob)
        assert m["accuracy"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["auc"] == 1.0

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])
        m = compute_binary_metrics(y_true, y_pred, y_prob)
        assert m["accuracy"] == 0.0
        assert m["f1"] == 0.0  # zero_division=0

    def test_returns_all_five_keys(self):
        m = compute_binary_metrics([0, 1], [0, 1], [0.2, 0.8])
        assert set(m.keys()) == {"accuracy", "precision", "recall", "f1", "auc"}


class TestMulticlassMetrics:
    def test_perfect_prediction(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        m = compute_multiclass_metrics(y_true, y_pred)
        assert m["accuracy"] == 1.0
        assert m["known_class_accuracy"] == 1.0

    def test_unseen_class_filtering(self):
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        m = compute_multiclass_metrics(y_true, y_pred, unseen_ids=(3, 4))
        # Classes 3,4 are filtered from known accuracy
        assert m["known_class_accuracy"] == 1.0  # 0,1,2 perfectly predicted
        assert m["accuracy"] == 1.0  # full also perfect
        assert "f1_macro" in m

    def test_mixed_accuracy(self):
        y_true = np.array([0, 0, 1, 1, 2])
        y_pred = np.array([0, 1, 1, 1, 2])
        m = compute_multiclass_metrics(y_true, y_pred, unseen_ids=(2,))
        assert abs(m["accuracy"] - 0.8) < 0.01
        assert abs(m["known_class_accuracy"] - 0.75) < 0.01


class TestF1ByCategory:
    def test_basic_categories(self):
        label_map = {0: "DoS", 1: "DoS", 2: "Probe", 3: "Probe"}
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3])
        m = compute_f1_by_category(y_true, y_pred, label_map)
        assert m["DoS"] == 1.0
        assert m["Probe"] == 1.0

    def test_imperfect_prediction(self):
        label_map = {0: "Normal", 1: "Attack"}
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        m = compute_f1_by_category(y_true, y_pred, label_map)
        # Normal: TP=1, FP=1, FN=1 → f1=0.5
        # Attack: TP=1, FP=1, FN=1 → f1=0.5
        assert abs(m["Normal"] - 0.5) < 0.01
        assert abs(m["Attack"] - 0.5) < 0.01
