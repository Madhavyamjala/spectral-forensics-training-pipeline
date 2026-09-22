"""
Regression tests for subset-mode evaluation probability handling.

Note:
    These tests cover the failure mode where a sklearn baseline is trained on fewer than the
    canonical tri-class labels and therefore returns fewer probability columns.

TODO:
    Add integration coverage that executes evaluate_ablation with a two-class test fixture.
"""

from __future__ import annotations

import numpy as np

from csf.eval.ablation import _align_probs_to_labels
from csf.eval.metrics import classification_report_dict


def test_align_probs_to_labels_inserts_missing_class() -> None:
    probs = np.array([[0.25, 0.75], [0.80, 0.20]], dtype=np.float64)
    classes = np.array([0, 1], dtype=int)
    aligned = _align_probs_to_labels(probs, classes)
    assert aligned.shape == (2, 3)
    np.testing.assert_allclose(aligned[:, :2], probs)
    np.testing.assert_array_equal(aligned[:, 2], np.zeros(2))


def test_two_class_report_does_not_crash_on_log_loss() -> None:
    y = np.array([0, 1, 0, 1], dtype=int)
    probs = np.array(
        [
            [0.90, 0.10, 0.0],
            [0.10, 0.90, 0.0],
            [0.80, 0.20, 0.0],
            [0.20, 0.80, 0.0],
        ],
        dtype=np.float64,
    )
    report = classification_report_dict(y, probs)
    assert report["log_loss"] is not None
    assert report["n"] == 4
