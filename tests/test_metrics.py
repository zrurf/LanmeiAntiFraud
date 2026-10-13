"""评估指标单元测试：Macro-F1 / 逐类 P-R-F1 / ECE / 混淆矩阵 / McNemar / bootstrap"""

from __future__ import annotations

import numpy as np

from train.common.metrics import (
    bootstrap_macro_f1_ci,
    confusion_matrix,
    expected_calibration_error,
    macro_f1,
    mcnemar_test,
    paired_bootstrap_f1_diff,
    per_class_prf,
)


def test_macro_f1_perfect_and_partial():
    assert macro_f1([0, 1, 2], [0, 1, 2]) == 1.0
    assert macro_f1([0, 1, 2], [0, 0, 2]) < 1.0


def test_per_class_prf_covers_full_label_table():
    prf = per_class_prf([0, 0], [0, 1], ["正常", "诈骗", "刷单"])
    assert list(prf) == ["正常", "诈骗", "刷单"]
    # 未出现的类别仍按完整标签表输出，数值为 0
    assert prf["刷单"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert prf["正常"]["precision"] == 1.0


def test_ece_is_zero_for_calibrated_and_positive_for_overconfident():
    conf = np.array([1.0, 1.0, 1.0])
    assert expected_calibration_error(conf, np.array([1, 1, 1])) == 0.0
    over = expected_calibration_error(np.array([0.9]), np.array([0]))
    assert over == 0.9


def test_confusion_matrix_counts():
    cm = confusion_matrix([0, 1, 1], [0, 1, 0], 2)
    assert cm == [[1, 0], [1, 1]]


def test_mcnemar_detects_direction():
    y_true = [0, 0, 0, 0]
    pred_a = [0, 0, 0, 0]
    pred_b = [1, 1, 1, 1]
    result = mcnemar_test(y_true, pred_a, pred_b)
    assert result["n01"] == 4
    assert result["n10"] == 0
    assert result["p_value"] < 0.2
    # 两者完全一致的退化情形
    assert mcnemar_test(y_true, pred_a, pred_a) == {"n01": 0, "n10": 0, "p_value": 1.0}


def test_bootstrap_macro_f1_interval_brackets_mean():
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    y_pred = [0, 1, 2, 0, 2, 2, 0, 1, 1, 0]
    ci = bootstrap_macro_f1_ci(y_true, y_pred, n_boot=50, seed=0)
    assert ci["ci95_low"] <= ci["mean"] <= ci["ci95_high"]


def test_paired_bootstrap_diff_sign():
    y_true = [0, 1, 0, 1, 0, 1]
    pred_a = [0, 0, 0, 0, 0, 0]
    pred_b = [0, 1, 0, 1, 0, 1]
    diff = paired_bootstrap_f1_diff(y_true, pred_a, pred_b, n_boot=50, seed=0)
    assert diff["mean"] > 0  # B 优于 A