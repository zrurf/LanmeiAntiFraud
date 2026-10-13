"""评估指标：Macro-F1、逐类 P-R-F1、ECE、混淆矩阵、McNemar 检验与 bootstrap 置信区间
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score, precision_recall_fscore_support


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def per_class_prf(
    y_true: list[int], y_pred: list[int], label_names: list[str]
) -> dict[str, dict[str, float]]:
    # 显式传入 labels：测试集只覆盖部分类别时，仍按完整标签表输出
    labels = list(range(len(label_names)))
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        label_names[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
        }
        for i in range(len(label_names))
    }


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> float:
    """ECE = Σ_k |B_k|/N · |acc(B_k) − conf(B_k)|"""
    conf = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for k in range(n_bins):
        mask = (conf > edges[k]) & (conf <= edges[k + 1])
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def mcnemar_test(
    y_true: list[int], pred_a: list[int], pred_b: list[int]
) -> dict[str, float]:
    """模型 A / B 在同一测试集上的差异显著性（精确二项检验）

    仅统计两者预测不一致的样本：n01 = A 对 B 错，n10 = A 错 B 对
    """
    y = np.asarray(y_true)
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)
    n01 = int(((a == y) & (b != y)).sum())
    n10 = int(((a != y) & (b == y)).sum())
    n = n01 + n10
    if n == 0:
        return {"n01": n01, "n10": n10, "p_value": 1.0}
    return {"n01": n01, "n10": n10, "p_value": float(binomtest(n01, n, 0.5).pvalue)}


def confusion_matrix(
    y_true: list[int], y_pred: list[int], num_classes: int
) -> list[list[int]]:
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1
    return cm


def bootstrap_macro_f1_ci(
    y_true: list[int], y_pred: list[int], n_boot: int = 1000, seed: int = 42
) -> dict[str, float]:
    """测试集 Macro-F1 的 bootstrap 95% 置信区间"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    n = len(y)
    scores = [
        macro_f1(y[idx].tolist(), p[idx].tolist())
        for idx in (rng.integers(0, n, size=n) for _ in range(n_boot))
    ]
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return {"mean": float(np.mean(scores)), "ci95_low": float(lo), "ci95_high": float(hi)}


def paired_bootstrap_f1_diff(
    y_true: list[int],
    pred_a: list[int],
    pred_b: list[int],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """配对 bootstrap：Macro-F1(B) − Macro-F1(A) 的 95% 区间（A=对照，B=主模型）"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(
            macro_f1(y[idx].tolist(), b[idx].tolist())
            - macro_f1(y[idx].tolist(), a[idx].tolist())
        )
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean": float(np.mean(diffs)), "ci95_low": float(lo), "ci95_high": float(hi)}