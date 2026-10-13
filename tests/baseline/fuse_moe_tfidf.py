"""MoE × TF-IDF 融合实验（读逐样本概率，直接做概率级融合与显著性检验）

用法:

    python -m tests.baseline.fuse_moe_tfidf                      # 默认用干净测试集
    python -m tests.baseline.fuse_moe_tfidf --variant homoglyph   # 指定扰动变体

说明：等权融合（0.5/0.5）是不调参的主结果；权重网格仅作敏感性参考，**不要**把网格
最优值当作可发表结论（那是在测试集上调参）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tests.baseline.common import label_names, write_json
from train.common.metrics import (
    bootstrap_macro_f1_ci,
    expected_calibration_error,
    mcnemar_test,
    macro_f1,
    paired_bootstrap_f1_diff,
    per_class_prf,
)
from train.config.config import Config, load_config


def evaluate(y_true: np.ndarray, proba: np.ndarray, names: list[str], n_boot: int) -> dict:
    y_pred = proba.argmax(axis=1).tolist()
    correct = (np.asarray(y_pred) == y_true).astype(float)
    return {
        "accuracy": float(correct.mean()),
        "macro_f1": macro_f1(y_true.tolist(), y_pred),
        "per_class": per_class_prf(y_true.tolist(), y_pred, names),
        "ece": expected_calibration_error(proba.max(axis=1), correct),
        "macro_f1_bootstrap": bootstrap_macro_f1_ci(y_true.tolist(), y_pred, n_boot=n_boot),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", default="clean")
    parser.add_argument("--n-boot", dest="n_boot", type=int, default=1000)
    parser.add_argument("--out", default="artifacts/results/fusion.json")
    args = parser.parse_args(argv)

    src = Path(f"artifacts/results/preds_{args.variant}.npz")
    if not src.exists():
        raise SystemExit(f"找不到逐样本预测文件：{src}（先跑 tests.baseline.eval_perturbed）")
    d = np.load(src)
    y_true = d["y_true"]
    p_moe, p_tf = d["moe_proba"], d["tfidf_proba"]
    cfg = load_config(Config, toml_path="config.toml", argv=[])
    names = label_names(cfg)

    print(f"[FUSION] 变体={args.variant}  样本={len(y_true):,}  类别={p_moe.shape[1]}")
    report: dict = {"variant": args.variant, "n_test": int(len(y_true)), "models": {}}

    def show(label: str, res: dict) -> None:
        ci = res["macro_f1_bootstrap"]
        print(f"[FUSION] {label:<18} accuracy={res['accuracy']:.4f} macro_f1={res['macro_f1']:.4f} "
              f"CI95=[{ci['ci95_low']:.4f}, {ci['ci95_high']:.4f}] ece={res['ece']:.4f}")

    base_moe = evaluate(y_true, p_moe, names, args.n_boot)
    base_tf = evaluate(y_true, p_tf, names, args.n_boot)
    report["models"]["moe_gated"] = base_moe
    report["models"]["tfidf_lr"] = base_tf
    show("moe_gated", base_moe)
    show("tfidf_lr", base_tf)

    # 等权融合：不调参的主结果
    equal = evaluate(y_true, (p_moe + p_tf) / 2.0, names, args.n_boot)
    report["models"]["fusion_equal"] = equal
    show("fusion_equal", equal)

    # 几何平均（对数概率平均）：对过度自信的一方更稳健
    geo = evaluate(y_true, np.exp((np.log(p_moe + 1e-12) + np.log(p_tf + 1e-12)) / 2.0), names, args.n_boot)
    report["models"]["fusion_geometric"] = geo
    show("fusion_geometric", geo)

    # 权重敏感性（探索性，请勿当作可发表结论）
    grid = {}
    for w in [round(0.1 * i, 1) for i in range(1, 10)]:
        grid[str(w)] = evaluate(y_true, w * p_moe + (1 - w) * p_tf, names, 200)["macro_f1"]
    report["weight_grid_exploratory"] = grid
    best_w = max(grid, key=grid.get)
    print(f"[FUSION] 权重网格（仅探索）：best w={best_w} → macro_f1={grid[best_w]:.4f}"
          f"（w 为 MoE 权重；等权 w=0.5 → {grid['0.5']:.4f}）")

    # 显著性：融合 vs 两个单模型（McNemar + 配对 bootstrap）
    for label, res in (("moe_gated", base_moe), ("tfidf_lr", base_tf)):
        y_pred_main = (p_moe + p_tf).argmax(axis=1).tolist()
        y_pred_other = (p_moe if label == "moe_gated" else p_tf).argmax(axis=1).tolist()
        report.setdefault("significance", {})[f"fusion_equal_vs_{label}"] = {
            "mcnemar": mcnemar_test(y_true.tolist(), y_pred_other, y_pred_main),
            "paired_f1_diff": paired_bootstrap_f1_diff(
                y_true.tolist(), y_pred_other, y_pred_main, n_boot=args.n_boot
            ),
        }
        sig = report["significance"][f"fusion_equal_vs_{label}"]
        print(f"[FUSION] 等权融合 vs {label:<10} McNemar p={sig['mcnemar']['p_value']:.3g}  "
              f"ΔF1={sig['paired_f1_diff']['mean']:+.5f} "
              f"CI95=[{sig['paired_f1_diff']['ci95_low']:.5f}, {sig['paired_f1_diff']['ci95_high']:.5f}]")

    write_json(args.out, report)
    print(f"[FUSION] 结果已保存到 {args.out}")


if __name__ == "__main__":
    main()
