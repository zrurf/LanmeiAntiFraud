"""对比实验与消融实验驱动

在同一测试集上跑齐三条基线与 MoE 的三种融合方式，输出统一对比报告：

    tfidf_nb                TF-IDF + 朴素贝叶斯
    tfidf_lr                TF-IDF + 逻辑回归
    macbert_dora_single     MacBERT + DoRA 单头
    moe_gated               MoE 门控融合（正式）
    moe_equal               MoE 等权融合（消融 A1）
    moe_top1                MoE top-1 硬选择（消融 A2）

所有配置共用同一测试集与指标口径，并以 MoE 门控为基准做 McNemar 精确
检验与配对 bootstrap Macro-F1 差值区间。缺少对应权重的配置会被跳过

用法:

    python -m tests.baseline.compare [--max-train N] [--max-features N] \
        [--n-boot N] [--out artifacts/results/compare.json] [配置项覆盖]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tests.baseline.common import (
    FUSIONS,
    label_names,
    load_checkpoint,
    load_split,
    make_loader,
    predict,
    score,
    write_json,
)
from tests.baseline.eval_testset import DEFAULT_STAGE
from tests.baseline.tfidf import run_tfidf
from train.common.metrics import mcnemar_test, paired_bootstrap_f1_diff
from train.config.config import Config, load_config
from train.utils.env import get_device

CONFIG_PATH = "config.toml"

# 显著性检验基准（正式配置）
REFERENCE = "moe_gated"


def resolve_checkpoint(cfg: Config, mode: str) -> Path | None:
    root = Path(cfg.output.export_path)
    for stage in DEFAULT_STAGE[mode]:
        candidate = root / stage / "model.pt"
        if candidate.exists():
            return candidate
    return None


def _print_table(rows: dict[str, dict]) -> None:
    print(f"{'config':<24}{'accuracy':>10}{'macro_f1':>10}{'ece':>10}", flush=True)
    for name, metrics in rows.items():
        print(
            f"{name:<24}{metrics['accuracy']:>10.4f}"
            f"{metrics['macro_f1']:>10.4f}{metrics['ece']:>10.4f}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max-train", dest="max_train", type=int, default=0)
    parser.add_argument("--max-features", dest="max_features", type=int, default=0)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    parser.add_argument("--n-boot", dest="n_boot", type=int, default=1000)
    parser.add_argument("--out", default="artifacts/results/compare.json")
    args, rest = parser.parse_known_args(argv)

    cfg = load_config(Config, toml_path=CONFIG_PATH, argv=rest)
    device = get_device()
    names = label_names(cfg)

    rows: dict[str, dict] = {}
    preds: dict[str, list[int]] = {}
    y_true: list[int] | None = None

    tfidf_report, y_true, tfidf_preds = run_tfidf(
        cfg, max_train=args.max_train, max_features=args.max_features, n_boot=args.n_boot
    )
    rows.update(tfidf_report["results"])
    preds.update(tfidf_preds)

    # MacBERT + DoRA 单头基线，与 MoE 消融共用测试集加载器
    for mode, label in (("single", "macbert_dora_single"), ("moe", None)):
        checkpoint = resolve_checkpoint(cfg, mode)
        if checkpoint is None:
            print(f"[COMPARE] 跳过 {mode}：未找到对应权重", flush=True)
            continue
        model = load_checkpoint(cfg, checkpoint, device)
        loader = make_loader(cfg, load_split(cfg, "test"), model.backbone.tokenizer, args.batch_size)
        for fusion in (FUSIONS if mode == "moe" else ("gated",)):
            name = label or f"moe_{fusion}"
            yt, yp, conf = predict(model, loader, mode, fusion, device)
            rows[name] = score(yt, yp, conf, names, n_boot=args.n_boot)
            preds[name] = yp
            y_true = yt

    # 显著性：各配置 vs MoE 门控
    significance = {}
    if y_true is not None and REFERENCE in preds:
        for name, yp in preds.items():
            if name == REFERENCE:
                continue
            significance[name] = {
                "mcnemar_vs_reference": mcnemar_test(y_true, yp, preds[REFERENCE]),
                "paired_f1_diff_reference_minus_this": paired_bootstrap_f1_diff(
                    y_true, yp, preds[REFERENCE], n_boot=args.n_boot
                ),
            }

    report = {
        "label_names": names,
        "n_test": len(y_true) if y_true is not None else 0,
        "reference": REFERENCE,
        "rows": rows,
        "significance": significance,
    }
    write_json(args.out, report)
    _print_table(rows)
    print(f"[COMPARE] 显著性基准：{REFERENCE}；结果已保存到 {args.out}", flush=True)


if __name__ == "__main__":
    main()