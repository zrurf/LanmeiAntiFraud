"""测试集正式评估：Accuracy / Macro-F1 / 逐类 P-R-F1 / 混淆矩阵 / ECE / Macro-F1 bootstrap 置信区间

支持两种模式：

    single  MacBERT + DoRA 单头基线（`clf_forward`）
    moe     多专家门控融合，`--fusion gated|equal|top1` 在推理期切换融合方式，
            equal / top1 即消融 A1（等权）/ A2（top-1 硬选择），无需重训

`--compare <另一 checkpoint>` 时在同一测试集上追加 McNemar 精确检验与配对
bootstrap Macro-F1 差值区间

用法:

    python -m tests.baseline.eval_testset --checkpoint artifacts/export/joint/model.pt \
        --mode moe --fusion gated --out artifacts/results/moe_gated.json
    python -m tests.baseline.eval_testset --checkpoint artifacts/export/joint/model.pt \
        --mode moe --compare artifacts/export/clf/model.pt --compare-mode single \
        --out artifacts/results/moe_vs_single.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tests.baseline.common import MODES, FUSIONS, evaluate, write_json
from train.common.metrics import mcnemar_test, paired_bootstrap_f1_diff
from train.config.config import Config, load_config
from train.utils.env import get_device

CONFIG_PATH = "config.toml"

# 各模式默认权重：单头基线取 clf，MoE 优先取联合训练产物
DEFAULT_STAGE = {"single": ("clf",), "moe": ("joint", "router")}


def default_checkpoint(cfg: Config, mode: str) -> Path:
    root = Path(cfg.output.export_path)
    for stage in DEFAULT_STAGE[mode]:
        candidate = root / stage / "model.pt"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{root} 下未找到 {mode} 模式可用权重，候选阶段：{' → '.join(DEFAULT_STAGE[mode])}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode", choices=list(MODES), default="moe")
    parser.add_argument("--fusion", choices=list(FUSIONS), default="gated")
    parser.add_argument("--compare", default=None)
    parser.add_argument("--compare-mode", dest="compare_mode", choices=list(MODES), default="single")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    parser.add_argument("--n-boot", dest="n_boot", type=int, default=1000)
    parser.add_argument("--out", default="artifacts/results/eval_testset.json")
    args, rest = parser.parse_known_args(argv)

    cfg = load_config(Config, toml_path=CONFIG_PATH, argv=rest)
    device = get_device()

    checkpoint = Path(args.checkpoint) if args.checkpoint else default_checkpoint(cfg, args.mode)
    result = evaluate(cfg, checkpoint, args.mode, args.fusion, args.batch_size, args.n_boot, device)
    y_true = result.pop("_y_true")
    y_pred_main = result.pop("_y_pred")

    if args.compare:
        compare_ckpt = Path(args.compare)
        other = evaluate(
            cfg, compare_ckpt, args.compare_mode, "gated", args.batch_size, args.n_boot, device
        )
        y_pred_other = other.pop("_y_pred")
        other.pop("_y_true")
        result["compare"] = {
            "checkpoint": str(compare_ckpt),
            "mode": args.compare_mode,
            **other,
            "mcnemar_vs_main": mcnemar_test(y_true, y_pred_other, y_pred_main),
            "paired_f1_diff_main_minus_compare": paired_bootstrap_f1_diff(
                y_true, y_pred_other, y_pred_main, n_boot=args.n_boot
            ),
        }

    write_json(args.out, result)
    print(
        f"[EVAL] mode={result['mode']} fusion={result['fusion']} "
        f"accuracy={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f} "
        f"CI95=[{result['macro_f1_bootstrap']['ci95_low']:.4f}, "
        f"{result['macro_f1_bootstrap']['ci95_high']:.4f}] ece={result['ece']:.4f}",
        flush=True,
    )
    print(f"[EVAL] 结果已保存到 {args.out}", flush=True)


if __name__ == "__main__":
    main()