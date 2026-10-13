"""融合模型的温度标定实验：能否在保持精度的同时修复校准？

做法：
1. 在**验证集**上取两个模型的概率（MoE 需 GPU 前向，TF-IDF 走缓存），拟合使 NLL 最小的 T；
2. 在**测试集**上（读 `artifacts/results/preds_clean.npz`）应用该 T，报告 ECE 与精度变化。

用法:

    python -m tests.baseline.calibrate_fusion --checkpoint artifacts/export/joint/model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch

from tests.baseline.common import label_names, load_checkpoint, load_split, write_json
from tests.baseline.eval_perturbed import TFIDF_CACHE, TextListDataset
from train.common.metrics import expected_calibration_error, macro_f1
from train.config.config import Config, load_config
from train.data.dataset import make_collate_fn, make_loader
from train.utils.env import get_device

CONFIG_PATH = "config.toml"


@torch.no_grad()
def moe_probs(cfg: Config, model, texts: list[str], batch_size: int, device) -> np.ndarray:
    """MoE 门控融合的逐样本概率（顺序加载，保证与输入严格对齐）"""
    items = [(t, 0) for t in texts]
    loader = make_loader(
        TextListDataset(items, cfg.data.max_length),
        make_collate_fn(model.backbone.tokenizer, cfg.data.max_length),
        batch_size, train=False, bucketing=False, num_workers=0,
    )
    out_list = []
    for batch in loader:
        out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        out_list.append(torch.softmax(out["logits"].float(), dim=-1).cpu().numpy())
    return np.concatenate(out_list)


def fit_temperature(y: np.ndarray, logits: np.ndarray, lo: float = 0.2, hi: float = 20.0) -> float:
    """一维搜索使 NLL 最小的温度（网格 + 细化，避免引入优化器依赖）"""
    def nll(t: float) -> float:
        z = logits / t
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-logp[np.arange(len(y)), y].mean())

    grid = np.linspace(lo, hi, 60)
    best = min(grid, key=nll)
    for span in (0.5, 0.1, 0.02):
        grid = np.linspace(max(lo, best - span), min(hi, best + span), 40)
        best = min(grid, key=nll)
    return float(best)


def report(label: str, y: np.ndarray, proba: np.ndarray) -> dict:
    correct = (proba.argmax(axis=1) == y).astype(float)
    return {
        "accuracy": float(correct.mean()),
        "macro_f1": macro_f1(y.tolist(), proba.argmax(axis=1).tolist()),
        "ece": expected_calibration_error(proba.max(axis=1), correct),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", default="artifacts/export/joint/model.pt")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    parser.add_argument("--rule", choices=("arithmetic", "geometric"), default="arithmetic")
    parser.add_argument("--out", default="artifacts/results/calibrate_fusion.json")
    args, rest = parser.parse_known_args(argv)

    cfg = load_config(Config, toml_path=CONFIG_PATH, argv=rest)
    device = get_device()
    names = label_names(cfg)

    cache = Path(TFIDF_CACHE)
    if not cache.exists():
        raise SystemExit(f"缺少 TF-IDF 缓存 {cache}（先跑 eval_perturbed --fit-only）")
    vectorizer, clf = joblib.load(cache)

    # ---- 验证集：拟合温度 ----
    val = load_split(cfg, "val")
    y_val = np.asarray(val.labels.tolist())
    model = load_checkpoint(cfg, Path(args.checkpoint), device)
    p_moe_val = moe_probs(cfg, model, val.texts, args.batch_size, device)
    p_tf_val = clf.predict_proba(vectorizer.transform(val.texts))
    p_val = (p_moe_val + p_tf_val) / 2.0 if args.rule == "arithmetic" else \
        np.exp((np.log(p_moe_val + 1e-12) + np.log(p_tf_val + 1e-12)) / 2.0)

    logits_val = np.log(p_val + 1e-12)
    t_star = fit_temperature(y_val, logits_val)
    print(f"[CAL] 规则={args.rule}  验证集拟合温度 T* = {t_star:.3f}", flush=True)
    for label, p in (("融合未标定", p_val),
                     ("融合已标定", _softmax(logits_val / t_star))):
        r = report(label, y_val, p)
        print(f"[CAL] (val) {label:<10} acc={r['accuracy']:.4f} macro_f1={r['macro_f1']:.4f} ece={r['ece']:.4f}",
              flush=True)

    # ---- 测试集：读已有逐样本概率，只应用温度 ----
    npz = Path("artifacts/results/preds_clean.npz")
    if not npz.exists():
        raise SystemExit(f"缺少 {npz}（先跑 eval_perturbed）")
    d = np.load(npz)
    y_test = d["y_true"]
    p_fused_test = (d["moe_proba"] + d["tfidf_proba"]) / 2.0 if args.rule == "arithmetic" else \
        np.exp((np.log(d["moe_proba"] + 1e-12) + np.log(d["tfidf_proba"] + 1e-12)) / 2.0)

    report_out = {"rule": args.rule, "temperature": t_star, "val": {}, "test": {}}
    for label, p in (("moe_only", d["moe_proba"]),
                     ("fusion_raw", p_fused_test),
                     ("fusion_calibrated", _softmax(np.log(p_fused_test + 1e-12) / t_star))):
        r = report(label, y_test, p)
        report_out["test"][label] = r
        print(f"[CAL] (test) {label:<18} acc={r['accuracy']:.4f} macro_f1={r['macro_f1']:.4f} ece={r['ece']:.4f}",
              flush=True)
    write_json(args.out, report_out)
    print(f"[CAL] 结果已保存到 {args.out}", flush=True)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    main()
