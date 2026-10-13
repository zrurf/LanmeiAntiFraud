"""扰动鲁棒性评估：在「施加强扰动的测试集」上对比 MoE 与传统 TF-IDF 基线

用法:

    python -m tests.baseline.eval_perturbed --checkpoint artifacts/export/joint/model.pt
    python -m tests.baseline.eval_perturbed --ops random_space,mixed --n-boot 200

产物：
- `artifacts/results/perturbed.json`：各扰动变体下两个模型的指标与相对干净集的退化幅度
- `artifacts/results/preds_<op>.npz`：逐样本预测（供后续 McNemar / 融合实验）
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset

from tests.baseline.common import (
    fuse_logits,
    label_names,
    load_checkpoint,
    load_split,
    score,
    write_json,
)
from tests.baseline.tfidf import _cut
from train.config.config import Config, load_config
from train.data.augment import build_augment_fn
from train.data.dataset import make_collate_fn, make_loader
from train.utils.env import get_device

CONFIG_PATH = "config.toml"
# 默认扰动：与配置里启用增强的三个专家一一对应，抽样率直接取自配置；
# `homoglyph_hard` 是同一算子的加强版（提高逐字替换概率），用来观察剂量反应
DEFAULT_OPS = ("emoji_replace", "random_space", "homoglyph", "homoglyph_hard", "mixed")
# 加强版攻击的逐字替换概率（训练时 homoglyph 默认 0.05）
STRESS_RATE = 0.25
# 已拟合的 TF-IDF 缓存路径（拟合约 11 分钟，可在训练期间用 --fit-only 预热）
TFIDF_CACHE = "artifacts/results/tfidf_cache.joblib"


def op_rates(cfg: Config) -> dict[str, float]:
    """从专家配置里读出每个增强算子的抽样率，保证与训练时一致"""
    rates: dict[str, float] = {}
    for expert in cfg.experts:
        for name in expert.augment:
            rates.setdefault(name, expert.augment_rate)
    return rates


class TextListDataset(Dataset):
    """(text, label) 列表的 Dataset，供复用 make_loader / make_collate_fn"""

    def __init__(self, items: list[tuple[str, int]], max_length: int):
        self.items = items
        self._lengths = [min(len(text) + 2, max_length) for text, _ in items]

    @property
    def lengths(self) -> list[int]:
        return self._lengths

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.items[index]


def perturb(texts: list[str], op: str, rates: dict[str, float], seed: int) -> list[str]:
    """按算子扰动文本

    - `mixed`：三个算子依次施加
    - `<算子>_hard`：同一算子，但把逐字替换概率提高到 `STRESS_RATE`（更强攻击）
    """
    if op == "mixed":
        plan: list[tuple[str, float, dict]] = [
            (name, rates.get(name, 1.0), {}) for name in rates
        ]
    elif op.endswith("_hard"):
        base = op[: -len("_hard")]
        plan = [(base, rates.get(base, 1.0), {f"{base}_rate": STRESS_RATE})]
    else:
        plan = [(op, rates.get(op, 1.0), {})]

    random.seed(seed)
    for name, sample_rate, overrides in plan:
        fn = build_augment_fn([name], sample_rate=sample_rate, **overrides)
        if fn is None:
            continue
        texts = [fn(text) for text in texts]
    return texts


@torch.no_grad()
def predict_moe(
    cfg: Config, model, items: list[tuple[str, int]], batch_size: int, device
) -> tuple[list[int], list[int], np.ndarray]:
    """MoE 门控融合推理，返回 (标签, 预测, 置信度, 完整概率矩阵)

    评估用**顺序**加载（不做长度分桶）：分桶会重排批次，使预测序列与原始样本错位，
    也会让不同扰动变体之间的逐样本结果无法对齐（后续融合/检验需要对齐）。
    标签直接从批次里取，保证与预测严格同序。
    """
    dataset = TextListDataset(items, cfg.data.max_length)
    loader = make_loader(
        dataset,
        make_collate_fn(model.backbone.tokenizer, cfg.data.max_length),
        batch_size,
        train=False,
        bucketing=False,
        num_workers=0,
    )
    y_true: list[int] = []
    y_pred: list[int] = []
    confs: list[np.ndarray] = []
    probas: list[np.ndarray] = []
    for batch in loader:
        out = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )
        probs = torch.softmax(fuse_logits(out, "gated").float(), dim=-1)
        conf, pred = probs.max(dim=-1)
        y_true += batch["labels"].tolist()
        y_pred += pred.cpu().tolist()
        confs.append(conf.cpu().numpy())
        probas.append(probs.cpu().numpy())
    return y_true, y_pred, np.concatenate(confs), np.concatenate(probas)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", default="artifacts/export/joint/model.pt")
    parser.add_argument("--ops", default=",".join(DEFAULT_OPS))
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    parser.add_argument("--n-boot", dest="n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-tfidf", dest="skip_tfidf", action="store_true")
    parser.add_argument("--skip-moe", dest="skip_moe", action="store_true")
    parser.add_argument("--fit-only", dest="fit_only", action="store_true",
                        help="只拟合 TF-IDF 并缓存后退出（可在 GPU 训练期间并行预热）")
    parser.add_argument("--refit-tfidf", dest="refit_tfidf", action="store_true",
                        help="忽略缓存重新拟合 TF-IDF")
    parser.add_argument("--max-train", dest="max_train", type=int, default=0,
                        help="TF-IDF 训练样本上限，0 表示全量")
    parser.add_argument("--out", default="artifacts/results/perturbed.json")
    args, rest = parser.parse_known_args(argv)

    cfg = load_config(Config, toml_path=CONFIG_PATH, argv=rest)
    device = get_device()
    names = label_names(cfg)
    rates = op_rates(cfg)
    ops = [op.strip() for op in args.ops.split(",") if op.strip()]
    print(f"[PERTURB] 扰动算子与抽样率（取自配置）：{ {k: rates[k] for k in rates} }", flush=True)

    test = load_split(cfg, "test")
    clean_items = list(zip(test.texts, test.labels.tolist()))
    print(f"[PERTURB] 测试集 {len(clean_items):,} 条；变体：clean + {ops}", flush=True)

    # TF-IDF 拟合很慢（约 11 分钟）且与模型无关，因此支持缓存；`--fit-only` 可在
    # GPU 训练期间先并行预热，收尾时直接复用
    if args.fit_only or not args.skip_tfidf:
        cache = Path(TFIDF_CACHE)
        if cache.exists() and not args.refit_tfidf:
            vectorizer, clf = joblib.load(cache)
            print(f"[PERTURB] 复用已拟合的 TF-IDF：{cache}", flush=True)
        else:
            train = load_split(cfg, "train")
            train_texts = train.texts if args.max_train <= 0 else train.texts[: args.max_train]
            train_labels = train.labels[: len(train_texts)].tolist()
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, tokenizer=_cut, token_pattern=None
            )
            x_train = vectorizer.fit_transform(train_texts)
            clf = LogisticRegression(max_iter=1000).fit(x_train, train_labels)
            print(f"[PERTURB] TF-IDF 拟合完成，特征 {x_train.shape[1]:,} 维", flush=True)
            cache.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump((vectorizer, clf), cache)
            print(f"[PERTURB] 已缓存到 {cache}", flush=True)
        if args.fit_only:
            return
    variants: dict[str, list[tuple[str, int]]] = {"clean": clean_items}
    labels = [y for _, y in clean_items]
    for op in ops:
        base_op = op[: -len("_hard")] if op.endswith("_hard") else op
        if base_op != "mixed" and base_op not in rates:
            print(f"[PERTURB] 跳过未在配置中启用的算子：{op}", flush=True)
            continue
        texts = perturb([t for t, _ in clean_items], op, rates, args.seed)
        variants[op] = list(zip(texts, labels))
        changed = sum(1 for (a, _), (b, _) in zip(clean_items, variants[op]) if a != b)
        print(f"[PERTURB] {op}: {changed / len(clean_items):.1%} 的样本被改写", flush=True)

    report: dict = {
        "label_names": names,
        "n_test": len(clean_items),
        "op_rates": rates,
        "variants": {},
    }

    # ---- MoE 门控 ----
    preds: dict[str, dict[str, np.ndarray]] = {}
    if not args.skip_moe:
        model = load_checkpoint(cfg, Path(args.checkpoint), device)
        for op, items in variants.items():
            y_true, y_pred, conf, proba_moe = predict_moe(cfg, model, items, args.batch_size, device)
            assert len(y_true) == len(items), f"批次顺序异常：{len(y_true)} != {len(items)}"
            metrics = score(y_true, y_pred, conf, names, n_boot=args.n_boot)
            report["variants"].setdefault(op, {})["moe_gated"] = metrics
            preds[op] = {"y_true": np.asarray(y_true), "moe": np.asarray(y_pred),
                         "moe_conf": np.asarray(conf), "moe_proba": proba_moe}
            print(
                f"[PERTURB] moe_gated  {op:<14} accuracy={metrics['accuracy']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f} ece={metrics['ece']:.4f}",
                flush=True,
            )

    # ---- TF-IDF + 逻辑回归（vectorizer/clf 来自缓存或上面刚拟合）----
    if not args.skip_tfidf:
        for op, items in variants.items():
            x = vectorizer.transform([t for t, _ in items])
            proba = clf.predict_proba(x)
            y_pred = [int(p) for p in proba.argmax(axis=1)]
            y_true = preds[op]["y_true"] if op in preds else np.asarray([y for _, y in items])
            metrics = score(y_true.tolist(), y_pred, proba.max(axis=1), names,
                            n_boot=args.n_boot)
            report["variants"].setdefault(op, {})["tfidf_lr"] = metrics
            preds.setdefault(op, {"y_true": y_true})["tfidf"] = np.asarray(y_pred)
            # 保存置信度供后续做 MoE × TF-IDF 融合实验（两者错误模式互补）
            preds[op]["tfidf_conf"] = proba.max(axis=1)
            preds[op]["tfidf_proba"] = proba
            print(
                f"[PERTURB] tfidf_lr   {op:<14} accuracy={metrics['accuracy']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f} ece={metrics['ece']:.4f}",
                flush=True,
            )

    for op, arrays in preds.items():
        Path("artifacts/results").mkdir(parents=True, exist_ok=True)
        np.savez(f"artifacts/results/preds_{op}.npz", **arrays)

    # ---- 退化幅度汇总 ----
    print("\n[PERTURB] Macro-F1 相对干净集的退化", flush=True)
    base = report["variants"]["clean"]
    print(f"{'变体':<16}{'moe_gated':>12}{'tfidf_lr':>12}{'差值(moe-tfidf)':>18}")
    for op, per_model in report["variants"].items():
        moe_f1 = per_model["moe_gated"]["macro_f1"]
        row = f"{op:<16}{moe_f1:>12.4f}"
        if "tfidf_lr" in per_model:
            tf_f1 = per_model["tfidf_lr"]["macro_f1"]
            row += f"{tf_f1:>12.4f}{moe_f1 - tf_f1:>+18.4f}"
        else:
            row += f"{'-':>12}{'-':>18}"
        print(row, flush=True)
    report["clean_moe_macro_f1"] = base["moe_gated"]["macro_f1"]
    write_json(args.out, report)
    print(f"\n[PERTURB] 结果已保存到 {args.out}", flush=True)


if __name__ == "__main__":
    main()
