"""TF-IDF 四配置基线（补齐「文本增强版」两行）

已有（`tests.baseline.tfidf` 与 `eval_perturbed`）：
- `tfidf_nb`      TF-IDF + 多项式朴素贝叶斯，干净训练集
- `tfidf_lr`      TF-IDF + 逻辑回归，干净训练集

本脚本补上公平对照——**训练集混入按增强算子改写的样本**后再拟合：
- `tfidf_nb_aug` / `tfidf_lr_aug`

动机：神经侧有 3 个专家训练时使用了文本增强（emoji_replace / random_space / homoglyph，
抽样率 0.3 / 0.2 / 0.4），而传统基线一直是干净数据训练的，对比不对称。
为与之量级对齐，这里对每个训练样本以 `--aug-rate`（默认 0.3）的概率随机施加其中一个算子
（算子自身的逐字概率沿用训练默认值）。

用法:

    python -m tests.baseline.tfidf_augmented --aug-rate 0.3
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from tests.baseline.common import label_names, load_split, score, write_json
from tests.baseline.tfidf import _cut
from train.config.config import Config, load_config
from train.data.augment import build_augment_fn

CONFIG_PATH = "config.toml"


def augment_training_set(
    texts: list[str], labels: list[int], ops: list[str], aug_rate: float, seed: int
) -> tuple[list[str], list[int]]:
    """在干净训练集基础上追加一部分改写样本"""
    fns = [(op, build_augment_fn([op])) for op in ops]
    rng = random.Random(seed)
    extra_texts: list[str] = []
    extra_labels: list[int] = []
    for text, label in zip(texts, labels):
        if rng.random() >= aug_rate:
            continue
        name, fn = fns[rng.randrange(len(fns))]
        if fn is None:
            continue
        out = fn(text)
        if out != text:
            extra_texts.append(out)
            extra_labels.append(label)
    print(
        f"[TFIDF-AUG] 干净 {len(texts):,} 条 + 改写 {len(extra_texts):,} 条"
        f"（aug_rate={aug_rate}，算子 {ops}）",
        flush=True,
    )
    return texts + extra_texts, labels + extra_labels


def op_names(cfg: Config) -> list[str]:
    """从专家配置里取出启用增强的算子（与训练时一致）"""
    names: list[str] = []
    for expert in cfg.experts:
        for name in expert.augment:
            if name not in names:
                names.append(name)
    return names


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--aug-rate", dest="aug_rate", type=float, default=0.3)
    parser.add_argument("--max-train", dest="max_train", type=int, default=0)
    parser.add_argument("--max-features", dest="max_features", type=int, default=0)
    parser.add_argument("--n-boot", dest="n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="artifacts/results/tfidf_augmented.json")
    args, rest = parser.parse_known_args(argv)

    cfg = load_config(Config, toml_path=CONFIG_PATH, argv=rest)
    names = label_names(cfg)
    ops = op_names(cfg)
    if not ops:
        raise SystemExit("配置里没有任何增强算子，无需本脚本")

    train = load_split(cfg, "train")
    test = load_split(cfg, "test")
    train_texts = train.texts if args.max_train <= 0 else train.texts[: args.max_train]
    train_labels = train.labels[: len(train_texts)].tolist()

    x_texts, y_train = augment_training_set(train_texts, train_labels, ops, args.aug_rate, args.seed)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=args.max_features or None,
        tokenizer=_cut, token_pattern=None,
    )
    x_train = vectorizer.fit_transform(x_texts)
    x_test = vectorizer.transform(test.texts)
    print(f"[TFIDF-AUG] 特征 {x_train.shape[1]:,} 维，测试集 {len(test.texts):,} 条", flush=True)

    y_test = test.labels.tolist()
    report = {
        "label_names": names,
        "n_train_clean": len(train_texts),
        "n_train_total": len(x_texts),
        "aug_rate": args.aug_rate,
        "ops": ops,
        "n_features": int(x_train.shape[1]),
        "results": {},
    }
    for name, clf in {"tfidf_nb_aug": MultinomialNB(),
                      "tfidf_lr_aug": LogisticRegression(max_iter=1000)}.items():
        clf.fit(x_train, y_train)
        proba = clf.predict_proba(x_test)
        y_pred = [int(p) for p in proba.argmax(axis=1)]
        metrics = score(y_test, y_pred, proba.max(axis=1), names, n_boot=args.n_boot)
        report["results"][name] = metrics
        print(f"[TFIDF-AUG] {name}: accuracy={metrics['accuracy']:.4f} "
              f"macro_f1={metrics['macro_f1']:.4f} ece={metrics['ece']:.4f}", flush=True)

    # 顺带把干净版的两行读出来，便于一次看全四个配置
    clean_candidates = [
        Path("artifacts/results/tfidf_baseline.json"),
        Path("artifacts/runs/v1-20261013-cb0.99999/results/tfidf_baseline.json"),
    ]
    for cand in clean_candidates:
        if cand.exists():
            clean = json.loads(cand.read_text(encoding="utf-8"))
            if "results" in clean:
                report["clean_baseline"] = {
                    k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"], "ece": v["ece"]}
                    for k, v in clean["results"].items()
                }
                print(f"[TFIDF-AUG] 干净版对照（读自 {cand}）："
                      + ", ".join(f"{k}={v['macro_f1']:.4f}" for k, v in report["clean_baseline"].items()),
                      flush=True)
            break

    write_json(args.out, report)
    print(f"[TFIDF-AUG] 结果已保存到 {args.out}")


if __name__ == "__main__":
    main()
