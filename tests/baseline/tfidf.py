"""传统方法基线：TF-IDF（词级 unigram + bigram）+ 朴素贝叶斯 / 逻辑回归

`TfidfVectorizer(ngram_range=(1,2), min_df=2)` +
`MultinomialNB` / `LogisticRegression(max_iter=1000)`，输出
accuracy / macro_f1 / per_class。

用法:

    python -m tests.baseline.tfidf [--max-train N] [--max-test N] \
        [--max-features N] [--n-boot N] [--out <json>] [配置项覆盖]

默认使用完整 train / test split；train 含 39.9 万条，全量 TF-IDF + 逻辑
回归较消耗内存，必要时用 `--max-train` / `--max-features` 限流
"""

from __future__ import annotations

import argparse

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from tests.baseline.common import label_names, load_split, score, write_json
from train.config.config import Config, load_config

CONFIG_PATH = "config.toml"


def _cut(text: str) -> list[str]:
    """jieba 词级切分，滤除纯空白 token"""
    return [token for token in jieba.lcut(text) if token.strip()]


def _take(items, limit: int):
    return items if limit <= 0 else items[:limit]


def run_tfidf(
    cfg: Config,
    max_train: int = 0,
    max_test: int = 0,
    max_features: int = 0,
    n_boot: int = 1000,
) -> tuple[dict, list[int], dict[str, list[int]]]:
    """训练 TF-IDF 基线并返回 (报告, y_true, {模型名: y_pred})"""
    train = load_split(cfg, "train")
    test = load_split(cfg, "test")

    x_train_text = _take(train.texts, max_train)
    y_train = _take(train.labels, max_train).tolist()
    x_test_text = _take(test.texts, max_test)
    y_test = _take(test.labels, max_test).tolist()
    names = label_names(cfg)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features or None,
        tokenizer=_cut,
        token_pattern=None,
    )
    x_train = vectorizer.fit_transform(x_train_text)
    x_test = vectorizer.transform(x_test_text)
    print(
        f"[TFIDF] train={len(y_train)} test={len(y_test)} features={x_train.shape[1]}",
        flush=True,
    )

    results: dict[str, dict] = {}
    preds: dict[str, list[int]] = {}
    for name, clf in {
        "tfidf_nb": MultinomialNB(),
        "tfidf_lr": LogisticRegression(max_iter=1000),
    }.items():
        clf.fit(x_train, y_train)
        proba = clf.predict_proba(x_test)
        y_pred = [int(p) for p in proba.argmax(axis=1)]
        metrics = score(y_test, y_pred, proba.max(axis=1), names, n_boot=n_boot)
        results[name] = metrics
        preds[name] = y_pred
        print(
            f"[TFIDF] {name}: accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )

    report = {
        "label_names": names,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_features": int(x_train.shape[1]),
        "results": results,
    }
    return report, y_test, preds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max-train", dest="max_train", type=int, default=0)
    parser.add_argument("--max-test", dest="max_test", type=int, default=0)
    parser.add_argument("--max-features", dest="max_features", type=int, default=0)
    parser.add_argument("--n-boot", dest="n_boot", type=int, default=1000)
    parser.add_argument("--out", default="artifacts/results/tfidf_baseline.json")
    args, rest = parser.parse_known_args(argv)

    cfg = load_config(Config, toml_path=CONFIG_PATH, argv=rest)
    report, _, _ = run_tfidf(
        cfg,
        max_train=args.max_train,
        max_test=args.max_test,
        max_features=args.max_features,
        n_boot=args.n_boot,
    )
    write_json(args.out, report)
    print(f"[TFIDF] 结果已保存到 {args.out}", flush=True)


if __name__ == "__main__":
    main()