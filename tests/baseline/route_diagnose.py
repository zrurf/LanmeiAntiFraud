"""路由诊断：在测试集上统计「真实类别 × 门控选中专家」矩阵

用途：判断专家分工是真的成立、还是路由器只把样本堆给了少数专家。
输出两个关键量：

- **对角占比**：真实类别 i 被路由到专家 i 的比例（理想应显著高于 10% 的随机水平）
- **行归一化矩阵**：每类的样本实际流向哪些专家
- **专家边际**：每个专家承接的样本占比（理想各约 10%）

用法:

    python -m tests.baseline.route_diagnose --checkpoint artifacts/export/joint/model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from tests.baseline.common import label_names, load_checkpoint, load_split, write_json
from train.config.config import Config, load_config
from train.data.dataset import make_collate_fn, make_loader
from train.utils.env import get_device

CONFIG_PATH = "config.toml"


class TextListDataset(Dataset):
    def __init__(self, items, max_length: int):
        self.items = items
        self._lengths = [min(len(t) + 2, max_length) for t, _ in items]

    @property
    def lengths(self):
        return self._lengths

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


@torch.no_grad()
def collect(cfg: Config, model, items, batch_size: int, device) -> tuple[np.ndarray, np.ndarray]:
    """顺序推理，返回 (真实标签, 门控选中专家)"""
    loader = make_loader(
        TextListDataset(items, cfg.data.max_length),
        make_collate_fn(model.backbone.tokenizer, cfg.data.max_length),
        batch_size,
        train=False,
        bucketing=False,   # 顺序加载：保证与样本顺序严格对齐
        num_workers=0,
    )
    y_true, gate_arg = [], []
    for batch in loader:
        out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        gate_arg += out["gate"].float().argmax(dim=-1).cpu().tolist()
        y_true += batch["labels"].tolist()
    return np.asarray(y_true), np.asarray(gate_arg)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", default="artifacts/export/joint/model.pt")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    parser.add_argument("--out", default="artifacts/results/routing_diagnose.json")
    args, rest = parser.parse_known_args(argv)

    cfg = load_config(Config, toml_path=CONFIG_PATH, argv=rest)
    device = get_device()
    names = label_names(cfg)
    n = cfg.num_classes

    test = load_split(cfg, "test")
    items = list(zip(test.texts, test.labels.tolist()))
    model = load_checkpoint(cfg, Path(args.checkpoint), device)
    y_true, gate = collect(cfg, model, items, args.batch_size, device)
    assert len(y_true) == len(gate) == len(items), "顺序异常"

    matrix = np.zeros((n, n), dtype=np.int64)
    for t, g in zip(y_true, gate):
        matrix[t, g] += 1
    row_sum = matrix.sum(axis=1, keepdims=True).clip(min=1)
    row_norm = matrix / row_sum
    diag = np.diag(row_norm)
    marginal = matrix.sum(axis=0) / matrix.sum()

    print(f"[ROUTE] 测试集 {len(y_true):,} 条，checkpoint={args.checkpoint}")
    print("\n[ROUTE] 行归一化矩阵（行=真实类别，列=选中专家，%）")
    header = "  " + " ".join(f"{i:>5}" for i in range(n))
    print("            " + header)
    for i in range(n):
        print(f"  {names[i][:8]:<9}" + " ".join(f"{row_norm[i, j] * 100:5.1f}" for j in range(n)))
    print("\n[ROUTE] 对角占比 = 该类样本被路由到同名专家的比例（随机=10%）")
    for i in range(n):
        flag = "✓" if diag[i] >= 0.15 else ("·" if diag[i] >= 0.11 else "✗")
        print(f"  {flag} {names[i]:<14} {diag[i] * 100:5.1f}%")
    print(f"\n[ROUTE] 对角占比均值 = {diag.mean() * 100:.1f}%（随机 10%）")
    print(f"[ROUTE] 专家边际（理想各 10%）："
          + " ".join(f"{names[i][:4]}:{marginal[i] * 100:.1f}%" for i in range(n)))
    print(f"[ROUTE] 边际标准差 = {marginal.std() * 100:.2f} 个百分点（越接近 0 越均衡）")

    write_json(args.out, {
        "label_names": names,
        "n_test": int(len(y_true)),
        "matrix": matrix.tolist(),
        "row_norm": row_norm.tolist(),
        "diagonal_share": diag.tolist(),
        "diagonal_share_mean": float(diag.mean()),
        "expert_marginal": marginal.tolist(),
        "expert_marginal_std": float(marginal.std()),
    })
    print(f"\n[ROUTE] 结果已保存到 {args.out}")


if __name__ == "__main__":
    main()
