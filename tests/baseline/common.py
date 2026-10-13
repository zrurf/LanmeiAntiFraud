"""基线 / 消融实验共享工具
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train.common.metrics import (
    bootstrap_macro_f1_ci,
    confusion_matrix,
    expected_calibration_error,
    macro_f1,
    per_class_prf,
)
from train.config.config import Config
from train.data.dataset import SplitDataset, make_collate_fn
from train.expert.moe import MoEClassifier
from train.model.backbone import Backbone
from train.utils.env import get_device

# 评估模式：single = MacBERT + DoRA 单头基线；moe = 多专家门控融合
MODES = ("single", "moe")
# 融合方式：gated = 门控加权（正式）；equal / top1 为消融 A1 / A2
FUSIONS = ("gated", "equal", "top1")


def label_names(cfg: Config) -> list[str]:
    """按专家的 `index` 还原类别名（专家 index 与类别 id 一一对应）"""
    names = [f"class_{i}" for i in range(cfg.num_classes)]
    for expert in cfg.experts:
        if 0 <= expert.index < cfg.num_classes:
            names[expert.index] = expert.name
    return names


def load_split(cfg: Config, use: str) -> SplitDataset:
    return SplitDataset(cfg.data.dataset_path, use)


def build_model(cfg: Config) -> MoEClassifier:
    backbone = Backbone(cfg.model, cfg.peft, is_mlm=False)
    backbone.build()
    return MoEClassifier(backbone, cfg.num_classes, cfg.experts, cfg.router)


def load_checkpoint(cfg: Config, checkpoint: str | Path, device: torch.device) -> MoEClassifier:
    model = build_model(cfg)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.to(device)
    model.eval()
    print(f"[EVAL] 装入权重：{checkpoint}", flush=True)
    return model


def make_loader(cfg: Config, dataset: SplitDataset, tokenizer, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=make_collate_fn(tokenizer, cfg.data.max_length),
    )


def fuse_logits(out: dict, fusion: str) -> torch.Tensor:
    """按融合方式从 MoE 输出还原分类 logits（equal / top1 为消融）"""
    if fusion == "equal":
        return out["expert_logits"].mean(dim=1)
    if fusion == "top1":
        hard = torch.nn.functional.one_hot(
            out["gate"].argmax(dim=-1), out["gate"].shape[-1]
        ).float()
        return (hard.unsqueeze(-1) * out["expert_logits"]).sum(dim=1)
    return out["logits"]


@torch.no_grad()
def predict(
    model: MoEClassifier, loader: DataLoader, mode: str, fusion: str, device: torch.device
) -> tuple[list[int], list[int], np.ndarray]:
    """返回 (y_true, y_pred, 最大 softmax 置信度)；`mode=single` 走单头基线前向"""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    confs: list[np.ndarray] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        if mode == "single":
            logits = model.clf_forward(input_ids, attention_mask)
        else:
            logits = fuse_logits(model(input_ids, attention_mask), fusion)
        probs = torch.softmax(logits.float(), dim=-1)
        conf, pred = probs.max(dim=-1)
        y_true += batch["labels"].tolist()
        y_pred += pred.cpu().tolist()
        confs.append(conf.cpu().numpy())
    return y_true, y_pred, np.concatenate(confs)


def score(
    y_true: list[int], y_pred: list[int], conf: np.ndarray, names: list[str], n_boot: int = 1000
) -> dict:
    correct = np.asarray([int(t == p) for t, p in zip(y_true, y_pred)])
    return {
        "accuracy": float(correct.mean()) if len(correct) else 0.0,
        "macro_f1": macro_f1(y_true, y_pred),
        "per_class": per_class_prf(y_true, y_pred, names),
        "confusion_matrix": confusion_matrix(y_true, y_pred, len(names)),
        "ece": expected_calibration_error(conf, correct),
        "macro_f1_bootstrap": bootstrap_macro_f1_ci(y_true, y_pred, n_boot=n_boot),
    }


def evaluate(
    cfg: Config,
    checkpoint: str | Path,
    mode: str,
    fusion: str,
    batch_size: int,
    n_boot: int,
    device: torch.device | None = None,
) -> dict:
    """在测试集上评估单个 checkpoint

    返回指标字典，另以 `_y_true` / `_y_pred` 附带逐样本序列，供上层做
    McNemar 与配对 bootstrap；落盘前需剔除下划线前缀字段
    """
    device = device or get_device()
    names = label_names(cfg)
    dataset = load_split(cfg, "test")
    model = load_checkpoint(cfg, checkpoint, device)
    loader = make_loader(cfg, dataset, model.backbone.tokenizer, batch_size)
    y_true, y_pred, conf = predict(model, loader, mode, fusion, device)

    result = score(y_true, y_pred, conf, names, n_boot=n_boot)
    result["mode"] = mode
    result["fusion"] = fusion if mode == "moe" else None
    result["checkpoint"] = str(checkpoint)
    result["_y_true"] = y_true
    result["_y_pred"] = y_pred
    return result


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")