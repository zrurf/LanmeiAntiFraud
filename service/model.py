from __future__ import annotations

import json
from pathlib import Path

import torch

from train.config.expert import ExpertConfig
from train.config.model import ModelConfig
from train.config.peft import LoraConfig, PeftConfig
from train.config.router import RouterConfig
from train.expert.moe import MoEClassifier
from train.model.backbone import Backbone

META_NAME = "predict_meta.json"
EAGER_NAME = "model.pt"
EXPORTED_NAME = "model.pt2"


def load_meta(export_dir: str | Path) -> dict:
    path = Path(export_dir) / META_NAME
    if not path.exists():
        raise FileNotFoundError(f"找不到导出元数据：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def label_names(meta: dict) -> list[str]:
    """按类别 id 还原类别名（专家 `index` 与类别 id 一一对应）"""
    num_classes = int(meta["num_classes"])
    names: list[str | None] = [None] * num_classes
    for expert in meta["experts"]:
        index = int(expert["index"])
        if 0 <= index < num_classes:
            names[index] = expert["name"]
    return [name if name is not None else f"class_{i}" for i, name in enumerate(names)]


def build_classifier(meta: dict) -> MoEClassifier:
    """按导出元数据重建 MoE 分类器结构（不含权重）"""
    backbone = Backbone(ModelConfig(**meta["model"]), _peft_config(meta["peft"]), is_mlm=False)
    backbone.build_model()
    return MoEClassifier(
        backbone,
        int(meta["num_classes"]),
        [ExpertConfig(**expert) for expert in meta["experts"]],
        RouterConfig(**meta["router"]),
    )


def load_classifier(export_dir: str | Path, meta: dict | None = None) -> MoEClassifier:
    """重建分类器并装入 eager 权重，供梯度类归因使用"""
    export_dir = Path(export_dir)
    meta = meta or load_meta(export_dir)
    model = build_classifier(meta)
    ckpt = torch.load(export_dir / EAGER_NAME, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    return model


def load_tokenizer(export_dir: str | Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(export_dir))


def load_exported_program(export_dir: str | Path):
    """加载 `torch.export` 产出的 ExportedProgram（前向推理用）"""
    from torch.export import load as load_exported

    path = Path(export_dir) / EXPORTED_NAME
    if not path.exists():
        raise FileNotFoundError(f"找不到导出图：{path}")
    return load_exported(str(path))


def _peft_config(meta: dict) -> PeftConfig:
    lora = meta.get("lora_config")
    return PeftConfig(
        method=meta["method"],
        lora_config=LoraConfig(**lora) if lora else None,
    )