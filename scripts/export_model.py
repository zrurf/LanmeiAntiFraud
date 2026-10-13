"""模型导出（训练侧）

将训练完成的权重导出为 PyTorch 官方部署格式——`torch.export` 的
ExportedProgram（落盘为 `model.pt2`），并一并写出运行时所需的元数据、
eager 权重与 tokenizer：

    deploy/
    ├── model.pt2           ExportedProgram，供运行时前向推理
    ├── model.pt            eager state_dict，供运行时梯度归因（IG）
    ├── predict_meta.json   类别名 / 专家配置 / 模型结构
    └── tokenizer 相关文件

用法:

    python -m scripts.export_model [--checkpoint <model.pt>] [--out <目录>] [配置项覆盖]

未指定 `--checkpoint` 时按 stage4 → stage3 → stage2 → stage1 的顺序自动
选取已存在的最佳权重；配置项覆盖语法与训练入口一致
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.export import Dim

from train.config.config import Config, load_config
from train.expert.moe import MoEClassifier
from train.model.backbone import Backbone

CONFIG_PATH = "config.toml"

# 自动选取权重时的阶段优先级：越靠后的阶段越完整
STAGE_PRIORITY = ("joint", "router", "expert", "clf")

EXPORTED_NAME = "model.pt2"
EAGER_NAME = "model.pt"
META_NAME = "predict_meta.json"


def resolve_checkpoint(cfg: Config, checkpoint: str | Path | None) -> Path:
    if checkpoint:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"找不到权重文件：{path}")
        return path

    root = Path(cfg.output.export_path)
    for stage in STAGE_PRIORITY:
        candidate = root / stage / EAGER_NAME
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{root} 下未找到可导出权重，候选阶段：{' → '.join(STAGE_PRIORITY)}"
    )


def build_meta(cfg: Config) -> dict:
    """导出元数据：运行时据此重建模型结构并还原类别名"""
    names: list[str | None] = [None] * cfg.num_classes
    for expert in cfg.experts:
        if not 0 <= expert.index < cfg.num_classes:
            raise ValueError(
                f"专家 {expert.id} 的 index={expert.index} 超出类别数 {cfg.num_classes}"
            )
        names[expert.index] = expert.name
    missing = [i for i, name in enumerate(names) if name is None]
    if missing:
        raise ValueError(f"类别 {missing} 缺少对应专家，无法还原类别名")

    return {
        "num_classes": cfg.num_classes,
        "max_length": cfg.data.max_length,
        "labels": names,
        "model": asdict(cfg.model),
        "peft": asdict(cfg.peft),
        "router": asdict(cfg.router),
        "experts": [asdict(expert) for expert in cfg.experts],
    }


def export_model(
    cfg: Config,
    checkpoint: str | Path | None = None,
    out_dir: str | Path | None = None,
    seq_length: int = 8,
) -> Path:
    """导出模型并返回导出目录"""
    checkpoint_path = resolve_checkpoint(cfg, checkpoint)
    target_dir = Path(out_dir) if out_dir else Path(cfg.output.export_path) / "deploy"
    target_dir.mkdir(parents=True, exist_ok=True)

    backbone = Backbone(cfg.model, cfg.peft, is_mlm=False)
    backbone.build()
    model = MoEClassifier(backbone, cfg.num_classes, cfg.experts, cfg.router)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    print(f"[EXPORT] 装入权重：{checkpoint_path}")

    # 示例 batch 取 2：batch=1 时 dynamo 会把 batch 维特化为常量，
    # 导致导出图无法接受其他 batch 大小
    example_ids = torch.ones(2, seq_length, dtype=torch.long)
    example_mask = torch.ones(2, seq_length, dtype=torch.long)
    program = torch.export.export(
        model,
        (example_ids, example_mask),
        dynamic_shapes={
            "input_ids": {0: Dim("batch"), 1: Dim("seq", min=1, max=cfg.data.max_length)},
            "attention_mask": {0: Dim("batch"), 1: Dim("seq", min=1, max=cfg.data.max_length)},
        },
        strict=False,
    )
    torch.export.save(program, str(target_dir / EXPORTED_NAME))

    torch.save({"model": model.state_dict()}, target_dir / EAGER_NAME)
    (target_dir / META_NAME).write_text(
        json.dumps(build_meta(cfg), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if hasattr(backbone.tokenizer, "save_pretrained"):
        backbone.tokenizer.save_pretrained(target_dir)
    print(f"[EXPORT] 已导出到 {target_dir}")
    return target_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=None)
    args, rest = parser.parse_known_args(argv)

    cfg = load_config(Config, toml_path=CONFIG_PATH, argv=rest)
    export_model(cfg, args.checkpoint, args.out)


if __name__ == "__main__":
    main()