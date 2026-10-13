from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.config.config import Config
from train.data.dataset import SplitDataset, make_collate_fn, make_loader
from train.data.sampler import set_loader_epoch
from train.expert.moe import MoEClassifier
from train.model.backbone import Backbone
from train.model.loss import ClassBalancedLoss, LoadBalanceLoss
from train.utils.env import amp_context, compile_model, get_device, set_seed


class RouterStage:
    """阶段三：路由网络训练"""

    cfg: Config

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._forward = None

    def run(self) -> None:
        set_seed(self.cfg.seed)

        self.backbone = Backbone(self.cfg.model, self.cfg.peft, is_mlm = False)
        self.backbone.build()

        moe_classifier = MoEClassifier(
            self.backbone,
            self.cfg.num_classes,
            self.cfg.experts,
            self.cfg.router
        )

        expert_path = Path(self.cfg.output.export_path) / "expert" / "model.pt"
        if not expert_path.exists():
            raise FileNotFoundError(f"[ROUTER] 找不到专家模型：{expert_path}")
        ckpt = torch.load(expert_path, map_location="cpu")
        moe_classifier.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        print(f"[ROUTER] 从 {expert_path} 装入专家权重")

        self._freeze(moe_classifier.backbone.model)
        self._freeze(moe_classifier.experts)

        device = get_device()
        moe_classifier.to(device)
        # 训练与评估共用同一份编译结果，避免按调用点重复编译
        self._forward = compile_model(moe_classifier, self.cfg.train.compile)

        train_dataset = SplitDataset(self.cfg.data.dataset_path, "train")
        val_dataset = SplitDataset(self.cfg.data.dataset_path, "val")
        collate = make_collate_fn(self.backbone.tokenizer, self.cfg.data.max_length)
        train_loader = make_loader(
            train_dataset,
            collate,
            self.cfg.train.router_train.batch_size,
            train = True,
            bucketing = self.cfg.train.length_bucketing,
            seed = self.cfg.seed,
            num_workers = self.cfg.train.num_workers,
        )
        val_loader = make_loader(
            val_dataset,
            collate,
            self.cfg.train.router_train.batch_size,
            train = False,
            bucketing = self.cfg.train.length_bucketing,
            seed = self.cfg.seed,
            num_workers = self.cfg.train.num_workers,
        )

        samples_per_cls = torch.bincount(
            train_dataset.labels, minlength=self.cfg.num_classes
        ).float().clamp(min=1)
        loss_fn = self._build_loss(samples_per_cls).to(device)
        balance_fn = LoadBalanceLoss(self.cfg.loss.defaults.balance_weight)

        optimizer = torch.optim.AdamW(
            moe_classifier.router.parameters(),
            lr = self.cfg.train.router_train.lr,
            weight_decay = self.cfg.train.weight_decay
        )
        steps = self.cfg.train.router_train.epochs * len(train_loader)
        scheduler = self._linear_warmup_scheduler(
            optimizer, int(self.cfg.train.warmup_ratio * steps), steps
        )

        ckpt_dir = Path(self.cfg.output.output_path) / "router"
        export_dir = Path(self.cfg.output.export_path) / "router"
        ckpt_path = ckpt_dir / "last.pt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        export_dir.mkdir(parents=True, exist_ok=True)

        best_f1, start_epoch = -1.0, 0
        if self.cfg.resume and ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            moe_classifier.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_epoch = state["epoch"]
            best_f1 = state["best_f1"]
            print(f"[ROUTER] 从 epoch {start_epoch} 断点续训", flush=True)

        for epoch in range(start_epoch, self.cfg.train.router_train.epochs):
            set_loader_epoch(train_loader, epoch)
            moe_classifier.eval()
            moe_classifier.router.train()
            total, count = 0.0, 0
            for batch in tqdm(train_loader, desc=f"ROUTER epoch {epoch + 1}/{self.cfg.train.router_train.epochs}"):
                with amp_context(device, self.cfg.train.amp):
                    out = self._forward(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                    loss = loss_fn(out["logits"], batch["labels"].to(device)) + balance_fn(out["gate"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total += loss.item()
                count += 1
            metrics = self._evaluate(moe_classifier, val_loader, device)
            print(
                f"[ROUTER] epoch {epoch + 1} loss={total / max(1, count):.4f} val_macro_f1={metrics['macro_f1']:.4f} "
                f"gate_f={np.round(metrics['gate_f'], 3).tolist()}",
                flush=True,
            )
            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                torch.save({"model": moe_classifier.state_dict()}, export_dir / "model.pt")
                if hasattr(self.backbone.tokenizer, "save_pretrained"):
                    self.backbone.tokenizer.save_pretrained(export_dir)
            torch.save(
                {"model": moe_classifier.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "best_f1": best_f1},
                ckpt_path,
            )

        (ckpt_dir / "metrics.json").write_text(
            json.dumps({"best_val_macro_f1": best_f1}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[ROUTER] 最佳 val_macro_f1={best_f1:.4f}，已保存到 {export_dir}/model.pt")

    def _build_loss(self, samples_per_cls: torch.Tensor) -> ClassBalancedLoss:
        base_loss = nn.CrossEntropyLoss(
            reduction = "none",
            label_smoothing = self.cfg.loss.defaults.label_smoothing,
        )
        return ClassBalancedLoss(
            samples_per_cls = samples_per_cls,
            beta = self.cfg.loss.defaults.beta,
            num_classes = self.cfg.num_classes,
            base_loss = base_loss,
        )

    @torch.no_grad()
    def _evaluate(self, model: MoEClassifier, loader: DataLoader, device: torch.device) -> dict:
        model.eval()
        # 评估走 eager，理由同 clf.py：避免编译推理图时的 fork 竞争
        forward = model
        preds, labels, gates = [], [], []
        with amp_context(device, self.cfg.train.amp):
            for batch in loader:
                out = forward(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                preds += out["logits"].float().argmax(dim=-1).tolist()
                labels += batch["labels"].tolist()
                gates += out["gate"].float().tolist()
        gate = np.asarray(gates)
        assign = np.bincount(gate.argmax(axis=1), minlength=gate.shape[1]) / len(gate)
        return {
            "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "gate_f": assign.tolist(),
            "gate_p": gate.mean(axis=0).tolist(),
        }

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _linear_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
        def fn(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)