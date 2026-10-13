import json
from pathlib import Path

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
from train.model.loss import ClassBalancedLoss
from train.utils.env import amp_context, compile_model, get_device, set_seed


class ClfStage:
    """阶段一：分类头训练"""

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

        self._freeze(moe_classifier.experts)
        self._freeze(moe_classifier.router)

        mlm_path = Path(self.cfg.output.export_path) / "mlm" / "model.pt"

        if mlm_path.exists():
            ckpt = torch.load(mlm_path, map_location="cpu")
            state = ckpt["model"] if "model" in ckpt else ckpt
            target = moe_classifier.backbone.model.state_dict()
            source = { self._strip_state_prefixes(k) : v for k, v in state.items()}
            matched = 0
            for key, value in target.items():
                src = source.get(self._strip_state_prefixes(key))
                if src is not None and src.shape == value.shape:
                    target[key] = src
                    matched += 1
            moe_classifier.backbone.model.load_state_dict(target)
            print(f"[CLF] 从 {mlm_path} 装入 backbone 权重：{matched}/{len(state)} 个键匹配")

        train_dataset = SplitDataset(
            self.cfg.data.dataset_path,
            "train",
        )
        val_dataset = SplitDataset(
            self.cfg.data.dataset_path,
            "val",
        )

        collate = make_collate_fn(self.backbone.tokenizer, self.cfg.data.max_length)

        train_loader = make_loader(
            train_dataset,
            collate,
            self.cfg.train.clf_train.batch_size,
            train = True,
            bucketing = self.cfg.train.length_bucketing,
            seed = self.cfg.seed,
            num_workers = self.cfg.train.num_workers,
        )
        val_loader = make_loader(
            val_dataset,
            collate,
            self.cfg.train.clf_train.batch_size,
            train = False,
            bucketing = self.cfg.train.length_bucketing,
            seed = self.cfg.seed,
            num_workers = self.cfg.train.num_workers,
        )
        optimizer = torch.optim.AdamW(
            (p for p in moe_classifier.parameters() if p.requires_grad),
            lr = self.cfg.train.clf_train.lr,
            weight_decay = self.cfg.train.weight_decay
        )
        device = get_device()
        moe_classifier.to(device)
        # 训练与评估共用同一份编译结果，避免按调用点重复编译
        self._forward = compile_model(moe_classifier.clf_forward, self.cfg.train.compile)

        loss_fn = self._build_loss(train_dataset.labels).to(device)

        steps = self.cfg.train.clf_train.epochs * len(train_loader)
        scheduler = self._linear_warmup_scheduler(optimizer, int(self.cfg.train.warmup_ratio * steps), steps)

        best_f1, history = -1.0, []
        start_epoch = 0

        ckpt_dir = Path(self.cfg.output.output_path) / "clf"
        export_dir = Path(self.cfg.output.export_path) / "clf"
        ckpt_path = ckpt_dir / "last.pt"

        if self.cfg.resume:
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                moe_classifier.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                scheduler.load_state_dict(ckpt["scheduler"])
                start_epoch = ckpt["epoch"]
                best_f1 = ckpt["best_f1"]
                history = ckpt.get("history", [])
                print(f"[CLF] 从 epoch {start_epoch} 断点续训", flush=True)
            else:
                print("[CLF] 找不到断点，从头开始训练", flush=True)

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        export_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(start_epoch, self.cfg.train.clf_train.epochs):
            set_loader_epoch(train_loader, epoch)
            moe_classifier.train()
            total, count = 0.0, 0
            for batch in tqdm(train_loader, desc=f"CLF epoch {epoch + 1}/{self.cfg.train.clf_train.epochs}"):
                with amp_context(device, self.cfg.train.amp):
                    logits = self._forward(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                    loss = loss_fn(logits, batch["labels"].to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total += loss.item()
                count += 1
            metrics = self._evaluate(moe_classifier, val_loader, device)
            history.append({"epoch": epoch + 1, "train_loss": total / max(1, count), "val_macro_f1": metrics["macro_f1"]})
            print(f"[CLF] epoch {epoch + 1} loss={history[-1]['train_loss']:.4f} val_macro_f1={metrics['macro_f1']:.4f}", flush=True)
            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                torch.save({"model": moe_classifier.state_dict()}, export_dir / "model.pt")
                if hasattr(self.backbone.tokenizer, "save_pretrained"):
                    self.backbone.tokenizer.save_pretrained(export_dir)
            torch.save(
                {"model": moe_classifier.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "best_f1": best_f1, "history": history},
                ckpt_path,
            )

        metrics_path = ckpt_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps({"best_val_macro_f1": best_f1, "history": history}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[CLF] 最佳 val_macro_f1={best_f1:.4f}，checkpoint 已保存到 {ckpt_dir}")

    def _build_loss(self, labels: torch.Tensor) -> ClassBalancedLoss:
        samples_per_cls = torch.bincount(
            labels, minlength=self.cfg.num_classes
        ).float().clamp(min=1)
        base_loss = nn.CrossEntropyLoss(
            reduction="none",
            label_smoothing=self.cfg.loss.defaults.label_smoothing,
        )
        return ClassBalancedLoss(
            samples_per_cls=samples_per_cls,
            beta=self.cfg.loss.defaults.beta,
            num_classes=self.cfg.num_classes,
            base_loss=base_loss,
        )

    @torch.no_grad()
    def _evaluate(self, model: MoEClassifier, loader: DataLoader, device: torch.device) -> dict:
        model.eval()
        # 评估走 eager：inductor 编译推理图时会 fork 编译子进程，实测与
        # lance 的 jemalloc 后台线程冲突并死锁在 Future 上，收益也不值得冒险
        forward = model.clf_forward
        preds, labels = [], []
        with amp_context(device, self.cfg.train.amp):
            for batch in loader:
                logits = forward(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                preds += logits.float().argmax(dim=-1).tolist()
                labels += batch["labels"].tolist()
        return {"macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0))}

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _strip_state_prefixes(key: str) -> str:
        return key.removeprefix("base_model.model.").removeprefix("bert.")

    @staticmethod
    def _linear_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
        def fn(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)