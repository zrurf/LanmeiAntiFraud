from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.config.config import Config
from train.data.augment import build_augment_fn
from train.data.dataset import SplitDataset, make_collate_fn, make_loader
from train.data.sampler import set_loader_epoch
from train.expert.moe import MoEClassifier
from train.model.backbone import Backbone
from train.model.loss import ClassBalancedLoss
from train.utils.env import amp_context, compile_model, get_device, set_seed


class ExpertStage:
    """阶段二：专家网络训练"""

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

        clf_path = Path(self.cfg.output.export_path) / "clf" / "model.pt"
        if not clf_path.exists():
            raise FileNotFoundError(f"[EXPERT] 找不到 CLF 模型：{clf_path}")
        ckpt = torch.load(clf_path, map_location="cpu")
        moe_classifier.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        print(f"[EXPERT] 从 {clf_path} 装入 CLF 权重")

        self._freeze(moe_classifier.backbone.model)

        device = get_device()
        moe_classifier.to(device)
        # 训练与评估共用同一份编译结果，避免按调用点重复编译
        self._forward = compile_model(moe_classifier, self.cfg.train.compile)

        train_dataset = SplitDataset(self.cfg.data.dataset_path, "train")
        val_dataset = SplitDataset(self.cfg.data.dataset_path, "val")
        val_loader = make_loader(
            val_dataset,
            make_collate_fn(self.backbone.tokenizer, self.cfg.data.max_length),
            self.cfg.train.expert_train.batch_size,
            train = False,
            bucketing = self.cfg.train.length_bucketing,
            seed = self.cfg.seed,
            num_workers = self.cfg.train.num_workers,
        )
        samples_per_cls = torch.bincount(
            train_dataset.labels, minlength=self.cfg.num_classes
        ).float().clamp(min=1)

        ckpt_dir = Path(self.cfg.output.output_path) / "expert"
        export_dir = Path(self.cfg.output.export_path) / "expert"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        export_dir.mkdir(parents=True, exist_ok=True)

        done_path = ckpt_dir / "experts_done.json"
        done: set[int] = set()
        history: dict[str, dict] = {}
        if self.cfg.resume and done_path.exists():
            info = json.loads(done_path.read_text(encoding="utf-8"))
            done = set(info.get("experts_done", []))
            history = info.get("history", {})
            for idx in sorted(done):
                expert_path = ckpt_dir / f"expert_{idx}.pt"
                if expert_path.exists():
                    state = torch.load(expert_path, map_location="cpu", weights_only=False)["expert_state"]
                    moe_classifier.experts[idx].load_state_dict(state)
                    self._freeze(moe_classifier.experts[idx])
            print(f"[EXPERT] 断点续训，跳过已完成专家 {sorted(done)}", flush=True)

        for expert_cfg in self.cfg.experts:
            idx = expert_cfg.index
            if idx in done:
                continue

            loss_fn = self._build_loss(samples_per_cls, idx).to(device)
            train_loader = make_loader(
                train_dataset,
                make_collate_fn(
                    self.backbone.tokenizer,
                    self.cfg.data.max_length,
                    augmentation = build_augment_fn(
                        expert_cfg.augment,
                        sample_rate = expert_cfg.augment_rate,
                    ),
                ),
                self.cfg.train.expert_train.batch_size,
                train = True,
                bucketing = self.cfg.train.length_bucketing,
                seed = self.cfg.seed,
                num_workers = self.cfg.train.num_workers,
            )
            optimizer = torch.optim.AdamW(
                moe_classifier.experts[idx].parameters(),
                lr = self.cfg.train.expert_train.lr,
                weight_decay = self.cfg.train.weight_decay
            )

            print(f"[EXPERT] 训练专家 {idx}（{expert_cfg.name}，负责类别 {idx}）", flush=True)
            best_f1, best_state = -1.0, None
            last_epoch = self.cfg.train.expert_train.epochs - 1
            for epoch in range(self.cfg.train.expert_train.epochs):
                set_loader_epoch(train_loader, epoch)
                moe_classifier.eval()
                # 全部专家置 train：损失只取 expert_logits[:, idx]，其余专家的 dropout
                # 不影响梯度，却能让 10 个专家共用一份 torch.compile 图（否则每个专家
                # 的状态组合不同，各触发一次编译）
                moe_classifier.experts.train()
                total, count = 0.0, 0
                for batch in tqdm(train_loader, desc=f"EXPERT[{expert_cfg.name}] epoch {epoch + 1}/{self.cfg.train.expert_train.epochs}"):
                    with amp_context(device, self.cfg.train.amp):
                        out = self._forward(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                        loss = loss_fn(out["expert_logits"][:, idx], batch["labels"].to(device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total += loss.item()
                    count += 1
                mean_loss = total / max(1, count)
                # `eval_last_only` 时跳过中间轮次的验证集评估：整轮评估约占专家阶段三分之一
                # 的时间，而实测最优轮次基本落在最后两轮，仅用末轮状态的代价极小
                if self.cfg.train.eval_last_only and epoch != last_epoch:
                    print(f"[EXPERT] epoch {epoch + 1} loss={mean_loss:.4f}（跳过验证集评估）", flush=True)
                    continue
                metrics = self._evaluate(moe_classifier, val_loader, device, idx)
                print(f"[EXPERT] epoch {epoch + 1} loss={mean_loss:.4f} val_macro_f1={metrics['macro_f1']:.4f}", flush=True)
                if metrics["macro_f1"] > best_f1:
                    best_f1 = metrics["macro_f1"]
                    best_state = {k: v.detach().cpu().clone() for k, v in moe_classifier.experts[idx].state_dict().items()}

            if best_state is not None:
                moe_classifier.experts[idx].load_state_dict(best_state)
            self._freeze(moe_classifier.experts[idx])
            history[expert_cfg.name] = {"best_val_macro_f1": best_f1}
            done.add(idx)
            torch.save(
                {"expert_state": moe_classifier.experts[idx].state_dict(), "index": idx, "best_val_macro_f1": best_f1},
                ckpt_dir / f"expert_{idx}.pt",
            )
            done_path.write_text(
                json.dumps({"experts_done": sorted(done), "history": history}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        torch.save({"model": moe_classifier.state_dict()}, export_dir / "model.pt")
        if hasattr(self.backbone.tokenizer, "save_pretrained"):
            self.backbone.tokenizer.save_pretrained(export_dir)
        (ckpt_dir / "metrics.json").write_text(
            json.dumps({"experts": history}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[EXPERT] 全部专家训练完成，已保存到 {export_dir}/model.pt")

    def _build_loss(self, samples_per_cls: torch.Tensor, self_class: int) -> ClassBalancedLoss:
        base_loss = nn.CrossEntropyLoss(
            reduction = "none",
            label_smoothing = self.cfg.loss.defaults.label_smoothing,
        )
        return ClassBalancedLoss(
            samples_per_cls = samples_per_cls,
            beta = self.cfg.loss.defaults.beta,
            num_classes = self.cfg.num_classes,
            base_loss = base_loss,
            lambda_self = self.cfg.loss.defaults.self_weight,
            self_class = self_class,
        )

    @torch.no_grad()
    def _evaluate(self, model: MoEClassifier, loader: DataLoader, device: torch.device, expert_idx: int) -> dict:
        model.eval()
        # 评估走 eager，理由同 clf.py：避免编译推理图时的 fork 竞争
        forward = model
        preds, labels = [], []
        with amp_context(device, self.cfg.train.amp):
            for batch in loader:
                out = forward(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                preds += out["expert_logits"][:, expert_idx].float().argmax(dim=-1).tolist()
                labels += batch["labels"].tolist()
        return {"macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0))}

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad_(False)