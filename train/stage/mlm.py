from pathlib import Path
from peft import LoraConfig, get_peft_model
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling

from train.config.config import Config
from train.data.dataset import SplitDataset, make_loader
from train.data.sampler import set_loader_epoch
from train.model.backbone import Backbone
from train.model.fine_tuning import FINE_TUNING_METHODS
from train.utils.env import amp_context, compile_model, get_device, set_seed

class MlmStage:
    """预训练：MLM训练"""

    cfg: Config
    backbone: Backbone

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self):
        set_seed(self.cfg.seed)
        self.backbone = Backbone(self.cfg.model, self.cfg.peft, is_mlm = True)
        self.backbone.build()
        dataset = SplitDataset(self.cfg.data.dataset_path, "mlm", columns=["id", "text"])
        collator = RawTextMLMCollator(
            tokenizer = self.backbone.tokenizer,
            max_length = self.cfg.data.max_length,
            mlm_probability = self.cfg.train.mlm_pretrain.probability,
        )
        loader = make_loader(
            dataset,
            collator,
            self.cfg.train.mlm_pretrain.batch_size,
            train = True,
            bucketing = self.cfg.train.length_bucketing,
            seed = self.cfg.seed,
            num_workers = self.cfg.train.num_workers,
        )
        optimizer = torch.optim.AdamW(
            (p for p in self.backbone.model.parameters() if p.requires_grad),
            lr = self.cfg.train.mlm_pretrain.lr,
            weight_decay = self.cfg.train.weight_decay
        )
        device = get_device()
        self.backbone.model.to(device)
        forward = compile_model(self.backbone.model, self.cfg.train.compile)
        steps = len(loader) * self.cfg.train.mlm_pretrain.epochs
        scheduler = self._linear_warmup_scheduler(
            optimizer, int(self.cfg.train.warmup_ratio * steps), steps
        )

        start_epoch = 0

        ckpt_dir = Path(self.cfg.output.output_path) / "mlm"
        export_dir = Path(self.cfg.output.export_path) / "mlm"
        ckpt_path = ckpt_dir / "last.pt"

        if self.cfg.resume:
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                self.backbone.model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                scheduler.load_state_dict(ckpt["scheduler"])
                start_epoch = ckpt["epoch"]
                print(f"[MLM] 从 epoch {start_epoch} 断点续训", flush=True)
            else:
                print("[MLM] 找不到断点，从头开始训练", flush=True)

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        export_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(start_epoch, self.cfg.train.mlm_pretrain.epochs):
            set_loader_epoch(loader, epoch)
            self.backbone.model.train()
            total, count = 0.0, 0
            for batch in tqdm(loader, desc=f"MLM epoch {epoch + 1}/{self.cfg.train.mlm_pretrain.epochs}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                with amp_context(device, self.cfg.train.amp):
                    out = forward(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    loss = out.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total += loss.item()
                count += 1
            print(f"[MLM] epoch {epoch + 1} loss = {total / max(1, count):.4f}", flush=True)
            torch.save(
                {"model": self.backbone.model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch + 1},
                ckpt_path,
            )

        torch.save(
            {"model": self.backbone.model.state_dict()},
            export_dir / "model.pt",
        )
        if hasattr(self.backbone.tokenizer, "save_pretrained"):
            self.backbone.tokenizer.save_pretrained(export_dir)
        print(f"[MLM] 已保存到 {export_dir}/model.pt")

    @staticmethod
    def _linear_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
        def fn(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return max(
                0.0,
                (total_steps - step) / max(1, total_steps - warmup_steps)
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)

class RawTextMLMCollator:
    def __init__(self, tokenizer, max_length: int, mlm_probability: float):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inner = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
        )

    def __call__(self, batch):
        texts = [x[0] for x in batch]
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
        )
        features = [
            {k: v[i] for k, v in enc.items()}
            for i in range(len(texts))
        ]
        return self.inner(features)