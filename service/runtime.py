from __future__ import annotations

from pathlib import Path

import torch

from service.attribution import integrated_gradients
from service.model import (
    EXPORTED_NAME,
    load_classifier,
    load_exported_program,
    load_meta,
    load_tokenizer,
)
from train.utils.env import get_device


class Runtime:
    """推理引擎，一个进程内复用同一实例"""

    def __init__(self, export_dir: str | Path, device: torch.device | None = None):
        self.export_dir = Path(export_dir)
        self.meta = load_meta(self.export_dir)
        self.device = device or get_device()
        self.tokenizer = load_tokenizer(self.export_dir)
        self.model = load_classifier(self.export_dir, self.meta).to(self.device)

        self.num_classes = int(self.meta["num_classes"])
        self.max_length = int(self.meta["max_length"])
        self.pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        )
        self.labels = self._build_labels()
        self.experts = self._build_experts()

        # torch.export 导图在 CPU 上导出，参数不随设备迁移，故仅在 CPU 上启用；
        # 其余情况前向回退 eager，与导图数值一致
        self._program = None
        if self.device.type == "cpu" and (self.export_dir / EXPORTED_NAME).exists():
            self._program = load_exported_program(self.export_dir)

    @property
    def forward_backend(self) -> str:
        return "pt2" if self._program is not None else "eager"

    def _build_labels(self) -> list[dict]:
        """按类别 id 还原标签目录，`key` 取专家英文 id"""
        keys = [f"class_{i}" for i in range(self.num_classes)]
        names = [f"class_{i}" for i in range(self.num_classes)]
        for expert in self.meta["experts"]:
            index = int(expert["index"])
            keys[index] = expert["id"]
            names[index] = expert["name"]
        return [{"id": i, "key": keys[i], "name": names[i]} for i in range(self.num_classes)]

    def _build_experts(self) -> list[dict]:
        """专家目录，顺序须与 expert_logits 的专家维一致"""
        return [
            {
                "id": int(expert["index"]),
                "key": expert["id"],
                "name": expert["name"],
                "pooling": expert["pooling"],
            }
            for expert in self.meta["experts"]
        ]

    def _forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        if self._program is not None:
            return self._program.module()(input_ids, attention_mask)
        return self.model(input_ids, attention_mask)

    def infer(
        self,
        texts: list[str],
        explain: bool = False,
        top_k: int = 8,
        steps: int = 32,
        target: int | None = None,
    ) -> dict:
        """批量推理，返回 `{"items": [...]}`

        `explain=True` 时每条 item 附带 IG Token 级归因；`target` 对所有
        texts 统一生效，`None` 表示归因到各自预测类别
        """
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            out = self._forward(input_ids, attention_mask)
            probs = out["logits"].softmax(dim=-1)
            expert_probs = out["expert_logits"].softmax(dim=-1)
            gate = out["gate"]
            top = probs.argmax(dim=-1)

        scores = targets = None
        if explain:
            scores, targets = integrated_gradients(
                self.model,
                input_ids,
                attention_mask,
                target_class=target,
                steps=steps,
                baseline_token_id=self.pad_token_id,
            )

        items = []
        for index, text in enumerate(texts):
            label_id = int(top[index])
            items.append(
                {
                    "text": text,
                    "label_id": label_id,
                    "label_key": self.labels[label_id]["key"],
                    "label_name": self.labels[label_id]["name"],
                    "confidence": float(probs[index][label_id]),
                    "probabilities": [float(value) for value in probs[index]],
                    "experts_opinion": [
                        {
                            "id": expert["id"],
                            "key": expert["key"],
                            "probabilities": [
                                float(value) for value in expert_probs[index][position]
                            ],
                            "weight": float(gate[index][position]),
                        }
                        for position, expert in enumerate(self.experts)
                    ],
                    "explanation": (
                        self._explain_item(index, input_ids, attention_mask, scores, targets, top_k)
                        if explain
                        else None
                    ),
                }
            )
        return {"items": items}

    def _explain_item(
        self,
        index: int,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        scores,
        targets,
        top_k: int,
    ) -> dict:
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[index].tolist())
        length = int(attention_mask[index].sum())
        attributions = [
            {"token": tokens[position], "score": float(scores[index][position])}
            for position in range(length)
        ]
        attributions.sort(key=lambda item: abs(item["score"]), reverse=True)
        return {
            "target_label_id": int(targets[index]),
            "attributions": attributions[:top_k],
        }