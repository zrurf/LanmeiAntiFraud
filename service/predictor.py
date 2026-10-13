from __future__ import annotations

from pathlib import Path

import torch

from service.attribution import integrated_gradients
from service.model import label_names, load_classifier, load_meta, load_tokenizer
from train.utils.env import get_device


class Predictor:
    """运行时推理与归因封装

    从导出目录加载元数据、tokenizer 与 eager 权重，提供批量预测与
    IG Token 级归因。eager 权重是梯度归因的前提，故运行环境需保留 torch。
    """

    def __init__(self, export_dir: str | Path, device: torch.device | None = None):
        self.export_dir = Path(export_dir)
        self.meta = load_meta(self.export_dir)
        self.labels = label_names(self.meta)
        self.max_length = int(self.meta["max_length"])
        self.expert_names = [expert["name"] for expert in self.meta["experts"]]

        self.device = device or get_device()
        self.tokenizer = load_tokenizer(self.export_dir)
        self.model = load_classifier(self.export_dir, self.meta).to(self.device)
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

    def encode(self, texts: str | list[str]) -> dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]

    @torch.no_grad()
    def predict_batch(self, texts: list[str]) -> list[dict]:
        batch = self.encode(texts)
        out = self.model(batch["input_ids"], batch["attention_mask"])
        probs = out["logits"].softmax(dim=-1)
        gate = out["gate"]

        results: list[dict] = []
        for i in range(len(texts)):
            prob = probs[i]
            weights = gate[i]
            top = int(prob.argmax())
            results.append({
                "label": self.labels[top],
                "label_id": top,
                "confidence": float(prob[top]),
                "probabilities": {
                    self.labels[j]: float(prob[j]) for j in range(len(self.labels))
                },
                "expert_weights": {
                    self.expert_names[j]: float(weights[j]) for j in range(weights.numel())
                },
            })
        return results

    def explain(
        self,
        text: str,
        target: int | None = None,
        steps: int = 32,
        top_k: int | None = None,
    ) -> dict:
        """预测并给出 IG Token 级归因

        `target` 为 `None` 时归因到预测类别；`top_k` 控制返回的 token 数量，
        按归因分数绝对值降序排列
        """
        prediction = self.predict(text)
        batch = self.encode([text])
        scores, targets = integrated_gradients(
            self.model,
            batch["input_ids"],
            batch["attention_mask"],
            target_class=prediction["label_id"] if target is None else target,
            steps=steps,
            baseline_token_id=self.pad_token_id,
        )

        tokens = self.tokenizer.convert_ids_to_tokens(batch["input_ids"][0].tolist())
        length = int(batch["attention_mask"][0].sum())
        attributions = [
            {"token": tokens[i], "score": float(scores[0][i])} for i in range(length)
        ]
        attributions.sort(key=lambda item: abs(item["score"]), reverse=True)
        if top_k is not None:
            attributions = attributions[:top_k]

        result = dict(prediction)
        result["target_label"] = self.labels[int(targets[0])]
        result["attributions"] = attributions
        return result