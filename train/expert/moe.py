import torch
import torch.nn as nn

from train.config.expert import ExpertConfig
from train.config.router import RouterConfig
from train.expert.experts import build_experts
from train.expert.pooling import MeanPooling
from train.model.backbone import Backbone
from train.router.router import Router

class MoEClassifier(nn.Module):
    backbone: Backbone
    experts_cfg: list[ExpertConfig]
    router_cfg: RouterConfig

    pool_mean: MeanPooling
    drop: nn.Dropout
    clf_head: nn.Linear
    experts: nn.ModuleList
    router: Router

    def __init__(
        self,
        backbone: Backbone,
        num_classes: int,
        experts_cfg: list[ExpertConfig],
        router_cfg: RouterConfig,
    ):
        super(MoEClassifier, self).__init__()
        self.backbone = backbone
        self.experts_cfg = experts_cfg
        self.router_cfg = router_cfg

        self.pool_mean = MeanPooling()
        self.drop = nn.Dropout(self.backbone.cfg.dropout)
        self.clf_head = nn.Linear(self.backbone.cfg.hidden_size, num_classes)
        self.experts = build_experts(experts_cfg, self.backbone.cfg.hidden_size, num_classes, self.backbone.cfg.dropout)
        self.router = Router(
            self.backbone.cfg.hidden_size,
            len(self.experts),
            self.router_cfg
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone.model(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        return out.last_hidden_state

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """取输入 embedding，作为梯度类归因（如 IG）的连续性入口"""
        return self.backbone.model.get_input_embeddings()(input_ids)

    def fuse(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        expert_logits = torch.stack([e(hidden, attention_mask) for e in self.experts], dim=1)  # [B, M, C]
        gate = self.router(hidden, attention_mask)  # [B, M]
        fused = (gate.unsqueeze(-1) * expert_logits).sum(dim=1)  # z = Σ g_i · z_i
        return {"logits": fused, "expert_logits": expert_logits, "gate": gate}

    def clf_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled = self.drop(
            self.pool_mean(
                self.encode(
                    input_ids,
                    attention_mask
                ),
                attention_mask
            )
        )
        return self.clf_head(pooled)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.fuse(self.encode(input_ids, attention_mask), attention_mask)

    def forward_from_embeds(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """与 `forward` 等价，但以连续 embedding 为入口，使梯度可回传到输入端"""
        out = self.backbone.model(
            inputs_embeds = inputs_embeds,
            attention_mask = attention_mask
        )
        return self.fuse(out.last_hidden_state, attention_mask)