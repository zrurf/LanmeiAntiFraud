import torch
import torch.nn as nn

from train.config.router import RouterConfig
from train.expert.pooling import MeanMaxPooling

class Router(nn.Module):

    cfg: RouterConfig
    pooling: MeanMaxPooling
    proj: nn.Linear
    norm: nn.LayerNorm
    drop: nn.Dropout
    mlp: nn.Sequential

    def __init__(
        self,
        backbone_dim: int,
        num_experts: int,
        cfg: RouterConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.pooling = MeanMaxPooling()
        self.proj = nn.Linear(backbone_dim * 2, cfg.dim)
        self.norm = nn.LayerNorm(cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.mlp = nn.Sequential(nn.Linear(cfg.dim, cfg.dim), nn.ReLU(), nn.Linear(cfg.dim, num_experts))

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        u = self.drop(self.norm(self.proj(self.pooling(hidden, attention_mask))))
        return self.mlp(u).softmax(dim=-1)