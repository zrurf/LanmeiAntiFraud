import torch
import torch.nn as nn

from train.config.expert import ExpertConfig
from train.expert.pooling import POOLINGS_WITH_PARAMS, build_pooling, pooling_output_mult

class Expert(nn.Module):
    cfg: ExpertConfig
    pooling: nn.Module
    mlp: nn.Sequential

    def __init__(
        self,
        cfg: ExpertConfig,
        backbone_dim: int,
        num_classes: int,
        dropout: int,
    ):
        super().__init__()

        if cfg.heads != 1 and cfg.pooling not in POOLINGS_WITH_PARAMS:
            raise ValueError(
                f"专家 {cfg.name}: pooling={cfg.pooling!r} 不支持 heads={cfg.heads}"
            )

        self.cfg = cfg
        self.pooling = build_pooling(cfg.pooling, backbone_dim, cfg.heads)
        in_dim = backbone_dim * pooling_output_mult(cfg.pooling, cfg.heads)
        blocks: list[nn.Module] = []
        d = in_dim
        for _ in range(cfg.layers):
            blocks += [nn.Linear(d, cfg.hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = cfg.hidden
        blocks.append(nn.Linear(d, num_classes))
        self.mlp = nn.Sequential(*blocks)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pooling(hidden, attention_mask))

def build_experts(experts_cfg: list[ExpertConfig], backbone_dim: int, num_classes: int, dropout: int) -> nn.ModuleList:
    return nn.ModuleList(Expert(
            cfg = e,
            backbone_dim=backbone_dim,
            num_classes = num_classes,
            dropout = dropout,
        )
        for e in experts_cfg
    )