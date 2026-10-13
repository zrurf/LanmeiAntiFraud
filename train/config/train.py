from __future__ import annotations

from dataclasses import dataclass

@dataclass
class TrainConfig:
    weight_decay: float
    warmup_ratio: float
    mlm_pretrain: PretrainStageConfig
    clf_train: TrainStageConfig
    expert_train: TrainStageConfig
    router_train: TrainStageConfig
    joint_train: TrainStageConfig
    amp: bool = True
    length_bucketing: bool = True
    num_workers: int = 6
    compile: bool = False
    eval_last_only: bool = False
    joint_freeze_router: bool = False

@dataclass
class PretrainStageConfig:
    epochs: int
    lr: float
    probability: float
    batch_size: int

@dataclass
class TrainStageConfig:
    epochs: int
    lr: float
    batch_size: int

@dataclass
class LossConfig:
    defaults: DefaultLossConfig


@dataclass
class DefaultLossConfig:
    class_weights: str | list[float] | None
    label_smoothing: float
    focal_gamma: float
    self_weight: float
    other_weight: float
    beta: float = 0.999
    balance_weight: float = 0.01

@dataclass
class RouterLossConfig:
    balance_weight: float
    entropy_weight: float
    z_loss_weight: float

@dataclass
class ClfLossConfig:
    type: str
    class_weights: str | list[float] | None
    label_smoothing: float

@dataclass
class ExpertLossConfig:
    use_expert_loss: bool
    router_balance_weight: float

@dataclass
class JointLossConfig:
    use_expert_loss: bool
    router_balance_weight: float
    router_entropy_weight: float