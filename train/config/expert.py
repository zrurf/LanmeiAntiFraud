from dataclasses import dataclass

@dataclass
class ExpertConfig:
    index: int
    id: str
    name: str
    pooling: str
    hidden: int
    layers: int
    augment: list[str]
    heads: int = 1
    augment_rate: float = 1.0

@dataclass
class ExpertLossConfig:
    self_weight: float
    other_weight: float
    focal_gamma: float
    label_smoothing: float
    class_weights: str | list[float] | None