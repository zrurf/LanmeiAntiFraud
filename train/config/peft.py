from dataclasses import dataclass, field
from enum import Enum

@dataclass
class LoraConfig:
    bias: str
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["query", "key", "value"])

@dataclass
class PeftConfig:
    method: str
    lora_config: LoraConfig