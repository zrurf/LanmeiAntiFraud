from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    path: str
    hidden_size: int = 768
    dropout: float = 0.1