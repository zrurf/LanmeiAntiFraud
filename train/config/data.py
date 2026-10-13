from dataclasses import dataclass

@dataclass
class DataConfig:
    dataset_path: str
    max_length: int
    corpus_path: str = ""

@dataclass
class OutputConfig:
    output_path: str
    checkpoint_path: str
    export_path: str
    log_path: str