import pathlib

from peft import LoraConfig, get_peft_model
import torch.nn as nn
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from train.config.model import ModelConfig
from train.config.peft import PeftConfig
from train.model.fine_tuning import FINE_TUNING_METHODS

class Backbone(nn.Module):
    cfg: ModelConfig
    peft_cfg: PeftConfig
    is_mlm: bool

    model: nn.Module
    tokenizer = None

    def __init__(self, model: ModelConfig, peft: PeftConfig = None, is_mlm: bool = False):
        super().__init__()
        self.cfg = model
        self.peft_cfg = peft
        self.is_mlm = is_mlm

    def build_model(self):
        if self.is_mlm:
            base_model = AutoModelForMaskedLM.from_pretrained(pathlib.Path(self.cfg.path))
        else:
            base_model = AutoModel.from_pretrained(pathlib.Path(self.cfg.path))

        if self.peft_cfg.method.strip().lower() in FINE_TUNING_METHODS:
            self.model = get_peft_model(base_model, LoraConfig(
                r = self.peft_cfg.lora_config.r,
                lora_alpha = self.peft_cfg.lora_config.alpha,
                lora_dropout = self.peft_cfg.lora_config.dropout,
                target_modules = self.peft_cfg.lora_config.target_modules,
                bias = self.peft_cfg.lora_config.bias,
                use_dora = self.peft_cfg.method.strip().lower() == "dora",
            ))
        else:
            self.model = base_model
    def build_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(pathlib.Path(self.cfg.path))

    def build(self):
        self.build_model()
        self.build_tokenizer()