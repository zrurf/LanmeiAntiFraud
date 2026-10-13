"""测试共享夹具
"""

from __future__ import annotations

import random
from pathlib import Path

import lance
import pyarrow as pa
import pytest
from transformers import AutoModel, AutoTokenizer, BertConfig, BertTokenizer

from train.config.config import Config
from train.config.data import DataConfig, OutputConfig
from train.config.expert import ExpertConfig
from train.config.model import ModelConfig
from train.config.peft import LoraConfig, PeftConfig
from train.config.router import RouterConfig
from train.config.train import (
    DefaultLossConfig,
    LossConfig,
    PretrainStageConfig,
    TrainConfig,
    TrainStageConfig,
)

# tiny 词表：特殊符号 + 覆盖测试文本用到的汉字
_SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
_CHARS = "诈骗转账刷单退款客服投资贷款中奖红包恭喜您好我他的是不在快点击领取关于链接验证码"
_VOCAB = _SPECIALS + list(dict.fromkeys(_CHARS))

_HIDDEN = 32
_MAX_LENGTH = 16


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> Path:
    model_dir = tmp_path_factory.mktemp("tiny_macbert")
    config = BertConfig(
        vocab_size=len(_VOCAB),
        hidden_size=_HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=_HIDDEN * 2,
        max_position_embeddings=_MAX_LENGTH * 4,
    )
    AutoModel.from_config(config).save_pretrained(model_dir)

    vocab_file = model_dir / "vocab.txt"
    vocab_file.write_text("\n".join(_VOCAB) + "\n", encoding="utf-8")
    BertTokenizer(vocab_file=str(vocab_file), do_lower_case=True).save_pretrained(model_dir)
    return model_dir


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_model_dir: Path):
    return AutoTokenizer.from_pretrained(str(tiny_model_dir))


@pytest.fixture
def tiny_dataset_path(tmp_path: Path) -> Path:
    """合成 Lance 数据集：train 15 条 / val 9 条 / mlm 12 条，覆盖 3 个类别"""
    rng = random.Random(0)
    texts = [
        "".join(rng.choice(_CHARS) for _ in range(20)) for _ in range(40)
    ]
    rows = {
        "id": [],
        "text": [],
        "label10": [],
        "subtype": [],
        "source": [],
        "use": [],
    }
    for index, text in enumerate(texts):
        if index < 15:
            use, label = "train", index % 3
        elif index < 24:
            use, label = "val", (index - 15) % 3
        elif index < 36:
            use, label = "mlm", 0
        else:
            use, label = "test", (index - 36) % 3
        rows["id"].append(index)
        rows["text"].append(text)
        rows["label10"].append(label)
        rows["subtype"].append("test")
        rows["source"].append("synthetic")
        rows["use"].append(use)

    uri = tmp_path / "dataset.lance"
    lance.write_dataset(pa.table(rows), str(uri))
    return uri


@pytest.fixture
def tiny_config(tiny_model_dir: Path, tiny_dataset_path: Path, tmp_path: Path) -> Config:
    experts = [
        ExpertConfig(index=0, id="normal", name="正常", pooling="mean_max", hidden=16, layers=1, augment=[]),
        ExpertConfig(index=1, id="advertising", name="广告骚扰", pooling="attention", hidden=16, layers=1, augment=["emoji_replace"]),
        ExpertConfig(index=2, id="fake_shopping", name="虚假购物", pooling="max_attn", hidden=16, layers=1, augment=["random_space"], heads=2),
    ]
    stages = TrainStageConfig(epochs=1, lr=1e-3, batch_size=4)
    return Config(
        model=ModelConfig(name="tiny", path=str(tiny_model_dir), hidden_size=_HIDDEN, dropout=0.1),
        data=DataConfig(
            dataset_path=str(tiny_dataset_path),
            corpus_path=str(tiny_dataset_path),
            max_length=_MAX_LENGTH,
        ),
        output=OutputConfig(
            output_path=str(tmp_path / "artifacts"),
            checkpoint_path=str(tmp_path / "artifacts" / "checkpoints"),
            export_path=str(tmp_path / "artifacts" / "export"),
            log_path=str(tmp_path / "artifacts" / "logs"),
        ),
        peft=PeftConfig(
            method="dora",
            lora_config=LoraConfig(
                bias="none", r=4, alpha=8, dropout=0.05, target_modules=["query", "value"]
            ),
        ),
        train=TrainConfig(
            weight_decay=0.01,
            warmup_ratio=0.05,
            mlm_pretrain=PretrainStageConfig(epochs=1, lr=1e-4, probability=0.15, batch_size=4),
            clf_train=stages,
            expert_train=stages,
            router_train=stages,
            joint_train=stages,
            num_workers=0,  # 测试不启子进程：微型数据集上启动开销远大于分词本身
        ),
        router=RouterConfig(dim=_HIDDEN, dropout=0.1),
        loss=LossConfig(
            defaults=DefaultLossConfig(
                class_weights="none",
                label_smoothing=0.0,
                focal_gamma=2.0,
                self_weight=3.0,
                other_weight=1.0,
                beta=0.99999,
                balance_weight=0.01,
            )
        ),
        experts=experts,
        seed=0,
        resume=False,
        num_classes=3,
    )


@pytest.fixture
def tiny_classifier(tiny_model_dir: Path):
    from train.expert.moe import MoEClassifier
    from train.model.backbone import Backbone

    experts = [
        ExpertConfig(index=0, id="normal", name="正常", pooling="mean_max", hidden=16, layers=1, augment=[]),
        ExpertConfig(index=1, id="adv", name="广告", pooling="attention", hidden=16, layers=1, augment=[]),
        ExpertConfig(index=2, id="shop", name="购物", pooling="max_attn", hidden=16, layers=1, augment=[], heads=2),
    ]
    backbone = Backbone(
        ModelConfig(name="tiny", path=str(tiny_model_dir), hidden_size=_HIDDEN, dropout=0.0),
        PeftConfig(method="none", lora_config=None),
        is_mlm=False,
    )
    backbone.build()
    model = MoEClassifier(backbone, 3, experts, RouterConfig(dim=_HIDDEN, dropout=0.0))
    model.eval()
    return model