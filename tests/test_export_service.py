import torch
import pytest

from scripts.export_model import build_meta, export_model
from service.model import (
    build_classifier,
    label_names,
    load_classifier,
    load_exported_program,
    load_meta,
    load_tokenizer,
)
from service.predictor import Predictor

TEXT = "诈骗转账请点击领取红包退款客服"


@pytest.fixture
def deployed(tiny_config, tmp_path):
    """按已训练权重的形态落一份 checkpoint，再走正式导出流程"""
    model = build_classifier(build_meta(tiny_config))
    checkpoint = tmp_path / "clf_model.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    return export_model(tiny_config, checkpoint=checkpoint, out_dir=tmp_path / "deploy")


@pytest.fixture
def predictor(deployed):
    return Predictor(deployed, device=torch.device("cpu"))


def test_export_writes_all_artifacts(deployed):
    for name in ("model.pt2", "model.pt", "predict_meta.json", "tokenizer.json",
                 "tokenizer_config.json"):
        assert (deployed / name).exists(), name


def test_meta_restores_label_names(deployed):
    assert label_names(load_meta(deployed)) == ["正常", "广告骚扰", "虚假购物"]


def test_classifier_can_be_rebuilt_and_loaded(deployed):
    model = load_classifier(deployed)
    assert model.training is False
    assert callable(load_tokenizer(deployed))


def test_exported_program_matches_eager(deployed, predictor):
    batch = predictor.encode([TEXT, "恭喜中奖"])
    program = load_exported_program(deployed)

    with torch.no_grad():
        eager = predictor.model(batch["input_ids"], batch["attention_mask"])
        exported = program.module()(batch["input_ids"], batch["attention_mask"])

    assert torch.allclose(eager["logits"], exported["logits"], atol=1e-5)
    assert torch.allclose(eager["expert_logits"], exported["expert_logits"], atol=1e-5)
    assert torch.allclose(eager["gate"], exported["gate"], atol=1e-5)


def test_exported_program_supports_dynamic_shapes(deployed, predictor):
    program = load_exported_program(deployed).module()
    short = predictor.encode(["诈骗转账"])
    long = predictor.encode(["诈骗转账请点击领取红包退款客服投资贷款中奖"])

    with torch.no_grad():
        out_short = program(short["input_ids"], short["attention_mask"])
        out_long = program(long["input_ids"], long["attention_mask"])

    assert out_short["logits"].shape[0] == 1
    assert out_long["logits"].shape[0] == 1
    assert out_long["logits"].shape[1] == len(predictor.labels)


def test_predict_returns_probability_distribution(predictor):
    result = predictor.predict(TEXT)

    assert result["label"] in predictor.labels
    assert 0.0 <= result["confidence"] <= 1.0
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert sum(result["expert_weights"].values()) == pytest.approx(1.0, abs=1e-4)
    assert result["probabilities"][result["label"]] == pytest.approx(result["confidence"])
    assert set(result["expert_weights"]) == {"正常", "广告骚扰", "虚假购物"}


def test_predict_batch_returns_one_result_per_text(predictor):
    results = predictor.predict_batch([TEXT, "恭喜中奖", "点击链接领取"])
    assert len(results) == 3
    assert all(item["label"] in predictor.labels for item in results)


def test_explain_ranks_token_attributions(predictor):
    result = predictor.explain(TEXT, steps=8, top_k=5)

    assert result["target_label"] == result["label"]
    assert len(result["attributions"]) == 5
    scores = [abs(item["score"]) for item in result["attributions"]]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(item["token"], str) for item in result["attributions"])


def test_explain_accepts_explicit_target(predictor):
    result = predictor.explain(TEXT, target=0, steps=4)
    assert result["target_label"] == predictor.labels[0]


def test_missing_exported_graph_raises(tmp_path, tiny_config):
    from service.model import load_exported_program

    with pytest.raises(FileNotFoundError):
        load_exported_program(tmp_path)