"""FastAPI 服务集成测试
"""

from __future__ import annotations

import time

import pytest
import torch
from fastapi.testclient import TestClient

from scripts.export_model import build_meta, export_model
from service.api import create_app
from service.model import build_classifier

TEXT = "诈骗转账请点击领取红包退款客服"


@pytest.fixture
def deployed(tiny_config, tmp_path):
    model = build_classifier(build_meta(tiny_config))
    checkpoint = tmp_path / "clf_model.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    return export_model(tiny_config, checkpoint=checkpoint, out_dir=tmp_path / "deploy")


@pytest.fixture
def client(deployed):
    app = create_app(deployed, workers=1, ttl=60)
    with TestClient(app) as client:
        yield client


def _await(client: TestClient, ticket: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/v1/infer/{ticket}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"任务 {ticket} 未在 {timeout}s 内结束")


def _infer(client: TestClient, **payload) -> dict:
    response = client.post("/api/v1/infer", json=payload)
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "queued"
    return _await(client, body["ticket"])


def test_health_reports_loaded_model(client):
    body = client.get("/api/v1/health").json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["error"] is None
    assert body["device"] in {"cpu", "cuda"}
    # 导图在 CPU 上导出，仅在 CPU 启用；其余设备回退 eager
    expected_backend = "pt2" if body["device"] == "cpu" else "eager"
    assert body["forward_backend"] == expected_backend
    assert body["attribution_backend"] == "eager"
    assert body["num_classes"] == 3


def test_meta_exposes_static_catalog(client):
    body = client.get("/api/v1/meta").json()

    assert body["num_classes"] == 3
    assert body["max_length"] == 16
    assert len(body["labels"]) == 3
    assert all(set(item) == {"id", "key", "name"} for item in body["labels"])
    assert all(
        set(item) == {"id", "key", "name", "pooling"} for item in body["experts"]
    )
    assert [item["name"] for item in body["labels"]] == ["正常", "广告骚扰", "虚假购物"]


def test_infer_returns_item_schema(client):
    body = _infer(client, texts=[TEXT])
    assert body["status"] == "done", body["error"]

    items = body["result"]["items"]
    assert len(items) == 1
    item = items[0]
    assert set(item) == {
        "text",
        "label_id",
        "label_key",
        "label_name",
        "confidence",
        "probabilities",
        "experts_opinion",
        "explanation",
    }
    assert item["text"] == TEXT
    assert 0 <= item["label_id"] < 3
    assert len(item["probabilities"]) == 3
    assert sum(item["probabilities"]) == pytest.approx(1.0, abs=1e-4)
    assert item["probabilities"][item["label_id"]] == pytest.approx(item["confidence"])
    assert item["explanation"] is None


def test_experts_opinion_is_object_array(client):
    item = _infer(client, texts=[TEXT])["result"]["items"][0]

    experts = item["experts_opinion"]
    assert len(experts) == 3
    for entry in experts:
        assert set(entry) == {"id", "key", "probabilities", "weight"}
        assert len(entry["probabilities"]) == 3
        assert sum(entry["probabilities"]) == pytest.approx(1.0, abs=1e-4)
    assert sum(entry["weight"] for entry in experts) == pytest.approx(1.0, abs=1e-4)


def test_infer_schema_keys_are_ascii(client):
    item = _infer(client, texts=[TEXT])["result"]["items"][0]

    for key in item:
        assert key.isascii(), key
    for entry in item["experts_opinion"]:
        for key in entry:
            assert key.isascii(), key


def test_infer_supports_batch_and_explain(client):
    body = _infer(client, texts=[TEXT, "恭喜中奖"], explain=True, top_k=3, steps=4)
    assert body["status"] == "done", body["error"]

    items = body["result"]["items"]
    assert len(items) == 2
    for item in items:
        explanation = item["explanation"]
        assert set(explanation) == {"target_label_id", "attributions"}
        assert 0 <= explanation["target_label_id"] < 3
        scores = [abs(entry["score"]) for entry in explanation["attributions"]]
        assert len(scores) <= 3
        assert scores == sorted(scores, reverse=True)
        assert all(set(entry) == {"token", "score"} for entry in explanation["attributions"])


def test_infer_respects_explicit_target(client):
    item = _infer(client, texts=[TEXT], explain=True, target=0, steps=4)["result"]["items"][0]
    assert item["explanation"]["target_label_id"] == 0


def test_infer_rejects_out_of_range_target(client):
    response = client.post("/api/v1/infer", json={"texts": [TEXT], "target": 99})
    assert response.status_code == 422


def test_infer_rejects_empty_texts(client):
    assert client.post("/api/v1/infer", json={"texts": []}).status_code == 422


def test_unknown_ticket_returns_404(client):
    assert client.get("/api/v1/infer/deadbeef").status_code == 404