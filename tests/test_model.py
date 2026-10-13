import pytest
import torch

from train.config.expert import ExpertConfig
from train.expert.pooling import (
    AVAILABLE_POOLINGS,
    POOLINGS_WITH_PARAMS,
    build_pooling,
    pooling_output_mult,
)

HIDDEN = 32
BATCH, LENGTH = 2, 8


@pytest.fixture
def hidden_and_mask():
    torch.manual_seed(0)
    hidden = torch.randn(BATCH, LENGTH, HIDDEN)
    mask = torch.ones(BATCH, LENGTH, dtype=torch.long)
    mask[0, 5:] = 0
    return hidden, mask


def test_backbone_is_registered_as_submodule(tiny_classifier):
    names = [name for name, _ in tiny_classifier.named_parameters()]
    assert any(name.startswith("backbone.model.") for name in names)
    assert any(key.startswith("backbone.model.") for key in tiny_classifier.state_dict())


def test_forward_shapes(tiny_classifier):
    input_ids = torch.randint(0, 40, (BATCH, LENGTH))
    mask = torch.ones(BATCH, LENGTH, dtype=torch.long)
    out = tiny_classifier(input_ids, mask)

    assert out["logits"].shape == (BATCH, 3)
    assert out["expert_logits"].shape == (BATCH, 3, 3)
    assert out["gate"].shape == (BATCH, 3)
    assert torch.allclose(out["gate"].sum(dim=-1), torch.ones(BATCH), atol=1e-5)


def test_clf_forward_shape(tiny_classifier):
    input_ids = torch.randint(0, 40, (BATCH, LENGTH))
    mask = torch.ones(BATCH, LENGTH, dtype=torch.long)
    assert tiny_classifier.clf_forward(input_ids, mask).shape == (BATCH, 3)


def test_forward_from_embeds_matches_forward(tiny_classifier):
    input_ids = torch.randint(0, 40, (BATCH, LENGTH))
    mask = torch.ones(BATCH, LENGTH, dtype=torch.long)
    mask[0, 5:] = 0

    with torch.no_grad():
        from_ids = tiny_classifier(input_ids, mask)
        from_embeds = tiny_classifier.forward_from_embeds(
            tiny_classifier.embed(input_ids), mask
        )

    assert torch.allclose(from_ids["logits"], from_embeds["logits"], atol=1e-6)
    assert torch.allclose(from_ids["gate"], from_embeds["gate"], atol=1e-6)


def test_embed_produces_gradient_from_embeds(tiny_classifier):
    input_ids = torch.randint(0, 40, (BATCH, LENGTH))
    mask = torch.ones(BATCH, LENGTH, dtype=torch.long)
    embeds = tiny_classifier.embed(input_ids).detach().requires_grad_(True)

    logits = tiny_classifier.forward_from_embeds(embeds, mask)["logits"]
    logits[:, 0].sum().backward()

    assert embeds.grad is not None
    assert torch.isfinite(embeds.grad).all()
    assert (embeds.grad != 0).any()


def test_all_poolings_produce_expected_output_dim(hidden_and_mask):
    hidden, mask = hidden_and_mask
    for name in AVAILABLE_POOLINGS:
        pooling = build_pooling(name, HIDDEN)
        out = pooling(hidden, mask)
        assert out.shape == (BATCH, HIDDEN * pooling_output_mult(name)), name


def test_attention_pooling_heads_double_output_dim(hidden_and_mask):
    hidden, mask = hidden_and_mask
    for name in ("attention", "max_attn"):
        out = build_pooling(name, HIDDEN, heads=4)(hidden, mask)
        assert out.shape == (BATCH, HIDDEN * pooling_output_mult(name, heads=4)), name


def test_pooling_ignores_padded_positions(hidden_and_mask):
    hidden, mask = hidden_and_mask
    for name in AVAILABLE_POOLINGS:
        if name in POOLINGS_WITH_PARAMS:
            continue
        pooling = build_pooling(name, HIDDEN)
        first = pooling(hidden, mask)
        perturbed = hidden.clone()
        perturbed[0, 5:] += 100.0  # 只扰动被 mask 掉的位置
        assert torch.allclose(first, pooling(perturbed, mask), atol=1e-5), name


def test_unknown_pooling_raises():
    with pytest.raises(KeyError):
        build_pooling("nope", HIDDEN)


def test_expert_rejects_heads_without_attention_pooling():
    from train.expert.experts import Expert

    cfg = ExpertConfig(
        index=0, id="x", name="x", pooling="mean", hidden=8, layers=1, augment=[], heads=2
    )
    with pytest.raises(ValueError):
        Expert(cfg, HIDDEN, 3, 0.0)