import pytest
import torch

from service.attribution import integrated_gradients

TEXT = "诈骗转账请点击领取红包退款客服投资"


def encode(tokenizer, texts: list[str]):
    enc = tokenizer(
        texts, return_tensors="pt", truncation=True, max_length=16, padding=True
    )
    return enc["input_ids"], enc["attention_mask"]


def test_scores_shape_and_default_target(tiny_classifier, tiny_tokenizer):
    input_ids, mask = encode(tiny_tokenizer, [TEXT])
    scores, targets = integrated_gradients(
        tiny_classifier, input_ids, mask, steps=8,
        baseline_token_id=tiny_tokenizer.pad_token_id,
    )

    assert scores.shape == input_ids.shape
    with torch.no_grad():
        expected = int(tiny_classifier(input_ids, mask)["logits"].argmax(dim=-1)[0])
    assert int(targets[0]) == expected


def test_completeness_axiom(tiny_classifier, tiny_tokenizer):
    input_ids, mask = encode(tiny_tokenizer, [TEXT])
    scores, targets = integrated_gradients(
        tiny_classifier, input_ids, mask, steps=64,
        baseline_token_id=tiny_tokenizer.pad_token_id,
    )
    target = int(targets[0])

    with torch.no_grad():
        baseline = torch.full_like(input_ids, tiny_tokenizer.pad_token_id)
        f_input = tiny_classifier(input_ids, mask)["logits"][0, target].item()
        f_baseline = tiny_classifier(baseline, mask)["logits"][0, target].item()

    assert scores.sum() == pytest.approx(f_input - f_baseline, abs=1e-3, rel=1e-2)


def test_explicit_target_is_respected(tiny_classifier, tiny_tokenizer):
    input_ids, mask = encode(tiny_tokenizer, [TEXT])
    _, targets = integrated_gradients(
        tiny_classifier, input_ids, mask, target_class=0, steps=4
    )
    assert list(targets) == [0]


def test_batched_target_class_is_supported(tiny_classifier, tiny_tokenizer):
    input_ids, mask = encode(tiny_tokenizer, [TEXT, "恭喜中奖"])
    scores, targets = integrated_gradients(
        tiny_classifier, input_ids, mask, target_class=torch.tensor([0, 2]), steps=4
    )
    assert scores.shape == input_ids.shape
    assert list(targets) == [0, 2]


def test_padded_positions_score_zero(tiny_classifier, tiny_tokenizer):
    input_ids, mask = encode(tiny_tokenizer, [TEXT, "恭喜中奖"])
    scores, _ = integrated_gradients(
        tiny_classifier, input_ids, mask, steps=4,
        baseline_token_id=tiny_tokenizer.pad_token_id,
    )
    padded = mask == 0
    assert padded.any()
    assert torch.allclose(
        torch.from_numpy(scores)[padded], torch.zeros(int(padded.sum())), atol=1e-6
    )


def test_invalid_steps_raises(tiny_classifier, tiny_tokenizer):
    input_ids, mask = encode(tiny_tokenizer, [TEXT])
    with pytest.raises(ValueError):
        integrated_gradients(tiny_classifier, input_ids, mask, steps=0)


def test_target_count_mismatch_raises(tiny_classifier, tiny_tokenizer):
    input_ids, mask = encode(tiny_tokenizer, [TEXT])
    with pytest.raises(ValueError):
        integrated_gradients(
            tiny_classifier, input_ids, mask, target_class=torch.tensor([0, 1]), steps=4
        )