import pytest
import torch
import torch.nn as nn

from train.model.loss import ClassBalancedLoss, LoadBalanceLoss


def make_cb_loss(samples_per_cls, beta=0.999, num_classes=3, **kwargs) -> ClassBalancedLoss:
    return ClassBalancedLoss(
        samples_per_cls=samples_per_cls,
        beta=beta,
        num_classes=num_classes,
        base_loss=nn.CrossEntropyLoss(reduction="none"),
        **kwargs,
    )


def test_cb_weights_are_normalized_and_favor_rare_classes():
    loss_fn = make_cb_loss([1000, 100, 10], beta=0.999)
    weights = loss_fn.cb_weights
    assert weights.sum().item() == pytest.approx(3.0, abs=1e-5)
    assert weights[2] > weights[1] > weights[0]
    assert weights[0] < 1.0 < weights[2]


def test_beta_one_falls_back_to_equal_weights():
    loss_fn = make_cb_loss([1000, 100, 10], beta=1.0)
    assert torch.allclose(loss_fn.cb_weights, torch.ones(3))


def test_self_weight_boosts_own_class():
    logits = torch.zeros(3, 3)
    loss_fn = make_cb_loss([10, 10, 10], beta=1.0, lambda_self=3.0, self_class=1)
    own = loss_fn(logits, torch.tensor([1, 1, 1]))
    other = loss_fn(logits, torch.tensor([0, 0, 0]))
    assert own.item() == pytest.approx(3.0 * other.item(), rel=1e-5)


def test_self_weight_is_ignored_without_self_class():
    logits = torch.zeros(2, 3)
    loss_fn = make_cb_loss([10, 10, 10], beta=1.0, lambda_self=3.0)
    loss = loss_fn(logits, torch.tensor([0, 1]))
    assert torch.isfinite(loss)


def test_backward_produces_gradients():
    logits = torch.randn(4, 3, requires_grad=True)
    loss = make_cb_loss([10, 10, 10], lambda_self=3.0, self_class=0)(
        logits, torch.tensor([0, 1, 2, 0])
    )
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("beta", [-0.1, 1.5])
def test_beta_out_of_range_raises(beta):
    with pytest.raises(ValueError):
        make_cb_loss([10, 10, 10], beta=beta)


def test_samples_length_mismatch_raises():
    with pytest.raises(ValueError):
        make_cb_loss([10, 10], num_classes=3)


def test_non_positive_samples_raises():
    with pytest.raises(ValueError):
        make_cb_loss([10, 0, 5])


def test_scalar_base_loss_raises():
    loss_fn = ClassBalancedLoss(
        samples_per_cls=[10, 10, 10],
        beta=0.999,
        num_classes=3,
        base_loss=nn.CrossEntropyLoss(),
    )
    with pytest.raises(RuntimeError):
        loss_fn(torch.randn(2, 3), torch.tensor([0, 1]))


def test_self_class_out_of_range_raises():
    loss_fn = make_cb_loss([10, 10, 10], lambda_self=3.0, self_class=5)
    with pytest.raises(ValueError):
        loss_fn(torch.randn(2, 3), torch.tensor([0, 1]))


def test_load_balance_loss_uniform_gate_equals_alpha():
    gate = torch.full((8, 4), 0.25)
    assert LoadBalanceLoss(alpha=0.01)(gate).item() == pytest.approx(0.01, rel=1e-6)


def test_load_balance_loss_is_differentiable_and_finite():
    logits = torch.randn(8, 4, requires_grad=True)
    loss = LoadBalanceLoss(alpha=0.01)(torch.softmax(logits, dim=-1))
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_load_balance_loss_penalizes_concentrated_routing():
    uniform = torch.full((8, 4), 0.25)
    concentrated = torch.tensor([[0.97, 0.01, 0.01, 0.01]] * 8)
    fn = LoadBalanceLoss(alpha=0.01)
    assert fn(concentrated).item() > fn(uniform).item()


def test_load_balance_loss_rejects_non_2d():
    with pytest.raises(ValueError):
        LoadBalanceLoss()(torch.rand(4))