from __future__ import annotations

import numpy as np
import torch

from train.expert.moe import MoEClassifier


def integrated_gradients(
    model: MoEClassifier,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_class: int | torch.Tensor | None = None,
    steps: int = 32,
    baseline_token_id: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Token 级积分梯度（Integrated Gradients）归因

    在 embedding 空间沿「基线 → 真实输入」的直线路径做积分：

        IG_i = (x_i - x'_i) · ∫_0^1 ∂F(x' + α(x - x')) / ∂x_i dα

    积分用梯形法离散为 `steps` 个区间。对目标类别 logit 满足完备性：
    `scores.sum() ≈ F(x)[target] - F(x')[target]`

    基线取 `baseline_token_id` 的 embedding，默认 PAD。

    Args:
        model: 已 `eval()` 的 MoE 分类器，需支持 `forward_from_embeds`
        input_ids: `[B, L]` 的 token id
        attention_mask: `[B, L]`
        target_class: 目标类别，标量或 `[B]`；`None` 时取真实输入下的预测类别
        steps: 路径积分的区间数，须 >= 1
        baseline_token_id: 基线 token id

    Returns:
        (scores, targets): `scores` 为 `[B, L]` 的归因分数，`targets` 为 `[B]` 的目标类别
    """
    if steps < 1:
        raise ValueError(f"steps 必须 >= 1，收到 {steps}")

    model.eval()
    input_ids = input_ids.long()
    baseline = torch.full_like(input_ids, int(baseline_token_id))

    with torch.no_grad():
        base_emb = model.embed(baseline)
        real_emb = model.embed(input_ids)
        logits = model(input_ids, attention_mask)["logits"]

    batch_size = logits.size(0)
    if target_class is None:
        targets = logits.argmax(dim=-1)
    else:
        targets = torch.as_tensor(target_class, device=logits.device).reshape(-1)
        if targets.numel() == 1:
            targets = targets.expand(batch_size)
        if targets.numel() != batch_size:
            raise ValueError(
                f"target_class 数量 ({targets.numel()}) 与 batch ({batch_size}) 不一致"
            )
        targets = targets.long()
    targets = targets.to(logits.device)

    delta = real_emb - base_emb
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=real_emb.device, dtype=real_emb.dtype)
    weights = torch.ones_like(alphas)
    weights[0] = weights[-1] = 0.5  # 梯形法：两端点各取一半权重
    row = torch.arange(batch_size, device=real_emb.device)

    grad_sum = torch.zeros_like(real_emb)
    for alpha, weight in zip(alphas, weights):
        embeds = (base_emb + alpha * delta).detach().requires_grad_(True)
        step_logits = model.forward_from_embeds(embeds, attention_mask)["logits"]
        grad = torch.autograd.grad(step_logits[row, targets].sum(), embeds)[0]
        grad_sum += weight * grad

    avg_grad = grad_sum / steps
    scores = (delta * avg_grad).sum(dim=-1)
    return scores.detach().cpu().numpy(), targets.detach().cpu().numpy()