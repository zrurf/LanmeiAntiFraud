from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _reduce(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """统一的 reduction 分派"""
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(
        f"Unsupported reduction: {reduction!r}. Use 'mean' | 'sum' | 'none'."
    )


def _self_weight_mask(
    labels: torch.Tensor,
    lambda_self: float | None,
    self_class: int | None,
    num_classes: int | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    """专家分工权重向量

    当 `labels_i == self_class` 时权重为 `lambda_self`，其余为 1

    若 `lambda_self` / `self_class` 任一为 `None`，返回 `None` 表示不启用
    """
    if lambda_self is None or self_class is None:
        return None
    if num_classes is not None and not (0 <= self_class < num_classes):
        raise ValueError(
            f"self_class ({self_class}) out of range [0, {num_classes})."
        )
    if dtype is None:
        dtype = torch.get_default_dtype()
    ones = torch.ones((), device=labels.device, dtype=dtype)
    lam = torch.as_tensor(lambda_self, device=labels.device, dtype=dtype)
    return torch.where(labels == self_class, lam, ones)


class WeightedCrossEntropy(nn.Module):
    """加权交叉熵损失

    逐样本损失为:
        L_i = w_{y_i} * m_i * CE(logits_i, y_i)

    其中:
    - `w_{y_i}`: 全局类别权重（可选），由 `class_weights` 提供，用于处理类别不平衡
    - `m_i`    : 二阶段专家分工权重。当 `y_i == self_class` 时取 `lambda_self`，否则取 1。
      阶段一不启用时应传入 `lambda_self=None`

    Args:
        class_weights: 长度为 `C` 的全局类别权重张量或可转换为张量的序列；
            `None` 表示不加权（等价于普通 CE）。
        lambda_self:   对自己负责类别的加权系数（阶段二）；`None` 表示不启用
        self_class:    本专家负责的类别 id；仅在 `lambda_self` 不为 `None` 时生效
        reduction:     `"mean"` | `"sum"` | `"none"`，默认 `"mean"`

    Inputs:
        logits: `[B, C]` 未归一化分数
        labels: `[B]` 的 `int64` 标签

    Returns:
        torch.Tensor: 标量损失（`reduction != "none"`）或逐样本损失 `[B]`

    Note:
        若 `class_weights` 不为 None，会自动注册为 buffer，随 `.to(device)` / `.cuda()` 一起迁移
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        lambda_self: float | None = None,
        self_class: int | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"Unsupported reduction: {reduction!r}. Use 'mean' | 'sum' | 'none'."
            )
        if class_weights is not None and not isinstance(class_weights, torch.Tensor):
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
        if class_weights is not None:
            class_weights = class_weights.detach().float()
        self.register_buffer("class_weights", class_weights)

        self.lambda_self = lambda_self
        self.self_class = self_class
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError(
                f"WeightedCrossEntropy expects logits of shape [B, C], got {tuple(logits.shape)}."
            )
        if logits.size(0) != labels.size(0):
            raise ValueError(
                f"Batch size mismatch: logits {logits.size(0)} vs labels {labels.size(0)}."
            )
        if labels.dtype != torch.long:
            labels = labels.long()

        num_classes = logits.size(-1)
        weight = self.class_weights
        if weight is not None:
            if weight.numel() != num_classes:
                raise ValueError(
                    f"class_weights length ({weight.numel()}) != num_classes ({num_classes})."
                )
            weight = weight.to(device=logits.device, dtype=logits.dtype)

        per_sample = F.cross_entropy(
            logits, labels, weight=weight, reduction="none"
        )
        mask = _self_weight_mask(
            labels,
            self.lambda_self,
            self.self_class,
            num_classes=num_classes,
            dtype=per_sample.dtype,
        )
        if mask is not None:
            per_sample = per_sample * mask
        return _reduce(per_sample, self.reduction)

    def extra_repr(self) -> str:
        return (
            f"lambda_self={self.lambda_self}, self_class={self.self_class}, "
            f"reduction={self.reduction}"
        )


class LoadBalanceLoss(nn.Module):
    """负载均衡损失

    损失为:
        L_{bal} = \\alpha \\cdot M \\cdot \\sum_{i=1}^{M} f_i \\cdot p_i

    其中:
    - `f_i`: 由 `argmax` 硬分配统计得到的专家 i 的样本占比（detached，不参与梯度）
    - `p_i`: 软路由概率在 batch 维度的均值（梯度仅经此回传）

    Args:
        alpha: 负载均衡损失系数，默认 0.01

    Inputs:
        gate: `[B, M]` 的软路由概率（通常为 softmax 输出，行和为 1）

    Returns:
        torch.Tensor: 标量负载均衡损失
    """

    def __init__(self, alpha: float = 0.01) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.dim() != 2:
            raise ValueError(
                f"LoadBalanceLoss expects gate of shape [B, M], got {tuple(gate.shape)}."
            )
        num_experts = gate.shape[1]
        batch_size = gate.shape[0]
        if batch_size == 0:
            return gate.sum() * 0.0

        # 计数与均值在 fp32 下算：AMP 时 gate 为 bf16，scatter_add_ 与求均值的
        # 精度损失会直接体现在负载均衡项上
        gate = gate.float()

        # argmax 无梯度，硬分配不会回传梯度，梯度只经 p 回传
        assign = gate.argmax(dim=-1)
        f = torch.zeros(num_experts, device=gate.device, dtype=gate.dtype)
        ones = torch.ones(batch_size, device=gate.device, dtype=gate.dtype)
        f.scatter_add_(0, assign, ones)
        f = f.detach() / batch_size
        p = gate.mean(dim=0)
        return self.alpha * num_experts * (f * p).sum()

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"


class FocalLoss(nn.Module):
    """Focal Loss

    对 `num_classes == 2` 走二分类 `BCEWithLogitsLoss` 分支，
    `num_classes > 2` 走多分类 `cross_entropy` 分支

    Args:
        num_classes: 类别数（须 >= 2）
        alpha:       平衡系数，默认 1.0（不做额外平衡）
        gamma:       聚焦参数，默认 2.0；`gamma == 0` 且 `alpha == 1` 时退化为普通 CE
        reduction:   `"mean"` | `"sum"` | `"none"`，默认 `"mean"`

    Inputs:
        logits:  `[B, num_classes]`（多分类）或 `[B]` / `[B, 1]`（二分类）
        targets: `[B]` 的标签；二分类时为 float 0/1 或可直接转换为 float

    Returns:
        torch.Tensor: 标量或逐样本损失
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(
                f"num_classes must be >= 2, got {num_classes}."
            )
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"Unsupported reduction: {reduction!r}. Use 'mean' | 'sum' | 'none'."
            )
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 2:
            if logits.dim() == 2:
                if logits.size(-1) != 1:
                    raise ValueError(
                        "For num_classes == 2, logits must be [B] or [B, 1], "
                        f"got {tuple(logits.shape)}. Use CrossEntropyLoss for [B, 2]."
                    )
                logits = logits.squeeze(-1)
            elif logits.dim() != 1:
                raise ValueError(
                    f"For num_classes == 2, logits must be [B] or [B, 1], got {tuple(logits.shape)}."
                )
            targets = targets.reshape(logits.shape).to(logits.dtype)
            ce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
        else:
            if targets.dtype != torch.long:
                targets = targets.long()
            ce_loss = F.cross_entropy(logits, targets, reduction="none")

        pt = torch.exp(-ce_loss)
        if self.alpha != 1.0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        else:
            alpha_t = 1.0
        focal = alpha_t * (1.0 - pt) ** self.gamma * ce_loss
        return _reduce(focal, self.reduction)

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, alpha={self.alpha}, "
            f"gamma={self.gamma}, reduction={self.reduction}"
        )


class ClassBalancedLoss(nn.Module):
    """类别平衡损失

    权重计算:
        effective_num = 1 - beta ** n_c
        w_c           = (1 - beta) / effective_num
        w_c           = w_c / sum(w_c) * num_classes

    `beta == 1.0` 时退化为等权（避免 0/0）。

    逐样本损失:
        L_i = w_{y_i} * m_i * base_loss(logits_i, y_i)

    其中 `m_i` 为二阶段专家分工权重（`y_i == self_class` 时取 `lambda_self`）。

    Args:
        samples_per_cls: 每个类别的样本数；list / ndarray / Tensor，长度为 `num_classes`
        beta:            平滑系数，常取 0.9 / 0.99 / 0.999 / 1.0，须在 [0, 1] 内
        num_classes:     类别数（须 >= 2）
        base_loss:       基础损失 (`nn.Module`)，**必须以 `reduction="none"` 构造**，
                         返回逐样本损失。可用 `nn.CrossEntropyLoss(reduction="none")`
                         或 `FocalLoss(..., reduction="none")`。
        lambda_self:     二阶段专家分工加权系数；`None` 表示不启用（阶段一）
        self_class:      本专家负责的类别 id
        reduction:       `"mean"` | `"sum"`，默认 `"mean"`

    Note:
        若 `base_loss` 自身也带类权重（如 `CrossEntropyLoss(weight=...)`），
        将与 CB 权重叠加，请按需关闭其中一处。

    Inputs:
        logits: `[B, C]` 未归一化分数
        targets: `[B]` 的 `int64` 标签

    Returns:
        torch.Tensor: 标量损失
    """

    def __init__(
        self,
        samples_per_cls,
        beta: float,
        num_classes: int,
        base_loss: nn.Module,
        lambda_self: float | None = None,
        self_class: int | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}.")
        if reduction not in ("mean", "sum"):
            raise ValueError(
                f"Unsupported reduction: {reduction!r}. Use 'mean' | 'sum'."
            )
        if not (0.0 <= float(beta) <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {beta}.")

        if isinstance(samples_per_cls, torch.Tensor):
            samples = samples_per_cls.detach().cpu().numpy()
        else:
            samples = np.asarray(samples_per_cls)
        samples = samples.reshape(-1).astype(np.float64)

        if samples.shape[0] != num_classes:
            raise ValueError(
                f"samples_per_cls length ({samples.shape[0]}) != "
                f"num_classes ({num_classes})."
            )
        if np.any(samples <= 0):
            raise ValueError("samples_per_cls must be strictly positive.")

        self.num_classes = int(num_classes)
        self.beta = float(beta)
        self.lambda_self = lambda_self
        self.self_class = self_class
        self.reduction = reduction

        self.base_loss = base_loss

        self.register_buffer(
            "cb_weights",
            self._compute_weights(samples, self.beta, self.num_classes),
        )

    @staticmethod
    def _compute_weights(
        samples: np.ndarray, beta: float, num_classes: int
    ) -> torch.Tensor:
        if abs(beta - 1.0) < 1e-12:
            weights = np.ones(num_classes, dtype=np.float64)
        else:
            effective_num = 1.0 - np.power(beta, samples)
            # 数值保护，避免样本数极大时下溢到 0
            effective_num = np.maximum(effective_num, 1e-12)
            weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes
        return torch.as_tensor(weights, dtype=torch.float32)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dtype != torch.long:
            targets = targets.long()

        base_loss = self.base_loss(logits, targets)
        if base_loss.dim() == 0:
            raise RuntimeError(
                "ClassBalancedLoss requires a per-sample base loss. "
                "Construct base_loss with reduction='none'."
            )
        base_loss = base_loss.reshape(-1)
        if base_loss.numel() != targets.numel():
            raise RuntimeError(
                f"base_loss numel ({base_loss.numel()}) != targets numel ({targets.numel()}). "
                "Ensure base_loss is constructed with reduction='none'."
            )

        weights = self.cb_weights.index_select(0, targets).to(base_loss.dtype)

        mask = _self_weight_mask(
            targets,
            self.lambda_self,
            self.self_class,
            num_classes=self.num_classes,
            dtype=base_loss.dtype,
        )
        if mask is not None:
            weights = weights * mask

        loss = weights * base_loss
        return _reduce(loss, self.reduction)

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, beta={self.beta}, "
            f"lambda_self={self.lambda_self}, self_class={self.self_class}, "
            f"reduction={self.reduction}"
        )