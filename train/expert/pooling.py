import torch
import torch.nn as nn


class MeanPooling(nn.Module):
    """按 attention_mask 加权求平均，输出 [batch, dim]"""

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class MaxPooling(nn.Module):
    """按 attention_mask 取逐维最大值，输出 [batch, dim]

    PAD 位置填 `dtype.min`，保证它们永远不会被 max 选中
    """

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).bool()
        neg = torch.finfo(hidden.dtype).min
        return hidden.masked_fill(~mask, neg).max(dim=1).values


class MinPooling(nn.Module):
    """按 attention_mask 取逐维最小值，输出 [batch, dim]

    与 MaxPooling 对称，PAD 位置填 `dtype.max`，保证不被 min 选中
    """

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).bool()
        pos = torch.finfo(hidden.dtype).max
        return hidden.masked_fill(~mask, pos).min(dim=1).values


class MedianPooling(nn.Module):
    """逐维取真实 token 的**下中位数**，输出 [batch, dim]
    """

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.bool()
        pos = torch.finfo(hidden.dtype).max
        big = hidden.masked_fill(~mask.unsqueeze(-1), pos)
        sorted_h, _ = big.sort(dim=1)
        n = mask.sum(dim=1).clamp(min=1)
        mid = ((n - 1) // 2).clamp(min=0)
        idx = mid.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
        return sorted_h.gather(1, idx).squeeze(1)


class CLSPooling(nn.Module):
    """取 [CLS] token 表示，输出 [batch, dim]"""

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return hidden[:, 0]


class LastMeanPooling(nn.Module):
    """取**后 1/3 真实 token** 的均值，输出 [batch, dim]
    """

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        n = attention_mask.sum(dim=1)                                  # [B] 真实长度
        start = (n - (n + 2) // 3).clamp(min=0)                        # 后 1/3 的起点
        ar = torch.arange(hidden.size(1), device=hidden.device).unsqueeze(0)
        last = ((ar >= start.unsqueeze(1)) & attention_mask.bool()).unsqueeze(-1).float()
        return (hidden * last).sum(dim=1) / last.sum(dim=1).clamp(min=1e-9)


class AttentionPooling(nn.Module):
    """注意力池化：可学习查询向量对每个 token 打分，softmax 后加权求和

    heads=1 输出 [batch, dim]
    heads=H 输出 [batch, H*dim]（各头加权结果拼接）
    """

    def __init__(self, dim: int, heads: int = 1):
        super().__init__()
        if heads < 1:
            raise ValueError(f"heads 必须 >= 1，收到 {heads}")
        self.heads = heads
        if heads == 1:
            self.query = nn.Parameter(torch.randn(dim) / dim**0.5)
        else:
            self.query = nn.Parameter(torch.randn(heads, dim) / dim**0.5)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.heads == 1:
            q = self.query                                             # [D]
            scores = (hidden @ q) / q.shape[0] ** 0.5                  # [B, L]
            scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
            weights = scores.softmax(dim=-1).unsqueeze(-1)             # [B, L, 1]
            return (hidden * weights).sum(dim=1)                       # [B, D]

        q = self.query                                                # [H, D]
        scores = torch.einsum("bld,hd->bhl", hidden, q) / q.shape[-1] ** 0.5   # [B, H, L]
        scores = scores.masked_fill(
            attention_mask.unsqueeze(1) == 0, torch.finfo(scores.dtype).min
        )
        weights = scores.softmax(dim=-1).unsqueeze(-1)                # [B, H, L, 1]
        out = (hidden.unsqueeze(1) * weights).sum(dim=2)              # [B, H, D]
        return out.flatten(1)                                         # [B, H*D]


class MeanMaxPooling(nn.Module):
    """均值池化与最大池化拼接，输出 [batch, 2*dim]"""

    def __init__(self):
        super().__init__()
        self.mean = MeanPooling()
        self.max = MaxPooling()

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.mean(hidden, attention_mask), self.max(hidden, attention_mask)], dim=-1)


class CLSMeanPooling(nn.Module):
    """[CLS] 表示与均值池化拼接，输出 [batch, 2*dim]
    """

    def __init__(self):
        super().__init__()
        self.cls = CLSPooling()
        self.mean = MeanPooling()

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.cls(hidden, attention_mask), self.mean(hidden, attention_mask)], dim=-1)


class MaxAttnPooling(nn.Module):
    """最大池化与注意力池化拼接，输出 [batch, (1 + heads)*dim]
    """

    def __init__(self, dim: int, heads: int = 1):
        super().__init__()
        self.max = MaxPooling()
        self.attn = AttentionPooling(dim, heads)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.max(hidden, attention_mask), self.attn(hidden, attention_mask)], dim=-1)


# 各算子的输出倍数（表内是 heads=1 的取值）；注意力类随 heads 变化，
# 取 in_dim 请统一走 pooling_output_mult()，不要直接查这张表
POOLING_OUTPUT_MULT: dict[str, int] = {
    "mean": 1,
    "max": 1,
    "min": 1,
    "median": 1,
    "cls": 1,
    "last_mean": 1,
    "attention": 1,
    "mean_max": 2,
    "cls_mean": 2,
    "max_attn": 2,
}

# 需要 heads 参数的池化算子；新增此类算子时三处都要改：本元组、
# POOLING_OUTPUT_MULT 与 build_pooling 的分派
POOLINGS_WITH_PARAMS = ("attention", "max_attn")

_SIMPLE = {
    "mean": MeanPooling,
    "max": MaxPooling,
    "min": MinPooling,
    "median": MedianPooling,
    "cls": CLSPooling,
    "last_mean": LastMeanPooling,
    "mean_max": MeanMaxPooling,
    "cls_mean": CLSMeanPooling,
}
_ATTENTION = {"attention": AttentionPooling, "max_attn": MaxAttnPooling}

AVAILABLE_POOLINGS = tuple(sorted(list(_SIMPLE) + list(_ATTENTION)))


def pooling_output_mult(name: str, heads: int = 1) -> int:
    if name == "attention":
        return heads
    if name == "max_attn":
        return 1 + heads
    if name not in POOLING_OUTPUT_MULT:
        raise KeyError(f"未知池化算子 {name!r}；可用：{list(AVAILABLE_POOLINGS)}")
    return POOLING_OUTPUT_MULT[name]


def build_pooling(name: str, dim: int, heads: int = 1) -> nn.Module:
    if name in _ATTENTION:
        return _ATTENTION[name](dim, heads)
    if name in _SIMPLE:
        return _SIMPLE[name]()
    raise KeyError(f"未知池化算子 {name!r}；可用：{list(AVAILABLE_POOLINGS)}")
