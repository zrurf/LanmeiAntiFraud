"""长度分桶采样器
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

from torch.utils.data import DataLoader, Sampler

__all__ = ["LengthGroupedSampler", "set_loader_epoch"]


class LengthGroupedSampler(Sampler[list[int]]):
    """按长度分桶组批，逐轮产出 batch 索引列表

    Args:
        lengths: 每个样本的长度（token 数的近似值），按索引可取
        batch_size: 批大小
        mega_batch_mult: 大块大小 = `batch_size * mega_batch_mult`；越大批内越整齐，
            但同一样本与哪些样本同批就越固定
        shuffle: `True` 供训练使用，逐轮重新打乱；`False` 供评估使用，全量按长度排序后
            组批（评估指标与顺序无关），批次完全确定
        seed: 打乱与分桶的随机种子
        drop_last: 丢弃最后一个不足批的 batch

    Note:
        大块大小取 `batch_size` 的整数倍，因此只有全局最后一批可能不足批，
        `__len__` 与实际产出的批数严格一致。
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        mega_batch_mult: int = 50,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size 必须 >= 1，收到 {batch_size}")
        if mega_batch_mult < 1:
            raise ValueError(f"mega_batch_mult 必须 >= 1，收到 {mega_batch_mult}")
        self.lengths = lengths
        self.batch_size = int(batch_size)
        self.mega_batch_size = self.batch_size * int(mega_batch_mult)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """设定当前轮次：打乱方式由 `(seed, epoch)` 决定，续训可复现同一顺序"""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        total = len(self.lengths)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        total = len(self.lengths)
        if not self.shuffle:
            order = sorted(range(total), key=self._key)
            return iter(self._chunk(order))

        rng = random.Random(self.seed + self.epoch)
        order = list(range(total))
        rng.shuffle(order)
        batches: list[list[int]] = []
        for start in range(0, total, self.mega_batch_size):
            chunk = sorted(order[start:start + self.mega_batch_size], key=self._key)
            batches.extend(self._chunk(chunk))
        rng.shuffle(batches)
        return iter(batches)

    def _key(self, index: int) -> int:
        return self.lengths[index]

    def _chunk(self, order: Sequence[int]) -> list[list[int]]:
        batches = [
            list(order[start:start + self.batch_size])
            for start in range(0, len(order), self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        return batches


def set_loader_epoch(loader: DataLoader, epoch: int) -> None:
    """通知 loader 进入下一轮（仅长度分桶采样器需要）"""
    sampler = loader.batch_sampler
    if isinstance(sampler, LengthGroupedSampler):
        sampler.set_epoch(epoch)
