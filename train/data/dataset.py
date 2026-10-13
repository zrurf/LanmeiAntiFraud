from __future__ import annotations

import logging
import os
import random

import lance
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Any, Callable, Iterable

from train.data.sampler import LengthGroupedSampler

logger = logging.getLogger(__name__)


COLUMNS = ["id", "text", "label10", "subtype", "source"]


class SplitDataset(Dataset):

    def __init__(self, uri: str, use: str, columns=COLUMNS):
        ds = lance.dataset(uri)
        data = ds.to_table(columns=columns, filter=f"`use` = '{use}'").to_pydict()
        self.texts = data["text"]
        self.labels = (
            torch.tensor(data["label10"], dtype=torch.long)
            if "label10" in data else None
        )
        self._lengths: list[int] | None = None

    @property
    def lengths(self) -> list[int]:
        """样本长度近似值（字符数 + [CLS]/[SEP]），供长度分桶采样器分组

        按字符计对中文语料是足够好的代理；不必走 tokenizer，避免逐轮重复分词
        """
        if self._lengths is None:
            self._lengths = [len(text) + 2 for text in self.texts]
        return self._lengths

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, int]:
        label = self.labels[index].item() if self.labels is not None else -1
        return self.texts[index], label

def make_collate_fn(
    tokenizer,
    max_length: int,
    augmentation: Callable[[str], str] | None = None,
):
    def collate(batch: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        texts = [text for text, _ in batch]
        labels = torch.tensor([label for _, label in batch], dtype=torch.long)

        if augmentation is not None:
            texts = [augmentation(t) for t in texts]

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

    return collate


def _init_worker(worker_id: int) -> None:
    """worker 初始化

    分词与数据增强都在 collate 里做、且是纯 CPU 的 Python 代码，因此把每个 worker
    限制成单线程，让 N 个 worker 各自占一个核，避免与主进程抢线程。
    快速分词器内部的 rayon 线程池同样收敛到 1（该变量在池已初始化时会被忽略，
    此时 fork 出来的子进程里没有可用线程，rayon 会自动退化为当前线程执行）。
    同时按 worker 的 torch 种子重播 `random`：数据增强依赖全局 `random`，
    否则各 worker 会继承同一个随机流、增强样本高度重复
    """
    os.environ["RAYON_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    random.seed(torch.initial_seed() % 2**32)


def _effective_workers(requested: int) -> int:
    """子进程数是否可用

    `collate` 里的 tokenizer 与增强是闭包，只有 fork 能把它们带进子进程；
    Windows / macOS 的 spawn 需要 pickle，因此这些平台上退回单进程
    """
    if requested <= 0:
        return 0
    if torch.multiprocessing.get_start_method(allow_none=False) != "fork":
        logger.warning(
            "train.num_workers=%d 已忽略：当前多进程 start method 不是 fork，"
            "collate 中的分词与增强闭包无法跨进程传递，改用单进程",
            requested,
        )
        return 0
    return requested


def make_loader(
    dataset: SplitDataset,
    collate_fn: Callable,
    batch_size: int,
    *,
    train: bool = False,
    bucketing: bool = True,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    """构造 DataLoader

    `bucketing=True` 时用长度分桶采样器替代 `shuffle` / `batch_size`：
    批内长度接近，动态 padding 补齐的算力浪费显著减少。训练侧逐轮重新分桶，
    评估侧全量按长度排序（指标与顺序无关），`train` 决定是否逐轮打乱

    `num_workers > 0` 时分词与增强在多进程里并行；单进程下分词速度约
    2.3k 样本/秒，在快卡上会成为瓶颈
    """
    kwargs: dict = {"collate_fn": collate_fn}
    # 评估集不启用子进程：它只在每轮末跑一次、不需要吞吐，同时避开 lance 在
    # 多进程 fork 下的已知问题（训练 loader 早早 fork 无事，评估 loader 在
    # CUDA 已热、训练 worker 已存在时才 fork，实测会死锁在 futex 上）
    workers = _effective_workers(num_workers) if train else 0
    if workers > 0:
        kwargs.update(
            num_workers = workers,
            worker_init_fn = _init_worker,
            persistent_workers = True,
            prefetch_factor = 4,
        )
    if not bucketing:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            **kwargs,
        )
    sampler = LengthGroupedSampler(
        dataset.lengths,
        batch_size,
        shuffle=train,
        seed=seed,
    )
    return DataLoader(dataset, batch_sampler=sampler, **kwargs)
