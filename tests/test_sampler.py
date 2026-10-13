"""长度分桶采样器与 AMP 上下文测试"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest
import torch

from train.data.dataset import SplitDataset, make_loader
from train.data.sampler import LengthGroupedSampler, set_loader_epoch
from train.utils.env import amp_context, compile_model


def _lengths(size: int = 20000, seed: int = 0) -> list[int]:
    """长尾长度分布，形态接近真实语料（均值远小于 p99）"""
    rng = np.random.default_rng(seed)
    return [int(x) + 2 for x in rng.lognormal(mean=5.0, sigma=0.9, size=size)]


def _padded_mean(batches, lengths) -> float:
    """动态 padding 下每批实际要算的长度均值"""
    return float(np.mean([max(lengths[i] for i in batch) for batch in batches]))


def _random_batches(lengths, batch_size: int, seed: int = 0):
    order = np.random.default_rng(seed).permutation(len(lengths))
    return [order[i:i + batch_size].tolist() for i in range(0, len(order), batch_size)]


def test_batches_cover_every_index_once():
    lengths = _lengths(1000)
    sampler = LengthGroupedSampler(lengths, 32, seed=1)
    flat = [i for batch in sampler for i in batch]
    assert sorted(flat) == list(range(len(lengths)))


def test_len_matches_produced_batches():
    lengths = _lengths(1000)
    sampler = LengthGroupedSampler(lengths, 32, seed=1)
    assert len(sampler) == len(list(sampler))
    assert len(sampler) == -(-len(lengths) // 32)


def test_bucketing_cuts_padding_waste():
    lengths = _lengths()
    batch_size = 32
    random_pad = _padded_mean(_random_batches(lengths, batch_size), lengths)
    grouped_pad = _padded_mean(LengthGroupedSampler(lengths, batch_size, seed=0), lengths)
    assert grouped_pad < random_pad / 2
    # 分桶后批长应贴近真实均值，而不是 p99 量级
    assert grouped_pad < float(np.mean(lengths)) * 1.35


def test_epoch_changes_batches_but_keeps_coverage():
    lengths = _lengths(500)
    sampler = LengthGroupedSampler(lengths, 16, seed=3)
    first = [tuple(b) for b in sampler]
    sampler.set_epoch(1)
    second = [tuple(b) for b in sampler]
    assert first != second
    assert sorted(i for b in second for i in b) == list(range(len(lengths)))


def test_same_seed_and_epoch_is_reproducible():
    lengths = _lengths(500)
    a = LengthGroupedSampler(lengths, 16, seed=7)
    b = LengthGroupedSampler(lengths, 16, seed=7)
    for sampler in (a, b):
        sampler.set_epoch(2)
    assert [tuple(x) for x in a] == [tuple(x) for x in b]


def test_eval_sampler_is_deterministic_and_sorted():
    lengths = _lengths(500)
    sampler = LengthGroupedSampler(lengths, 32, shuffle=False)
    batches = list(sampler)
    assert [tuple(x) for x in sampler] == [tuple(x) for x in batches]
    maxima = [max(lengths[i] for i in batch) for batch in batches]
    assert maxima == sorted(maxima)


def test_drop_last():
    lengths = _lengths(100)
    sampler = LengthGroupedSampler(lengths, 32, drop_last=True)
    batches = list(sampler)
    assert all(len(batch) == 32 for batch in batches)
    assert len(sampler) == len(batches) == 3


def test_invalid_arguments():
    with pytest.raises(ValueError):
        LengthGroupedSampler([1, 2, 3], 0)
    with pytest.raises(ValueError):
        LengthGroupedSampler([1, 2, 3], 2, mega_batch_mult=0)


def test_set_loader_epoch_tolerates_plain_sampler():
    from torch.utils.data import DataLoader

    loader = DataLoader(list(range(8)), batch_size=2)
    set_loader_epoch(loader, 3)  # 普通采样器无需处理，不应报错
    assert len(list(loader)) == 4


def test_make_loader_switches_on_bucketing(tiny_dataset_path):
    dataset = SplitDataset(str(tiny_dataset_path), "train")
    assert dataset.lengths == [len(text) + 2 for text in dataset.texts]
    assert len(dataset.lengths) == len(dataset)

    bucketed = make_loader(dataset, lambda batch: batch, 4, train=True, bucketing=True)
    assert isinstance(bucketed.batch_sampler, LengthGroupedSampler)
    covered = sorted(i for batch in bucketed.batch_sampler for i in batch)
    assert covered == list(range(len(dataset)))
    assert len(bucketed) == len(list(bucketed))

    plain = make_loader(dataset, lambda batch: batch, 4, train=True, bucketing=False)
    assert len(plain) == -(-len(dataset) // 4)


def test_make_loader_with_workers(tiny_dataset_path):
    """worker 路径要能真正跑通（分词与增强都在 collate 里，跨进程容易踩坑）

    `collate` 是闭包，只有 fork 能带进子进程；Windows / macOS 的 spawn 需 pickle，
    `make_loader` 会退回单进程
    """
    dataset = SplitDataset(str(tiny_dataset_path), "train")
    loader = make_loader(dataset, lambda batch: batch, 4, train=True, bucketing=True, num_workers=2)
    fork = torch.multiprocessing.get_start_method(allow_none=False) == "fork"
    assert loader.num_workers == (2 if fork else 0)
    assert len(list(loader)) == len(loader)


def test_amp_context_disabled_on_cpu():
    assert isinstance(amp_context(torch.device("cpu")), contextlib.nullcontext)
    assert isinstance(amp_context(torch.device("cpu"), enabled=False), contextlib.nullcontext)


def test_compile_model_is_opt_in():
    def forward(x):
        return x + 1

    assert compile_model(forward) is forward
    assert compile_model(forward, enabled=False) is forward
    wrapped = compile_model(forward, enabled=True)
    assert wrapped is not forward  # torch.compile 返回包装器，首次调用才真正编译
    assert callable(wrapped)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_amp_context_follows_bf16_support():
    device = torch.device("cuda")
    context = amp_context(device)
    if torch.cuda.is_bf16_supported():
        assert not isinstance(context, contextlib.nullcontext)
    else:
        assert isinstance(context, contextlib.nullcontext)
    assert isinstance(amp_context(device, enabled=False), contextlib.nullcontext)
