"""实例侧：开跑前的耗时与显存预检

用真实配置、真实模型与真实数据各测一遍训练单步，并按数据集的真实长度分布
估算五个阶段的总耗时，避免「先开跑再发现批大小放不下 / 时间超预算」。

两种单步分别对应真实阶段：

- ``backbone 参与训练`` —— `stage1`（DoRA + 分类头）与 `pretrain`（MLM），
  backbone 每一层都要存激活，显存占用最高；
- ``backbone 冻结`` —— `stage2` / `stage3` / `stage4`，backbone 不建计算图，
  只有专家与路由器反传，单步更便宜。

两种精度都会实测（fp32 / bf16），并按 `train.amp` 与 `train.length_bucketing`
的当前配置给出预估，同时打印关闭优化后的对比，便于核对收益。

用法:

    python -m scripts.autodl.preflight
    python scripts/autodl/preflight.py --samples 512 --project-batches 2000
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

if __package__ in (None, ""):  # 支持以脚本方式直接运行
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.config.config import Config, load_config  # noqa: E402
from train.data.dataset import SplitDataset, make_collate_fn  # noqa: E402
from train.data.sampler import LengthGroupedSampler  # noqa: E402
from train.expert.moe import MoEClassifier  # noqa: E402
from train.model.backbone import Backbone  # noqa: E402
from train.model.loss import ClassBalancedLoss  # noqa: E402
from train.utils.env import amp_context, compile_model  # noqa: E402

CONFIG_PATH = "config.toml"

# 阶段 → (消费的 split, 批大小配置项, 每轮遍数)；backbone 是否训练见 BACKBONE_TRAINED
STAGE_PLAN = (
    ("pretrain (MLM)", "mlm", "mlm_pretrain", 1),
    ("stage1 (CLF)", "train", "clf_train", 1),
    ("stage2 (expert)", "train", "expert_train", 10),   # 10 个专家各跑一遍训练集
    ("stage3 (router)", "train", "router_train", 1),
    ("stage4 (joint)", "train", "joint_train", 1),
)
BACKBONE_TRAINED = {"stage1 (CLF)"}
BACKBONE_TRAIN_KEY = "backbone 参与训练"
BACKBONE_FROZEN_KEY = "backbone 冻结"
MLM_KEY = "MLM（含 30k 词表头）"


def step_key(stage_name: str) -> str:
    """训练阶段 → 实测的步型"""
    if stage_name.startswith("pretrain"):
        return MLM_KEY
    return BACKBONE_TRAIN_KEY if stage_name in BACKBONE_TRAINED else BACKBONE_FROZEN_KEY


_LENGTH_CACHE: dict[str, np.ndarray] = {}
_PAD_CACHE: dict[tuple[str, int, bool], float] = {}


def split_lengths(cfg: Config, use: str) -> np.ndarray:
    """按 MacBERT 字符级词表近似 token 长度（+2 为 [CLS] / [SEP]）"""
    if use not in _LENGTH_CACHE:
        ds = SplitDataset(cfg.data.dataset_path, use, columns=["id", "text"])
        _LENGTH_CACHE[use] = np.array(ds.lengths, dtype=np.int64)
    return _LENGTH_CACHE[use]


def random_batch_len(lens: np.ndarray, batch_size: int, max_length: int, trials: int, rng) -> float:
    """随机组批时，动态 padding 对齐批内最长样本的期望批长"""
    idx = rng.integers(0, len(lens), size=(trials, batch_size))
    return float(np.clip(lens[idx].max(axis=1), None, max_length).mean())


def bucketed_batch_len(lengths, batch_size: int, max_length: int, trials: int) -> float:
    """长度分桶组批时的期望批长

    训练侧采样器逐轮打乱批次顺序，取前 `trials` 批即无偏样本
    """
    sampler = LengthGroupedSampler(lengths, batch_size, seed=0)
    batches = list(islice(sampler, trials))
    return float(np.mean([min(max(lengths[i] for i in batch), max_length) for batch in batches]))


def padded_len(cfg: Config, use: str, batch_size: int, bucketing: bool, trials: int, rng) -> float:
    """带缓存的批长估计：预检会多次投影不同配置，避免重复重排大数组"""
    key = (use, batch_size, bucketing)
    if key not in _PAD_CACHE:
        lens = split_lengths(cfg, use)
        _PAD_CACHE[key] = (
            bucketed_batch_len(lens, batch_size, cfg.data.max_length, trials)
            if bucketing
            else random_batch_len(lens, batch_size, cfg.data.max_length, trials, rng)
        )
    return _PAD_CACHE[key]


def bench(fn, batch, iters: int, warmup: int = 1) -> tuple[float, int]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        fn(batch)
    torch.cuda.synchronize()
    started = time.time()
    for _ in range(iters):
        fn(batch)
    torch.cuda.synchronize()
    elapsed = (time.time() - started) / iters
    padded = batch["input_ids"].shape[0] * batch["input_ids"].shape[1]
    return elapsed, padded


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1024, help="取多少条真实样本做基准")
    parser.add_argument("--project-batches", type=int, default=2000, help="估算批长时的采样批数")
    parser.add_argument("--iterations", type=int, default=3, help="每种配置的计时次数")
    parser.add_argument("--max-batch", type=int, default=0,
                        help="测速用的批大小上限；在显存小于目标机的卡上做预检时用它避免换页，0 表示不限")
    args, rest = parser.parse_known_args(argv)
    max_batch = args.max_batch if args.max_batch > 0 else None

    # 其余参数按项目统一语法覆盖配置，例如 --no-train-compile / --train-amp
    cfg = load_config(Config, toml_path=CONFIG_PATH, argv=rest)
    if not torch.cuda.is_available():
        raise SystemExit("没有可用 GPU，预检无意义")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"显存 {props.total_memory / 2**30:.1f} GiB  SM {props.multi_processor_count}  "
          f"torch {torch.__version__}", flush=True)
    print(f"配置: amp={cfg.train.amp} length_bucketing={cfg.train.length_bucketing}  "
          f"bf16 支持={torch.cuda.is_bf16_supported()}", flush=True)

    clf_backbone = Backbone(cfg.model, cfg.peft, is_mlm=False)
    clf_backbone.build()
    model = MoEClassifier(clf_backbone, cfg.num_classes, cfg.experts, cfg.router)
    model.to(device)
    clf_forward = compile_model(model.clf_forward, cfg.train.compile)
    moe_forward = compile_model(model, cfg.train.compile)
    collate = make_collate_fn(clf_backbone.tokenizer, cfg.data.max_length)

    dataset = SplitDataset(cfg.data.dataset_path, "train")
    rng = np.random.default_rng(cfg.seed)
    picked = rng.choice(len(dataset), size=min(args.samples, len(dataset)), replace=False)
    samples = [dataset[int(i)] for i in picked]

    samples_per_cls = torch.bincount(dataset.labels, minlength=cfg.num_classes).float().clamp(min=1)
    cb_loss = ClassBalancedLoss(
        samples_per_cls=samples_per_cls,
        beta=cfg.loss.defaults.beta,
        num_classes=cfg.num_classes,
        base_loss=nn.CrossEntropyLoss(reduction="none"),
    ).to(device)

    def make_optimizer() -> torch.optim.Optimizer:
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)

    optimizer = make_optimizer()

    def step_clf(batch, amp: bool):
        with amp_context(device, amp):
            logits = clf_forward(batch["input_ids"], batch["attention_mask"])
            loss = cb_loss(logits, batch["labels"])
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    def step_frozen(batch, amp: bool):
        with amp_context(device, amp):
            out = moe_forward(batch["input_ids"], batch["attention_mask"])
            loss = cb_loss(out["logits"], batch["labels"])
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    measurements: dict[tuple[str, bool], dict] = {}

    def probe_batch(batch_size: int, length: int) -> dict:
        """计时只关心形状：用合成 token 精确控制批长，避免真实文本长度不可控"""
        ids = torch.randint(100, 2000, (batch_size, length), device=device)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": torch.randint(0, cfg.num_classes, (batch_size,), device=device),
        }

    def measure(name: str, step, amp: bool, wanted: int, length: int) -> dict:
        fitted = wanted
        while True:
            batch = probe_batch(fitted, length)
            try:
                elapsed, padded = bench(lambda b: step(b, amp), batch, args.iterations)
                break
            except torch.cuda.OutOfMemoryError:
                del batch
                torch.cuda.empty_cache()
                if fitted == 1:
                    raise SystemExit(f"显存不足：{name}（amp={amp}）连 batch=1 都无法容纳")
                print(f"   !! batch={fitted} 显存不足，降为 batch={fitted // 2} 测速", flush=True)
                fitted = max(1, fitted // 2)

        peak = torch.cuda.max_memory_allocated() / 2**30
        flag = "" if fitted == wanted else f"  ← 目标 batch={wanted}"
        print(f"[{name} | {'bf16' if amp else 'fp32'}] batch={fitted} 批长={length}"
              f"（{padded:,} token） {elapsed * 1000:6.1f} ms/步  峰值显存 {peak:5.2f} GiB{flag}", flush=True)
        del batch
        torch.cuda.empty_cache()
        return {"ms": elapsed * 1000, "tokens": padded, "batch_size": fitted, "peak_gib": peak}

    def start_batch(configured: int) -> int:
        """`--max-batch` 用于在本机显存小于目标机时避免换页拖慢测速"""
        return configured if max_batch is None else min(configured, max_batch)

    def fit(name: str, step, amp: bool, wanted: int) -> None:
        """三点最小二乘拟合：耗时 = F + S × 批内 token 数

        eager 下 F（每步数百个算子的派发开销）与 token 数无关，只按「每 token 成本」
        外推会严重低估小批次阶段的耗时，因此必须把两项分开。两点法在测速噪声稍大时
        会拟合出离谱斜率（甚至负值），所以用三个批长做最小二乘，并把斜率钳到非负
        """
        points = [measure(name, step, amp, wanted, length) for length in PROBE_LENGTHS]
        tokens = np.array([p["tokens"] for p in points], dtype=float)
        times = np.array([p["ms"] for p in points], dtype=float)
        slope, intercept = np.polyfit(tokens, times, 1)
        slope = max(float(slope), 0.0)
        if not np.all(np.diff(times) > 0):
            print("   !! 测速点耗时非单调递增（噪声偏大），固定开销按较大值取", flush=True)
        fixed = max(float(intercept), 0.0)
        measurements[(name, amp)] = {
            "fixed_ms": fixed,
            "ms_per_1k": slope * 1000,
            "batch_size": points[-1]["batch_size"],
            "target": wanted,
            "peak_gib": max(p["peak_gib"] for p in points),
        }
        print(f"   → 固定开销 {fixed:.1f} ms/步，边际 {slope * 1000:.3f} ms/千token", flush=True)

    # 三点法的探针批长：小批长暴露固定开销，大批长暴露边际成本
    PROBE_LENGTHS = (64, 256, 512)
    amp_on = cfg.train.amp and torch.cuda.is_bf16_supported()
    # 开启编译时只测配置值：编译成本高，再测一份 fp32 会让预检多花十几分钟
    precisions = [amp_on] if cfg.train.compile else ([False, True] if torch.cuda.is_bf16_supported() else [False])

    for amp in precisions:
        fit(BACKBONE_TRAIN_KEY, step_clf, amp, start_batch(cfg.train.clf_train.batch_size))

    # 与 stage2/3/4 一致：冻结 backbone，只训练专家与路由器
    for param in model.backbone.model.parameters():
        param.requires_grad_(False)
    optimizer = make_optimizer()
    for amp in precisions:
        fit(BACKBONE_FROZEN_KEY, step_frozen, amp, start_batch(cfg.train.joint_train.batch_size))

    # MLM 阶段的头是 30k 词表、且对全部 token 计算，成本结构与分类路径完全不同，单独测
    mlm_backbone = Backbone(cfg.model, cfg.peft, is_mlm=True)
    mlm_backbone.build()
    mlm_backbone.model.to(device).train()
    mlm_forward = compile_model(mlm_backbone.model, cfg.train.compile)
    mlm_opt = torch.optim.AdamW(
        [p for p in mlm_backbone.model.parameters() if p.requires_grad], lr=cfg.train.mlm_pretrain.lr
    )

    def step_mlm(batch, amp: bool):
        with amp_context(device, amp):
            out = mlm_forward(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"].clone(),
            )
        mlm_opt.zero_grad(); out.loss.backward(); mlm_opt.step()

    for amp in precisions:
        fit(MLM_KEY, step_mlm, amp, start_batch(cfg.train.mlm_pretrain.batch_size))
    del mlm_backbone, mlm_forward, mlm_opt
    torch.cuda.empty_cache()

    if len(precisions) > 1:
        for key in (BACKBONE_TRAIN_KEY, BACKBONE_FROZEN_KEY, MLM_KEY):
            fp32, bf16 = measurements[(key, False)]["ms_per_1k"], measurements[(key, True)]["ms_per_1k"]
            print(f"bf16 相对 fp32（边际）：{key} {fp32 / bf16:.2f}x", flush=True)

    def step_ms(name: str, pad: float, batch_size: int, amp: bool) -> float:
        """按拟合结果算单步耗时：固定开销 + 边际成本"""
        m = measurements[(step_key(name), amp)]
        return m["fixed_ms"] + m["ms_per_1k"] * (pad * batch_size) / 1000

    print("\n== 各阶段耗时预估")
    print(f"{'阶段':<18}{'真实均值':>9}{'随机组批':>9}{'分桶组批':>9}{'ms/步':>9}{'批数':>10}{'预计小时':>10}")

    def project(bucketing: bool, amp: bool) -> tuple[float, list[float]]:
        total = 0.0
        rows: list[float] = []
        for name, use, stage_key, passes in STAGE_PLAN:
            stage_cfg = getattr(cfg.train, stage_key)
            lens = split_lengths(cfg, use)
            pad = padded_len(cfg, use, stage_cfg.batch_size, bucketing, args.project_batches, rng)
            batches = len(lens) * passes * stage_cfg.epochs / stage_cfg.batch_size
            hours = batches * step_ms(name, pad, stage_cfg.batch_size, amp) / 1000 / 3600
            total += hours
            rows.append(hours)
        return total, rows

    total, rows = project(cfg.train.length_bucketing, amp_on)
    for (name, use, stage_key, passes), hours in zip(STAGE_PLAN, rows):
        stage_cfg = getattr(cfg.train, stage_key)
        lens = split_lengths(cfg, use)
        pad = padded_len(cfg, use, stage_cfg.batch_size, cfg.train.length_bucketing, args.project_batches, rng)
        print(f"{name:<18}{lens.mean():>9.0f}"
              f"{padded_len(cfg, use, stage_cfg.batch_size, False, args.project_batches, rng):>9.0f}"
              f"{pad:>9.0f}{step_ms(name, pad, stage_cfg.batch_size, amp_on):>9.1f}"
              f"{len(lens) / stage_cfg.batch_size * passes * stage_cfg.epochs:>10,.0f}{hours:>10.1f}")
    print(f"{'合计':<18}{'':>9}{'':>9}{'':>9}{'':>9}{'':>10}{total:>10.1f}")

    if cfg.train.length_bucketing or amp_on:
        print("\n作为对比：")
        variants = [
            (False, amp_on, "关闭长度分桶"),
            (cfg.train.length_bucketing, False, "关闭 AMP"),
            (False, False, "两者都关（原始配置）"),
        ]
        for bucketing, amp, label in variants:
            if bucketing == cfg.train.length_bucketing and amp == amp_on:
                continue
            other, _ = project(bucketing, amp)
            print(f"  {label}：{other:.1f} 小时（当前 {total:.1f} 小时，慢 {other / total:.1f}x）")

    undersized = [name for (name, amp), m in measurements.items()
                  if amp == amp_on and m["batch_size"] < m["target"]]
    if undersized:
        print(f"\n!! {'、'.join(sorted(set(undersized)))}：这批配置在当前精度下显存放不下，"
              f"上表仍按配置值推算，实际训练前需调小对应阶段的 batch_size")
    peak_train = measurements[(BACKBONE_TRAIN_KEY, amp_on)]
    print(f"\n注：未计入每轮验证集评估（约占总时长的 5%–10%，stage2 按每个专家每轮都验证的现状推算）。")
    print(f"实测峰值显存（{'bf16' if amp_on else 'fp32'}）："
          f"{BACKBONE_TRAIN_KEY} batch={peak_train['batch_size']} → {peak_train['peak_gib']:.2f} GiB，"
          f"{BACKBONE_FROZEN_KEY} batch={measurements[(BACKBONE_FROZEN_KEY, amp_on)]['batch_size']} → "
          f"{measurements[(BACKBONE_FROZEN_KEY, amp_on)]['peak_gib']:.2f} GiB")

    # 数据供给能力：分词与增强都在 CPU 上做，样本吞吐要跟得上 GPU 步速
    probe = samples[:64]
    collate(probe)
    started = time.time()
    for _ in range(5):
        collate(probe)
    cpu_rate = 64 * 5 / (time.time() - started)

    print(f"\n== 数据供给（collate 在 CPU 上做分词与增强）")
    print(f"单进程 {cpu_rate:,.0f} samples/s（不含量化增强；带增强的专家约 1.4k samples/s）"
          f"   当前配置 num_workers={cfg.train.num_workers}")
    worst = None
    for name, use, stage_key, passes in STAGE_PLAN:
        stage_cfg = getattr(cfg.train, stage_key)
        lens = split_lengths(cfg, use)
        batch_size = stage_cfg.batch_size
        pad = padded_len(cfg, use, batch_size, cfg.train.length_bucketing, args.project_batches, rng)
        per_step = step_ms(name, pad, batch_size, amp_on)
        need = batch_size * 1000 / per_step
        if worst is None or need > worst[1]:
            worst = (name, need)
        flag = "  ← 单进程供不上" if need > cpu_rate else ""
        print(f"{name:<18}每步 {pad * batch_size:>6.0f} token / {per_step:>7.1f} ms 需 {need:>8,.0f} samples/s{flag}")
    workers_needed = int(worst[1] / cpu_rate) + 1
    print(f"最吃 CPU 的阶段：{worst[0]}，需 {worst[1]:,.0f} samples/s，单进程可提供 {cpu_rate:,.0f}，"
          f"建议 num_workers ≥ {workers_needed}")
    print("换机器换算：固定开销与边际成本都要在目标机上重测（固定开销与卡无关、与算子数有关，"
          "不能按算力线性折算）。")


if __name__ == "__main__":
    main()
