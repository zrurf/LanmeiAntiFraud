from __future__ import annotations

import contextlib
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def amp_context(device: torch.device, enabled: bool = True):
    """自动混合精度上下文

    bf16 与 fp32 的指数范围一致，不需要 GradScaler；CUDA 不支持 bf16 时静默退化为
    空上下文，训练精度不受影响
    """
    if enabled and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()

def compile_model(obj, enabled: bool = False, dynamic: bool = True):
    """按需用 `torch.compile` 包装模型或前向方法
    """
    if not enabled:
        return obj
    return torch.compile(obj, dynamic=dynamic)