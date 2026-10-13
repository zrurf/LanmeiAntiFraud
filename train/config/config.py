from __future__ import annotations

import argparse
import os
import tomllib
import types
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Sequence, Union, get_args, get_origin, get_type_hints

from train.config.data import DataConfig, OutputConfig
from train.config.expert import ExpertConfig
from train.config.model import ModelConfig
from train.config.peft import PeftConfig
from train.config.router import RouterConfig
from train.config.train import LossConfig, TrainConfig

@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    output: OutputConfig
    peft: PeftConfig
    train: TrainConfig
    router: RouterConfig
    loss: LossConfig
    experts: list[ExpertConfig] = field(default_factory=list)
    seed: int = 42
    resume: bool = False
    num_classes: int = 10

def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def _unwrap_optional(tp):
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp

def _coerce(val, tp):
    if tp is Any or val is None or not isinstance(tp, type):
        return val
    if isinstance(val, tp) and not (tp is int and isinstance(val, bool)):
        return val
    if tp is bool and isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "on", "y"}
    try:
        return tp(val)
    except (ValueError, TypeError):
        return val

def load_toml(path: str | Path | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("rb") as f:
        return tomllib.load(f)

def load_env(prefix: str, sep: str = "__") -> dict:
    out: dict = {}
    prefix = (prefix.rstrip("_") + "_") if prefix else ""
    sep = sep.lower()
    for key, value in os.environ.items():
        if prefix and not key.upper().startswith(prefix.upper()):
            continue
        rest = key[len(prefix):].lower()
        if not rest:
            continue
        parts = rest.split(sep)
        d = out
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value
    return out

def _add_cli_args(parser: argparse.ArgumentParser, cls: type, prefix: str = "") -> None:
    hints = get_type_hints(cls)
    for f in fields(cls):
        if not f.init:
            continue
        tp = _unwrap_optional(hints.get(f.name, Any))
        path = f"{prefix}{f.name}"

        if is_dataclass(tp):
            _add_cli_args(parser, tp, prefix=f"{path}.")
            continue

        flag = "--" + path.replace("_", "-").replace(".", "-")
        kwargs: dict = {"dest": path, "default": argparse.SUPPRESS}
        if tp is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        elif isinstance(tp, type):
            kwargs["type"] = tp
        else:
            # 联合类型 / 容器等无法作为单值转换函数，退化为字符串接收
            kwargs["type"] = str
        parser.add_argument(flag, **kwargs)

def _ns_to_nested(ns: argparse.Namespace) -> dict:
    out: dict = {}
    for key, value in vars(ns).items():
        parts = key.split(".")
        d = out
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value
    return out

def _build(cls: type, data: dict):
    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if not f.init or f.name not in data:
            continue
        tp = _unwrap_optional(hints.get(f.name, Any))
        val = data[f.name]
        if is_dataclass(tp) and isinstance(val, dict):
            kwargs[f.name] = _build(tp, val)
            continue

        # 支持 list[dataclass]，如 [[experts]] 形式的配置
        args = get_args(tp) if get_origin(tp) is list else ()
        if args and is_dataclass(args[0]) and isinstance(val, list):
            kwargs[f.name] = [
                _build(args[0], v) if isinstance(v, dict) else v for v in val
            ]
            continue

        kwargs[f.name] = _coerce(val, tp)
    return cls(**kwargs)

def _make_parser(
    cls: type,
    positionals: Sequence[tuple[str, Sequence[str]]] = (),
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"{cls.__name__} options")
    for name, choices in positionals:
        parser.add_argument(
            name, choices=list(choices), help=f"{name}: {', '.join(choices)}"
        )
    _add_cli_args(parser, cls)
    return parser

def _resolve(
    cls: type,
    ns: argparse.Namespace,
    toml_path: str | Path | None,
    env_prefix: str,
    env_sep: str,
):
    # 优先级：命令行 > 配置文件 > 环境变量
    cli_data = _ns_to_nested(ns)
    file_data = load_toml(toml_path)
    env_data = load_env(env_prefix, env_sep)

    merged = _deep_merge(env_data, file_data)
    merged = _deep_merge(merged, cli_data)
    return _build(cls, merged)

def load_config(
    cls: type,
    *,
    toml_path: str | Path | None = None,
    env_prefix: str = "LMAF",
    env_sep: str = "__",
    argv: list[str] | None = None,
):
    parser = _make_parser(cls)
    ns = parser.parse_args(argv)
    return _resolve(cls, ns, toml_path, env_prefix, env_sep)

def load_run(
    cls: type,
    *,
    stages: Sequence[str],
    toml_path: str | Path | None = None,
    env_prefix: str = "LMAF",
    env_sep: str = "__",
    argv: list[str] | None = None,
) -> tuple[str, Any]:
    """解析训练入口参数，返回 `(stage, cfg)`

    `stage` 取自命令行首个位置参数，必须属于 `stages`；其余参数
    与配置文件、环境变量按优先级合并为 `cls` 实例
    """
    parser = _make_parser(cls, [("stage", stages)])
    ns = parser.parse_args(argv)
    return ns.stage, _resolve(cls, ns, toml_path, env_prefix, env_sep)