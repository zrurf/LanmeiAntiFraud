from pathlib import Path

import pytest

from train.config.config import Config, load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.toml"


def load_project_config(argv: list[str] | None = None) -> Config:
    return load_config(Config, toml_path=CONFIG_PATH, argv=argv or [])


def test_project_config_parses():
    cfg = load_project_config()
    assert cfg.num_classes == 10
    assert len(cfg.experts) == 10
    assert [expert.index for expert in cfg.experts] == list(range(10))
    assert cfg.data.max_length == 512
    assert cfg.peft.method == "dora"


def test_loss_defaults():
    defaults = load_project_config().loss.defaults
    # beta 不断言具体数值：它是实验选定的超参（0.99999 / 0.9999 / 0.999 三点探针后取 0.9999，
    # 见 config.toml 注释），会随实验结果变化，断言具体值只会让改超参时误报失败
    assert 0.0 <= defaults.beta <= 1.0
    assert defaults.self_weight == 3.0
    assert defaults.balance_weight == 0.01
    assert defaults.label_smoothing == 0.0


def test_expert_poolings_and_augments_are_registered():
    from train.data.augment import AUGMENT_FACTORIES
    from train.expert.pooling import AVAILABLE_POOLINGS

    for expert in load_project_config().experts:
        assert expert.pooling in AVAILABLE_POOLINGS, expert.pooling
        for name in expert.augment:
            assert name in AUGMENT_FACTORIES, name


def test_cli_overrides_take_precedence():
    cfg = load_project_config(["--num-classes", "7", "--loss-defaults-beta", "0.9"])
    assert cfg.num_classes == 7
    assert cfg.loss.defaults.beta == pytest.approx(0.9)


def test_env_is_parsed_with_nested_separator(monkeypatch):
    from train.config.config import load_env

    monkeypatch.setenv("LMAF_NUM_CLASSES", "5")
    monkeypatch.setenv("LMAF_LOSS__DEFAULTS__SELF_WEIGHT", "2.5")
    assert load_env("LMAF") == {
        "num_classes": "5",
        "loss": {"defaults": {"self_weight": "2.5"}},
    }


def test_file_overrides_env(monkeypatch):
    # 优先级为 命令行 > 配置文件 > 环境变量
    monkeypatch.setenv("LMAF_NUM_CLASSES", "5")
    assert load_project_config().num_classes == 10