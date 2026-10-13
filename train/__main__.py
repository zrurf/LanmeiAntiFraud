"""训练入口

用法:

    python -m train <stage> [--resume] [配置项覆盖]

`stage` 用于选择训练阶段，可选值:

    pretrain  MLM 预训练
    stage1    分类头训练
    stage2    专家网络训练
    stage3    路由网络训练
    stage4    联合训练
    all       全流程，按 pretrain → stage1 → stage2 → stage3 → stage4 依次执行

`--resume` 启用断点续训，会读取并在训练过程中更新最新权重与断点位置，未指定时（或显式传入 `--no-resume`）执行全新训练
"""

from __future__ import annotations

from train.config.config import Config, load_run
from train.stage.clf import ClfStage
from train.stage.expert import ExpertStage
from train.stage.joint import JointStage
from train.stage.mlm import MlmStage
from train.stage.router import RouterStage

CONFIG_PATH = "config.toml"

STAGES = {
    "pretrain": MlmStage,
    "stage1": ClfStage,
    "stage2": ExpertStage,
    "stage3": RouterStage,
    "stage4": JointStage,
}

PIPELINE = ("pretrain", "stage1", "stage2", "stage3", "stage4")


def main(stage: str, cfg: Config) -> None:
    for name in PIPELINE if stage == "all" else (stage,):
        print(f"[PIPELINE] 开始 {name}", flush=True)
        STAGES[name](cfg).run()


if __name__ == "__main__":
    stage, cfg = load_run(Config, stages=[*STAGES, "all"], toml_path=CONFIG_PATH)
    main(stage, cfg)