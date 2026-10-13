"""全链路冒烟：用 tiny 模型与合成数据串跑 pretrain → stage4，再导出并推理
"""

from pathlib import Path


def test_full_pipeline_smoke(tiny_config):
    from train.stage.clf import ClfStage
    from train.stage.expert import ExpertStage
    from train.stage.joint import JointStage
    from train.stage.mlm import MlmStage
    from train.stage.router import RouterStage

    MlmStage(tiny_config).run()
    ClfStage(tiny_config).run()
    ExpertStage(tiny_config).run()
    RouterStage(tiny_config).run()
    JointStage(tiny_config).run()

    export_root = Path(tiny_config.output.export_path)
    for stage in ("mlm", "clf", "expert", "router", "joint"):
        assert (export_root / stage / "model.pt").exists(), stage


def test_export_and_predict_after_training(tiny_config):
    from scripts.export_model import export_model
    from service.predictor import Predictor
    from train.stage.clf import ClfStage
    from train.stage.expert import ExpertStage
    from train.stage.joint import JointStage
    from train.stage.router import RouterStage

    ClfStage(tiny_config).run()
    ExpertStage(tiny_config).run()
    RouterStage(tiny_config).run()
    JointStage(tiny_config).run()

    deploy_dir = export_model(tiny_config, out_dir=Path(tiny_config.output.export_path) / "deploy")
    assert (deploy_dir / "model.pt2").exists()

    predictor = Predictor(deploy_dir)
    result = predictor.predict("诈骗转账请点击领取红包退款")
    assert result["label"] in predictor.labels

    explained = predictor.explain("诈骗转账请点击领取红包退款", steps=4, top_k=3)
    assert len(explained["attributions"]) == 3
    assert explained["target_label"] in predictor.labels