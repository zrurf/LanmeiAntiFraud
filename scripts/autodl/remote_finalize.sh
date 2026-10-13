#!/usr/bin/env bash
# 实例侧：训练结束后的收尾——导出部署产物 + 测试集评估 + 基线/消融
#
# 用法（在实例上、项目根目录下）：
#   bash scripts/autodl/remote_finalize.sh              # 导出 + 评估（默认跳过 TF-IDF）
#   LMAF_WITH_TFIDF=1 bash scripts/autodl/remote_finalize.sh   # 额外跑传统方法基线
#
# 任一步失败都不会中断后续步骤（时间紧时也要把能拿的结果拿到），最后打印产物清单。

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH=/root/miniconda3/bin:$PATH
mkdir -p artifacts/results

step() { echo; echo "===== $* ====="; }

step "导出部署产物（自动按 joint → router → expert → clf 选权重）"
if ls artifacts/export/*/model.pt >/dev/null 2>&1; then
    python -m scripts.export_model 2>&1 | tail -4
else
    echo "!! 没有任何 export/*/model.pt，跳过导出"
fi

run_eval() {
    local label="$1"; shift
    step "测试集评估：${label}"
    if python -m tests.baseline.eval_testset "$@" 2>&1 | tail -3; then
        :
    else
        echo "!! ${label} 失败"
    fi
}

run_eval "MoE 门控融合（主结果）" --mode moe --fusion gated --out artifacts/results/moe_gated.json
run_eval "单头基线 MacBERT+DoRA" --mode single --out artifacts/results/macbert_dora_single.json
run_eval "消融 A1：等权融合" --mode moe --fusion equal --out artifacts/results/moe_equal.json
run_eval "消融 A2：top-1 硬选择" --mode moe --fusion top1 --out artifacts/results/moe_top1.json

step "MoE 对比单头（McNemar + 配对 bootstrap）"
if [ -f artifacts/export/clf/model.pt ] && [ -f artifacts/export/joint/model.pt ]; then
    python -m tests.baseline.eval_testset \
        --mode moe --fusion gated \
        --compare artifacts/export/clf/model.pt --compare-mode single \
        --out artifacts/results/moe_vs_single.json 2>&1 | tail -3
else
    echo "!! 缺少 clf 或 joint 权重，跳过对比"
fi

if [ "${LMAF_WITH_TFIDF:-0}" = "1" ]; then
    step "传统方法基线（TF-IDF + NB / LR，较吃 CPU 与内存）"
    python -m tests.baseline.tfidf 2>&1 | tail -4
fi

step "产物清单"
ls -la artifacts/export/*/model.pt artifacts/export/deploy/ 2>/dev/null | head -20
ls -la artifacts/results/*.json 2>/dev/null
echo
echo "===== 收尾完成 ====="
