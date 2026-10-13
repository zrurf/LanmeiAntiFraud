#!/usr/bin/env bash
# 本地侧：把训练产物从实例拉回本地
#
# 用法：
#   LMAF_HOST=connect.westb.seetacloud.com LMAF_PORT=12345 bash scripts/autodl/pull_artifacts.sh
#
# 默认拉回评估与部署所需的产物（约 1.8 GB）：
#   artifacts/export/{clf,router,joint}   单头基线与 MoE 主/备权重
#   artifacts/export/deploy               导出的部署产物
#   artifacts/{mlm,clf,expert,router,joint}/metrics.json、experts_done.json
#   artifacts/results                     基线与消融实验报告
#
# 可选：
#   LMAF_PULL_ALL=1   连同 export/{mlm,expert} 一起拉
#   LMAF_PULL_CKPT=1  连同各阶段 last.pt 断点一起拉

set -euo pipefail

: "${LMAF_HOST:?请设置 LMAF_HOST（AutoDL 登录地址）}"
: "${LMAF_PORT:?请设置 LMAF_PORT（SSH 端口）}"
LMAF_USER="${LMAF_USER:-root}"
LMAF_KEY="${LMAF_KEY:-$HOME/.ssh/autodl_lanmei}"
LMAF_DEST="${LMAF_DEST:-/root/LanmeiAntiFraud}"
LMAF_PULL_CKPT="${LMAF_PULL_CKPT:-0}"
LMAF_PULL_ALL="${LMAF_PULL_ALL:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="$LMAF_USER@$LMAF_HOST"
SSH=(ssh -i "$LMAF_KEY" -p "$LMAF_PORT" -o StrictHostKeyChecking=accept-new "$TARGET")
SCP=(scp -i "$LMAF_KEY" -P "$LMAF_PORT" -o StrictHostKeyChecking=accept-new)

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PATHS=(artifacts/export/clf artifacts/export/router artifacts/export/joint
       artifacts/export/deploy artifacts/results)
for stage in mlm clf expert router joint; do
    PATHS+=("artifacts/$stage/metrics.json" "artifacts/$stage/experts_done.json")
done
[ "$LMAF_PULL_ALL" = "1" ] && PATHS+=(artifacts/export/mlm artifacts/export/expert)
if [ "$LMAF_PULL_CKPT" = "1" ]; then
    for stage in mlm clf expert router joint; do
        PATHS+=("artifacts/$stage/last.pt")
    done
    PATHS+=("artifacts/expert/expert_*.pt")
fi

echo "== 在实例上打包产物"
EXISTING=()
for p in "${PATHS[@]}"; do
    if "${SSH[@]}" "[ -e '$LMAF_DEST/$p' ]" 2>/dev/null; then
        EXISTING+=("$p")
    fi
done
if [ "${#EXISTING[@]}" -eq 0 ]; then
    echo "!! 实例上还没有产物，先确认训练是否在跑：tail -f $LMAF_DEST/artifacts/logs/train_*.log"
    exit 1
fi
echo "   将拉回：${EXISTING[*]}"

"${SSH[@]}" "cd '$LMAF_DEST' && tar czf /tmp/lmaf_artifacts.tgz ${EXISTING[*]} && du -sh /tmp/lmaf_artifacts.tgz"

echo "== 下载"
"${SCP[@]}" "$TARGET:/tmp/lmaf_artifacts.tgz" "$TMP/lmaf_artifacts.tgz"
"${SSH[@]}" "rm -f /tmp/lmaf_artifacts.tgz"

echo "== 解包到 $ROOT"
mkdir -p "$ROOT/artifacts"
tar xzf "$TMP/lmaf_artifacts.tgz" -C "$ROOT"
du -sh "$ROOT/artifacts"
echo "== 完成"
