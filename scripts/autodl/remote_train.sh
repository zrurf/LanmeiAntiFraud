#!/usr/bin/env bash
# 实例侧：后台启动训练
#
# 用法（在实例上、项目根目录下）：
#   bash scripts/autodl/remote_train.sh                 # 全流程全新训练
#   bash scripts/autodl/remote_train.sh --resume        # 断点续训
#   bash scripts/autodl/remote_train.sh stage2          # 只跑某阶段，可用 --resume
#   bash scripts/autodl/remote_train.sh all --train-clf-train-batch-size 16
#
# 以 nohup 后台运行，SSH 断开不影响；日志见 artifacts/logs/train_<时间戳>.log

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

STAGE="all"
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
    STAGE="$1"
    shift
fi

RESUME_FLAG="--no-resume"
EXTRA=()
for arg in "$@"; do
    if [ "$arg" = "--resume" ]; then
        RESUME_FLAG=""
    else
        EXTRA+=("$arg")
    fi
done

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="artifacts/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/train_$STAMP.log"

if [ -f "$LOG_DIR/train.pid" ] && kill -0 "$(cat "$LOG_DIR/train.pid")" 2>/dev/null; then
    echo "!! 已有训练在运行（PID $(cat "$LOG_DIR/train.pid")），请先停止或换用 --resume"
    exit 1
fi

CMD=(python -m train "$STAGE" $RESUME_FLAG "${EXTRA[@]+"${EXTRA[@]}"}")
echo "启动: ${CMD[*]}"
PYTHONUNBUFFERED=1 nohup "${CMD[@]}" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$LOG_DIR/train.pid"

sleep 3
if ! kill -0 "$PID" 2>/dev/null; then
    echo "!! 进程已退出，日志末尾："
    tail -n 30 "$LOG"
    exit 1
fi

echo "PID $PID，日志 $LOG"
echo "查看进度：tail -f $LOG"
echo "停止训练：kill \$(cat $LOG_DIR/train.pid)"
