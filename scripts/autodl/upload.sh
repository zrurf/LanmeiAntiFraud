#!/usr/bin/env bash
# 本地侧：把源码 / 数据集 / 基础模型推送到 AutoDL 实例
#
# 用法：
#   LMAF_HOST=connect.westb.seetacloud.com LMAF_PORT=12345 bash scripts/autodl/upload.sh
#
# 环境变量见 scripts/autodl/README.md；数据集按分片上传，重复执行会跳过已完成分片。

set -euo pipefail

: "${LMAF_HOST:?请设置 LMAF_HOST（AutoDL 登录地址）}"
: "${LMAF_PORT:?请设置 LMAF_PORT（SSH 端口）}"
LMAF_USER="${LMAF_USER:-root}"
LMAF_KEY="${LMAF_KEY:-$HOME/.ssh/autodl_lanmei}"
LMAF_DEST="${LMAF_DEST:-/root/LanmeiAntiFraud}"
LMAF_CHUNK_MB="${LMAF_CHUNK_MB:-64}"
LMAF_MODEL_MODE="${LMAF_MODEL_MODE:-download}"   # download | upload | skip

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="$LMAF_USER@$LMAF_HOST"
SSH=(ssh -i "$LMAF_KEY" -p "$LMAF_PORT" -o StrictHostKeyChecking=accept-new "$TARGET")
SCP=(scp -i "$LMAF_KEY" -P "$LMAF_PORT" -o StrictHostKeyChecking=accept-new)

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== 目标 $TARGET:$LMAF_DEST"
"${SSH[@]}" "mkdir -p '$LMAF_DEST' && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true"

echo "== 上传源码"
tar czf "$TMP/code.tar.gz" -C "$ROOT" \
    --exclude=./datasets --exclude=./models --exclude=./artifacts \
    --exclude=./.git --exclude=./.pytest_cache --exclude='./**/__pycache__' \
    --exclude='*/__pycache__' --exclude='*.pyc' .
"${SCP[@]}" "$TMP/code.tar.gz" "$TARGET:$LMAF_DEST/code.tar.gz"
"${SSH[@]}" "tar xzf '$LMAF_DEST/code.tar.gz' -C '$LMAF_DEST' && rm -f '$LMAF_DEST/code.tar.gz'"
echo "   源码就位"

echo "== 上传数据集（$LMAF_CHUNK_MB MB 分片）"
"${SSH[@]}" "mkdir -p '$LMAF_DEST/datasets'"
tar cf "$TMP/dataset.tar" -C "$ROOT/datasets" dataset.lance
split -b "${LMAF_CHUNK_MB}M" -d -a 3 "$TMP/dataset.tar" "$TMP/dataset.part-"
local_total=$(stat -c %s "$TMP/dataset.tar")
for part in "$TMP"/dataset.part-*; do
    base="$(basename "$part")"
    local_size="$(stat -c %s "$part")"
    remote_size="$("${SSH[@]}" "stat -c %s '$LMAF_DEST/datasets/$base' 2>/dev/null || echo 0")"
    if [ "$remote_size" = "$local_size" ]; then
        echo "   跳过 $base（远端已完整）"
        continue
    fi
    echo "   上传 $base（$((local_size / 1048576)) MB）"
    "${SCP[@]}" "$part" "$TARGET:$LMAF_DEST/datasets/$base"
done
"${SSH[@]}" "cd '$LMAF_DEST/datasets' && cat dataset.part-* > dataset.lance.tar \
    && [ \"\$(stat -c %s dataset.lance.tar)\" = '$local_total' ] \
    && tar xf dataset.lance.tar && rm -f dataset.part-* dataset.lance.tar && du -sh dataset.lance"
echo "   数据集就位"

case "$LMAF_MODEL_MODE" in
download)
    echo "== 基础模型改为在实例上下载（见 remote_setup.sh），跳过上传"
    ;;
upload)
    echo "== 上传基础模型（仅推理需要的文件，跳过 tf / flax 权重）"
    "${SSH[@]}" "mkdir -p '$LMAF_DEST/models/macbert'"
    for f in pytorch_model.bin config.json vocab.txt tokenizer.json tokenizer_config.json \
             special_tokens_map.json added_tokens.json configuration.json; do
        "${SCP[@]}" "$ROOT/models/macbert/$f" "$TARGET:$LMAF_DEST/models/macbert/$f"
    done
    echo "   模型就位"
    ;;
skip)
    echo "== 按 LMAF_MODEL_MODE=skip 跳过基础模型"
    ;;
*)
    echo "未知 LMAF_MODEL_MODE=$LMAF_MODEL_MODE（可选 download / upload / skip）" >&2
    exit 2
    ;;
esac

echo "== 完成：$TARGET:$LMAF_DEST"
