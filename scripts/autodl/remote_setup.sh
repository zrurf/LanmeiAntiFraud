#!/usr/bin/env bash
# 实例侧：装环境 + 自检
#
# 用法（在实例上、项目根目录下）：
#   bash scripts/autodl/remote_setup.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "===== 1 环境信息 ====="
python -c "import sys; print('python', sys.version)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
nvidia-smi | grep -i "CUDA Version" || true
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" \
    || { echo "!! 需要 Python >= 3.11（train/config/config.py 使用标准库 tomllib）"; exit 1; }

echo "===== 2 备份当前依赖 ====="
pip freeze > /root/pip_freeze_before.txt 2>/dev/null || true
echo "   已写入 /root/pip_freeze_before.txt"

echo "===== 3 安装依赖 ====="
if python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    SPEC="$(python -c "import torch; print(f'torch=={torch.__version__}')")"
    echo "   自带 torch 可用：$SPEC（CUDA $(python -c 'import torch; print(torch.version.cuda)')）"
    echo "   锁住该版本，只安装其余依赖"
    echo "$SPEC" > /tmp/lmaf_constraints.txt
    grep -v -iE '^torch(==|>=|<=|~=|>|<|$)' requirements.txt > /tmp/req_no_torch.txt
    pip install -c /tmp/lmaf_constraints.txt -r /tmp/req_no_torch.txt
else
    echo "   自带 torch 不可用，按 requirements.txt 安装（含 torch）"
    pip install -r requirements.txt
fi
python -c "import torch; print('   torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available(), '| arch', 'sm_120' in torch.cuda.get_arch_list())"

echo "===== 4 基础模型 ====="
if [ -f models/macbert/pytorch_model.bin ] && [ -f models/macbert/vocab.txt ]; then
    echo "   已存在 models/macbert"
else
    echo "   用 modelscope 下载 hfl/chinese-macbert-base 到 models/macbert（只取 PyTorch 权重与分词器）"
    pip install -q modelscope
    modelscope download --model hfl/chinese-macbert-base --local_dir ./models/macbert \
        --include 'pytorch_model.bin' 'config.json' 'configuration.json' 'vocab.txt' \
                  'tokenizer.json' 'tokenizer_config.json' 'special_tokens_map.json' 'added_tokens.json'
fi
du -sh models/macbert datasets/dataset.lance

echo "===== 5 自检（pytest，约 1 分钟）====="
python -m pytest -q

echo "===== 完成 ====="
