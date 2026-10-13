# AutoDL 部署

在一台 AutoDL 实例上完成「上传 → 装环境 → 预检 → 训练 → 取回产物」的脚本集合。

## 0 准备连接

实例的 SSH 登录信息取自 AutoDL 控制台的「登录指令」，形如
`ssh -p 12345 root@connect.westb.seetacloud.com`。

如没有 `sshpass` 与 `rsync`，则使用 `ssh` / `scp` 与密钥登录，因此需要先把本机公钥加进实例：

```bash
cat ~/.ssh/autodl_lanmei.pub          # 复制输出的整行
```

粘贴位置二选一：

- 创建实例时在「SSH 公钥」处填入；
- 或实例创建后在「设置 → SSH 公钥」处追加。

所有脚本通过以下环境变量寻址，未设置即报错退出：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LMAF_HOST` | — | 登录地址，如 `connect.westb.seetacloud.com` |
| `LMAF_PORT` | — | SSH 端口 |
| `LMAF_USER` | `root` | 登录用户 |
| `LMAF_KEY` | `~/.ssh/autodl_lanmei` | 私钥路径 |
| `LMAF_DEST` | `/root/LanmeiAntiFraud` | 实例上的项目目录 |

## 1 上传

```bash
LMAF_HOST=connect.westb.seetacloud.com LMAF_PORT=12345 bash scripts/autodl/upload.sh
```

上传内容与策略：

| 内容 | 大小 | 方式 |
| --- | --- | --- |
| 源码 | 约 0.3 MB | 打包后单次 `scp` |
| `datasets/dataset.lance` | 约 558 MB | 切成 64 MB 分片，按远端文件大小跳过已完成分片，可断点续传 |
| `models/macbert` | 约 410 MB | 默认在实例上用 modelscope 下载（`LMAF_MODEL_MODE=download`），也可改为上传（`=upload`） |

`datasets/`、`models/`、`artifacts/` 在 `.gitignore` 中，不走 git，只能按上面的方式搬运。

## 2 装环境

```bash
bash scripts/autodl/remote_setup.sh
```

依次执行：打印 Python / CUDA / 驱动 / 现有 torch 信息 → 备份 `pip freeze` →
安装 `requirements.txt`（`torch` 那行装不上时自动剔除，沿用镜像自带 torch）→
跑 `pytest` 自检。

## 3 预检（可选但推荐）

```bash
python -m scripts.autodl.preflight
```

用真实数据与真实模型测一遍单步耗时与显存峰值，fp32 与 bf16 各测一次，
再按 `train.amp` / `train.length_bucketing` 的当前配置给出各阶段耗时预估，
并打印「关闭优化后」的对比，便于核对收益。开跑前就能确认批大小是否放得下、总共要多久。

## 4 训练

```bash
bash scripts/autodl/remote_train.sh              # 全部 5 个阶段，全新训练
bash scripts/autodl/remote_train.sh --resume     # 断点续训
```

以 `nohup` 方式后台运行，日志写入 `artifacts/logs/train_<时间戳>.log`，PID 写入
`artifacts/logs/train.pid`。SSH 断开不影响训练。

```bash
tail -f artifacts/logs/train_<时间戳>.log      # 查看进度
kill "$(cat artifacts/logs/train.pid)"         # 停止训练
```

分词与数据增强在 CPU 上做，`train.num_workers`（`config.toml` 中为 8）应与实例核数匹配：
单进程约 2.3k 样本/秒，GPU 越快越需要多开，并留出核给主进程。实例为 22 核，8 个 worker 有 5 倍以上余量。

## 5 取回产物

```bash
LMAF_HOST=... LMAF_PORT=... bash scripts/autodl/pull_artifacts.sh
```

默认拉回评估与部署所需的部分：`export/{clf,router,joint}`、`export/deploy`、
各阶段 `metrics.json` 与 `artifacts/results/`。中间阶段权重与断点默认不拉：

- `LMAF_PULL_ALL=1` 追加 `export/mlm`、`export/expert`；
- `LMAF_PULL_CKPT=1` 追加各阶段 `last.pt` 断点（仅续训需要）。
