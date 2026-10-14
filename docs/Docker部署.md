# Docker 部署

把在线服务（`service/` + `train/` 推理路径）与模型资产一起打进镜像，跑起来的容器开箱即用。
默认 **CPU 版**（任何机器可跑），`--build-arg TORCH_INDEX=cu130` 出 GPU 版。

---

## 1 前置条件

| 项 | 要求 |
| --- | --- |
| 架构 | x86_64（所用 wheel 均为 `manylinux_2_28_x86_64`；arm64 需自行核对 wheel） |
| Docker | 24 以上，含 `docker compose` 插件（可选） |
| CPU 版 | 无额外要求 |
| GPU 版 | NVIDIA 驱动 ≥ 580（CUDA 13）、`nvidia-container-toolkit`、`--gpus all` |

镜像自带全部 CUDA 运行时库（`cu130` wheel 内嵌 `nvidia-*` 依赖），宿主只需驱动，不需要装 CUDA Toolkit。

## 2 构建与运行

```bash
# CPU 版
docker build -t lanmei-antifraud:latest .
docker run -d --name lanmei-antifraud -p 8000:8000 lanmei-antifraud:latest

# GPU 版
docker build --build-arg TORCH_INDEX=cu130 -t lanmei-antifraud:cu130 .
docker run -d --name lanmei-antifraud --gpus all -p 8000:8000 lanmei-antifraud:cu130

# 或交给 compose（默认 CPU 版；GPU 版见文件内注释）
docker compose up -d --build
```

构建参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `TORCH_INDEX` | `cpu` | torch wheel 索引后缀，`cpu` 或 `cu130` |
| `TORCH_VERSION` | `2.14.0` | 须与导出 `model.pt2` 时的版本一致，跨版本导图不保证可加载 |
| `PIP_INDEX` | `https://pypi.org/simple` | 其余依赖的来源，国内可换清华镜像 |

## 3 验证

```bash
curl -s localhost:8000/api/v1/health
# {"status":"ok","model_loaded":true,"error":null,"device":"cpu",
#  "forward_backend":"pt2","attribution_backend":"eager","num_classes":10}

curl -s localhost:8000/api/v1/meta          # 类别表 / 专家表 / max_length

# 提交推理任务并轮询结果
TICKET=$(curl -s -X POST localhost:8000/api/v1/infer \
  -H 'Content-Type: application/json' \
  -d '{"texts":["点击链接领取红包，逾期冻结账户"],"explain":false}' \
  | python -c 'import json,sys; print(json.load(sys.stdin)["ticket"])')
curl -s localhost:8000/api/v1/infer/$TICKET
```

容器内实测（CPU 版镜像，开发机 20 逻辑核，`LMAF_WORKERS=4`）：

| 场景 | 耗时 |
| --- | --- |
| 进程启动到 `/health` 可用 | 9–12 s |
| 单条文本推理（`explain=false`） | 0.42–0.46 s |
| 9 条文本批量 | 0.55–0.68 s |
| IG 归因（`explain=true`，`steps=32`，`top_k=5`） | 3.3–3.7 s |
| IG 归因（`steps=16`，指定 `target`） | 2.1 s |
| 4 个任务并发（各 4 条文本） | 总 1.9 s，各自约 1.85 s |

容器内结果与宿主机直接运行逐位一致（同一输入均为「正常 0.8061」，IG 首位 Token 同为 `[CLS] -1.489`），说明 pt2 导图路径在 Linux 容器里工作正常。Python 3.12 与 3.14.8 两个基础镜像跑出的数值完全相同。

## 4 配置

镜像内已写死以下默认值（与 `service/api.py` 的默认一致），运行时可用 `-e` 覆盖：

| 环境变量 | 镜像默认值 | 说明 |
| --- | --- | --- |
| `LMAF_EXPORT_DIR` | `/app/artifacts/export/deploy` | 导出产物目录 |
| `LMAF_WORKERS` | `2` | 推理线程数（`ThreadPoolExecutor`） |
| `LMAF_TASK_TTL` | `3600` | 任务结果保留秒数 |
| `LMAF_HOST` | `0.0.0.0` | 监听地址 |
| `LMAF_PORT` | `8000` | 监听端口 |

几点注意：

- **并发**：单个 uvicorn 进程 + `LMAF_WORKERS` 个推理线程，批内并行靠 torch 自己的线程。CPU 机型把 `LMAF_WORKERS` 设到核数附近即可，再多只会互相抢核；用 `--cpus` 限制容器配额时需要同步调小。
- **端口**：改 `LMAF_PORT` 必须同时改 `-p`（镜像里的 `EXPOSE 8000` 只是声明）。
- **健康检查**：以 `/api/v1/health` 的 `model_loaded` 为准，模型没装上容器会变 `unhealthy`（进程仍在，推理接口返回 503），`start-period` 给了 120 s 覆盖加载时间。
- **内存**：常驻约 2.5–3 GB（`model.pt2` 449 MB + eager 449 MB + 基础模型 411 MB + 运行时开销），建议 `--memory` 不低于 4 GB。

## 5 换模型

**重建镜像（推荐）**：重新导出后重建，模型版本与镜像一一对应。

```bash
python -m scripts.export_model          # 重新产出 artifacts/export/deploy
docker build -t lanmei-antifraud:latest .
```

**挂载卷（不重建）**：模型目录以只读方式覆盖镜像内的同名路径。

```bash
docker run -d --name lanmei-antifraud -p 8000:8000 \
  -v "$PWD/artifacts/export/deploy:/app/artifacts/export/deploy:ro" \
  -v "$PWD/models/macbert:/app/models/macbert:ro" \
  lanmei-antifraud:latest
```

挂载的 `predict_meta.json` 里 `model.path` 仍按 `models/macbert` 解析，基础模型目录的位置不要改。

## 6 排障

| 现象 | 原因与处理 |
| --- | --- |
| `/api/v1/health` 返回 `model_loaded=false` | 看 `docker logs` 里的 `[API] 模型加载失败：...`；多为挂载的产物目录缺文件或 `models/macbert` 缺权重 |
| `forward_backend` 是 `eager` 而不是 `pt2` | 正常：导图在 CPU 上导出，只在 CPU 启用；GPU 上按设计回退 eager |
| 容器 `unhealthy` 但进程还在 | 就是上面的模型加载失败；这是有意设计，便于编排系统重启或告警 |
| 构建上下文过大 / 构建很慢 | `.dockerignore` 已把 `datasets/`、其余 `artifacts/` 阶段、`models/macbert` 的 TF/Flax 权重排除；若新增目录后构建变慢，检查是否漏了排除规则 |
| 首次调用比后续慢 | 首个请求包含线程池预热与 pt2 首次执行，属正常 |
| GPU 版报 `CUDA driver version is insufficient` | 宿主驱动低于 580，升级驱动或换用 CPU 版镜像 |
| Windows + Git Bash 下 `-e LMAF_EXPORT_DIR=/app/...` 被改写成 `D:/Git/app/...` | MSYS 的路径转换所致（Linux 上不存在）；命令前加 `MSYS_NO_PATHCONV=1`，或改用 `docker compose`（环境变量写在 compose 文件里不受影响） |
