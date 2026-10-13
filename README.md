# LanmeiAntiFraud

## 1 环境准备

### 1.1 依赖

| 类别 | 依赖 |
| --- | --- |
| 训练 | `torch`（>= 2.x）、`transformers`、`peft` |
| 数据 | `lance`、`pyarrow` |
| 指标 | `scikit-learn`、`scipy`、`numpy` |
| 中文处理 | `jieba` |
| 实验增强 | `emoji`、`moodemoji`、`char_similar_z`、`nlpcda`、`HomoglyphsCJK` |
| 服务 | `fastapi`、`uvicorn`、`pydantic`、`httpx` |
| 测试 | `pytest` |

### 1.2 准备基础模型

```bash
pip install modelscope
modelscope download --model hfl/chinese-macbert-base --local_dir ./models/macbert
```

### 1.3 准备数据集

数据集为 Lance 格式，路径 `datasets/dataset.lance`，列包含 `id / text / label10 / subtype / source / use / meta`。按 `use` 列划分：

| split | 行数 | 用途 |
| --- | --- | --- |
| `mlm` | 1,373,661 | MLM 预训练 |
| `train` | 399,959 | 各阶段训练 |
| `val` | 49,993 | 阶段内选优 |
| `test` | 49,993 | 正式评估 |
| `test_synthetic` | 1,341,078 | 合成鲁棒性测试 |

### 1.4 类别与专家对应

**专家的 `index` 即类别 id**，二者严格一一对应（导出、服务与评估均依赖该约定）。类别名取自对应专家配置的 `name`：

| index | id | 类别名 | 池化 | heads | hidden | layers | 数据增强 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `normal` | 正常 | `mean_max` | 1 | 256 | 2 | — |
| 1 | `advertising` | 广告骚扰 | `mean` | 1 | 256 | 1 | `emoji_replace`（30%） |
| 2 | `fake_shopping` | 虚假购物 | `cls_mean` | 1 | 512 | 3 | `random_space`（20%） |
| 3 | `brushing` | 刷单 | `attention` | 1 | 256 | 3 | — |
| 4 | `impersonation_customer_service` | 冒充客服 | `attention` | 4 | 512 | 3 | — |
| 5 | `impersonation_public_sec` | 冒充公检法 | `max` | 1 | 384 | 2 | `homoglyph`（40%） |
| 6 | `impersonation_acquaintances` | 冒充领导熟人 | `last_mean` | 1 | 384 | 2 | — |
| 7 | `investment` | 投资理财 | `median` | 1 | 512 | 2 | — |
| 8 | `loan_credit` | 贷款征信 | `min` | 1 | 384 | 2 | — |
| 9 | `relationship` | 关系型 | `max_attn` | 1 | 512 | 3 | — |

可用池化算子：`mean / max / min / median / cls / last_mean / attention / mean_max / cls_mean / max_attn`。
可用增强算子：`emoji_replace / emoji_mood / homoglyph / random_space / homophone / random_delete / char_swap / random_entity`。

括号内为专家的 `augment_rate`：**被增强的样本比例**（默认 `1.0`）。它与算子自身的命中率是相乘关系。
以训练集均匀随机抽取 5 万条实测，三个算子的自身命中率分别为：`random_space` 95.2%、`emoji_replace` 66.4%、
`homoglyph` 52.6%（77 对核定字表；换成原 10 对字表且开启相似度闸门时只有 2.8%）。
因此三者分别取 20%、30%、40% 的抽样率，**整体改写率被对齐在 19%~21%**（5 轮训练里每个样本平均被改写约 1 次），
而不是让某一个算子的剂量显著偏大或偏小。增强只作用于训练流，验证集与测试集不做增强；
`augment = []` 的专家完全不付这项开销。

---

## 2 目录结构

```
train/                  训练侧
├── config/             配置数据类与解析（TOML + 环境变量 + 命令行）
├── data/               Lance 数据集读取与数据增强算子
├── model/              backbone 封装、损失函数（CB / Focal / 加权 CE / 负载均衡）
├── expert/             专家网络、池化算子、MoE 分类器
├── router/             门控路由器
├── stage/              五个训练阶段
├── common/             评估指标（训练与实验共用）
└── __main__.py         训练入口
service/                推理服务
├── model.py            导出产物加载
├── attribution.py      IG Token 级归因
├── predictor.py        同步推理封装
├── runtime.py          服务运行时（eager + .pt2 双路径）
├── tasks.py            任务队列
└── api.py              FastAPI 应用
scripts/export_model.py 模型导出
tests/                  测试
└── baseline/           基线 / 消融实验入口
docs/tutorial/          环境与使用文档
config.toml             项目配置
```

---

## 3 配置

配置优先级：**命令行 > 配置文件 > 环境变量**。

- 配置文件：`config.toml`
- 环境变量前缀 `LMAF`，层级用 `__` 分隔，例如 `LMAF_LOSS__DEFAULTS__SELF_WEIGHT=2.5`
- 命令行覆盖：字段名中的 `_` 与层级 `.` 均写成 `-`，例如

```bash
python -m train stage1 --no-resume --train-clf-train-batch-size 64 --loss-defaults-beta 0.9999
```

配置项覆盖语法对训练、导出、基线实验三个入口一致。

> `config.toml` 默认 `resume = true`：若存在断点则续训，否则从头开始。全新训练建议显式传入 `--no-resume` 以免误续。

---

## 4 训练

### 4.1 全流程

```bash
python -m train all --no-resume
```

按 `pretrain → stage1 → stage2 → stage3 → stage4` 依次执行，任一阶段失败即中断。

### 4.2 分阶段

```bash
python -m train pretrain    # MLM 预训练
python -m train stage1      # 分类头训练（MacBERT + DoRA 单头基线）
python -m train stage2      # 专家网络训练
python -m train stage3      # 路由网络训练
python -m train stage4      # 联合训练
```

| 阶段 | 训练目标 | 冻结 | 输入权重 | 输出 |
| --- | --- | --- | --- | --- |
| `pretrain` | MLM | — | `models/macbert` | `artifacts/export/mlm/model.pt` |
| `stage1` | `clf_head`（+ backbone 微调） | 专家、路由器 | `export/mlm` | `artifacts/export/clf/model.pt` |
| `stage2` | 10 个专家（逐个） | backbone | `export/clf` | `artifacts/export/expert/model.pt` |
| `stage3` | 路由器 | backbone、专家 | `export/expert` | `artifacts/export/router/model.pt` |
| `stage4` | 专家 + 路由器联合 | backbone、`clf_head` | `export/router` | `artifacts/export/joint/model.pt` |

每个阶段按 val Macro-F1 选优写 `export/<stage>/model.pt`，同时把断点、优化器状态与历史写入 `artifacts/<stage>/`。

### 4.3 损失函数

- 全部阶段使用 **Class-Balanced Loss**（`loss.defaults.beta`，默认 `0.99999`）作为基础分类损失；
- `stage2` 额外启用专家分工权重 `lambda_self`（`self_weight`，默认 `3.0`）：专家 `i` 对自身负责类别加权；
- `stage3` / `stage4` 在融合输出上叠加 **负载均衡损失**（`balance_weight`，默认 `0.01`），鼓励样本在专家间均匀分配。

### 4.4 速度相关

三项默认开启的优化，分别削减「无效 token」「单位 token 耗时」与「数据供给瓶颈」：

| 配置项 | 默认 | 说明 |
| --- | --- | --- |
| `train.amp` | `true` | bf16 自动混合精度（`torch.autocast`）。损失仍在 fp32 下计算；bf16 与 fp32 指数范围一致，因此不需要 GradScaler；GPU 不支持 bf16 时自动退化为 fp32 |
| `train.length_bucketing` | `true` | 长度分桶采样：训练侧先把样本随机切块、块内按长度排序组批（批内长度接近，样本组合仍随机），评估侧全量按长度排序（指标与顺序无关）。逐轮重新分桶，顺序由 `(seed, epoch)` 决定，续训可复现 |
| `train.num_workers` | `6` | 分词与数据增强的子进程数。两者都在 `collate` 里做、是纯 CPU 的 Python 代码，单进程约 2.3k 样本/秒，在快卡上会成为瓶颈。每个 worker 固定单线程并按 worker 种子重播 `random`，避免各进程增强样本重复 |

按原始配置训练（三项都关）：

```bash
python -m train all --no-train-amp --no-train-length-bucketing --train-num-workers 0
```

本机（开发机）实测，用于预估训练机上正式训练的耗时（**正式训练不在本机进行，而是在 AutoDL 的 GPU 服务器上**（RTX PRO 6000 Blackwell，环境见 `docs/实验记录.md`））；本表为 RTX 4060 Laptop 8GB / 70W 限功耗、真实模型与真实数据、`scripts/autodl/preflight.py` 同一轮输出：

| 指标 | 原始 | 优化后 |
| --- | --- | --- |
| train 集批长（batch=32） | 486（真实均值 182） | 176（0.97× 真实均值） |
| mlm 集批长（batch=32） | 227（真实均值 47） | 53（1.13×） |
| 单步耗时，backbone 参与训练（batch=16×410） | 993 ms | 346 ms（2.87×） |
| 单步耗时，backbone 冻结（batch=16×410） | 326 ms | 167 ms（1.95×） |
| 全流程预估（不含验证集评估） | 250.4 h | 40.0 h |

> 表中的"真实均值"是数据集预计算的长度估计 `len(text) + 2`（见 `SplitDataset.lengths`，直接作为长度分桶的分组依据），
> 不是分词后的 token 数——中文语料下两者很接近（如 train 集按此估计均值 182、按字符数均值 180.3）。

分项收益（相对 40.0 h）：只关长度分桶 → 115.3 h（慢 2.9×）；只关 AMP → 85.7 h（慢 2.1×）。

> 动态 padding 的浪费来自长尾分布：随机组批时批内最长样本普遍远高于真实均值
> （train 集 2.67×、mlm 集 4.89×），这部分补齐的算力不带来任何信息增益。
> bf16 的数值影响实测很小：同一初始化下训练 30 步，参数更新量的模长相差 0.02%、方向余弦 0.998。
>
> 换机器后可用 `python -m scripts.autodl.preflight` 重测：它分别测 fp32/bf16，
> 按数据集真实长度分布给出各阶段耗时预估，并打印关闭优化后的对比。

---

## 5 导出

将训练权重导出为 PyTorch 官方部署格式 `torch.export` 的 ExportedProgram（`model.pt2`）：

```bash
# 自动按 joint → router → expert → clf 优先级选取权重
python -m scripts.export_model

# 指定权重与输出目录
python -m scripts.export_model \
    --checkpoint artifacts/export/joint/model.pt \
    --out artifacts/export/deploy
```

产物目录（自包含，运行时只需该目录）：

```
artifacts/export/deploy/
├── model.pt2           ExportedProgram，运行时前向推理
├── model.pt            eager state_dict，运行时梯度归因（IG）
├── predict_meta.json   类别名 / 专家配置 / 模型结构 / 最大长度
└── tokenizer 相关文件
```

导出图支持**动态 batch 与序列长度**（序列上限为 `data.max_length`）。

> **设备约束**：导出图由 `torch.export` 在 **CPU** 上生成，其参数不随设备迁移。因此服务在 CPU 上走 `model.pt2`，在 GPU 上自动回退 eager（数值与导图一致），`/api/v1/health` 的 `forward_backend` 会如实返回 `pt2` 或 `eager`。

---

## 6 在线服务

### 6.1 启动

```bash
python -m service.api
# 或
uvicorn service.api:app --host 0.0.0.0 --port 8000
```

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LMAF_EXPORT_DIR` | `artifacts/export/deploy` | 导出产物目录 |
| `LMAF_WORKERS` | `2` | 推理线程数 |
| `LMAF_TASK_TTL` | `3600` | 任务结果保留秒数 |
| `LMAF_HOST` | `0.0.0.0` | 监听地址 |
| `LMAF_PORT` | `8000` | 监听端口 |

模型加载失败不会中断进程：`/health` 返回 `model_loaded=false` 与错误信息，推理接口返回 `503`。

### 6.2 端点

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/v1/infer` | 提交推理任务，返回 `ticket`（`202`） |
| `GET` | `/api/v1/infer/{ticket}` | 查询任务状态与结果 |
| `GET` | `/api/v1/meta` | 静态元数据（类别表、专家表、`max_length`） |
| `GET` | `/api/v1/health` | 健康检查与后端信息 |

predict 与 explain 合并为同一端点，由 `explain` 开关区分；批量由 `texts` 数组承载。

### 6.3 Schema

提交请求：

```json
{
  "texts": ["点击链接领取红包，逾期冻结账户"],
  "explain": false,
  "top_k": 8,
  "steps": 32,
  "target": null
}
```

约束：`texts` 长度 1–256；`top_k` 1–32；`steps` 1–128；`target` 为 `null` 或类别 id（越界返回 `422`）。

提交响应（`202`）：

```json
{ "ticket": "3f9c...", "status": "queued" }
```

查询响应：

```json
{
  "ticket": "3f9c...",
  "status": "queued | running | done | failed",
  "created_at": "2026-10-12T09:00:00+00:00",
  "started_at": "2026-10-12T09:00:00+00:00",
  "finished_at": "2026-10-12T09:00:01+00:00",
  "error": null,
  "result": {
    "items": [
      {
        "text": "点击链接领取红包，逾期冻结账户",
        "label_id": 4,
        "label_key": "impersonation_customer_service",
        "label_name": "冒充客服",
        "confidence": 0.9731,
        "probabilities": [0.001, 0.002, 0.003, 0.001, 0.9731, 0.01, 0.004, 0.002, 0.002, 0.0019],
        "experts_opinion": [
          {
            "id": 0,
            "key": "normal",
            "probabilities": [0.1, 0.1, 0.1, 0.1, 0.3, 0.1, 0.05, 0.05, 0.05, 0.05],
            "weight": 0.12
          }
        ],
        "explanation": null
      }
    ]
  }
}
```

- `experts_opinion` 为对象数组，顺序与专家维一致；`weight` 为该样本上的门控权重；
- `explain=true` 时 `explanation` 为 `{ "target_label_id": int, "attributions": [{ "token": str, "score": float }] }`，按 `|score|` 降序取 `top_k` 条；否则为 `null`。

### 6.4 curl 示例

```bash
# 提交
TICKET=$(curl -s -X POST http://127.0.0.1:8000/api/v1/infer \
  -H 'Content-Type: application/json' \
  -d '{"texts":["点击链接领取红包"],"explain":true,"top_k":5,"steps":32}' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["ticket"])')

# 轮询
curl -s http://127.0.0.1:8000/api/v1/infer/$TICKET

# 元数据与健康检查
curl -s http://127.0.0.1:8000/api/v1/meta
curl -s http://127.0.0.1:8000/api/v1/health
```

---

## 7 基线 / 消融实验

三条基线与 MoE 消融共用同一测试集与指标口径（Accuracy / Macro-F1 / 逐类 P-R-F1 / 混淆矩阵 / ECE / Macro-F1 bootstrap 置信区间 / McNemar 精确检验 / 配对 bootstrap 差值），结果写入 `artifacts/results/`。

| 配置名 | 说明 |
| --- | --- |
| `tfidf_nb` | TF-IDF（jieba 词级 1–2 gram）+ 多项式朴素贝叶斯 |
| `tfidf_lr` | TF-IDF + 逻辑回归 |
| `macbert_dora_single` | MacBERT + DoRA 单头（取 `export/clf`） |
| `moe_gated` | MoE 门控融合（正式） |
| `moe_equal` | 消融 A1：等权平均融合 |
| `moe_top1` | 消融 A2：top-1 硬选择融合 |

### 7.1 传统方法基线

```bash
python -m tests.baseline.tfidf
# 限流（全量 TF-IDF + 逻辑回归较吃内存）
python -m tests.baseline.tfidf --max-train 100000 --max-features 200000
```

### 7.2 单配置评估

```bash
# MoE 门控（默认权重：export/joint，缺失时回退 export/router）
python -m tests.baseline.eval_testset --mode moe --fusion gated

# 消融：等权 / top-1
python -m tests.baseline.eval_testset --mode moe --fusion equal
python -m tests.baseline.eval_testset --mode moe --fusion top1

# 单头基线
python -m tests.baseline.eval_testset --mode single

# 与另一 checkpoint 对照（追加 McNemar 与配对 bootstrap）
python -m tests.baseline.eval_testset \
    --mode moe --fusion gated \
    --compare artifacts/export/clf/model.pt --compare-mode single \
    --out artifacts/results/moe_vs_single.json
```

### 7.3 一键对比

```bash
python -m tests.baseline.compare
```

跑齐全部配置并打印对比表，以 `moe_gated` 为基准输出显著性检验。缺少对应权重的配置会被自动跳过（`macbert_dora_single` 需要 `export/clf/model.pt`，MoE 需要 `export/joint` 或 `export/router`）。

---

## 8 测试

```bash
python -m pytest -q
```

---

## 9 产物目录

```
artifacts/
├── mlm/ clf/ expert/ router/ joint/    各阶段断点、优化器状态、metrics.json
├── export/
│   ├── mlm/ clf/ expert/ router/ joint/    各阶段最优权重（model.pt + tokenizer）
│   └── deploy/                             导出的部署产物（model.pt2 + model.pt + meta）
└── results/                                基线与消融实验报告（JSON）
```

---

## 10 AutoDL 部署

`scripts/autodl/` 提供在 AutoDL 实例上完成「上传 → 装环境 → 预检 → 训练 → 取回产物」的一组脚本，
详见 [`scripts/autodl/README.md`](scripts/autodl/README.md)。典型顺序：

```bash
# 本地：推送源码与数据集（分片上传，可续传），基础模型在实例上用 modelscope 下载
LMAF_HOST=<host> LMAF_PORT=<port> bash scripts/autodl/upload.sh
# 实例：装依赖（requirements 中的 torch 与镜像不匹配时自动沿用镜像自带版本）+ pytest 自检
bash scripts/autodl/remote_setup.sh
# 实例：真实数据下的耗时与显存预检
python -m scripts.autodl.preflight
# 实例：后台训练（nohup，日志 artifacts/logs/train_<时间戳>.log）
bash scripts/autodl/remote_train.sh
# 本地：取回产物
LMAF_HOST=<host> LMAF_PORT=<port> bash scripts/autodl/pull_artifacts.sh
```

> 首次训练请传 `--no-resume`（`remote_train.sh` 默认已带），避免误续旧断点。后续训练请传 `--resume`。