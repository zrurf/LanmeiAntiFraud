"""FastAPI 推理服务

启动:

    uvicorn service.api:app
    python -m service.api

环境变量:

    LMAF_EXPORT_DIR  导出产物目录，默认 artifacts/export/deploy
    LMAF_WORKERS     推理线程数，默认 2
    LMAF_TASK_TTL    任务结果保留秒数，默认 3600
    LMAF_HOST        uvicorn 监听地址，默认 0.0.0.0
    LMAF_PORT        uvicorn 监听端口，默认 8000
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from service.runtime import Runtime
from service.tasks import QUEUED, TaskQueue

DEFAULT_EXPORT_DIR = "artifacts/export/deploy"

MAX_TEXTS = 256
# 后台清理任务的轮询间隔（秒）
CLEANUP_INTERVAL = 60.0


class InferRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_TEXTS)
    explain: bool = False
    top_k: int = Field(default=8, ge=1, le=32)
    steps: int = Field(default=32, ge=1, le=128)
    target: int | None = None


async def _cleanup_loop(queue: TaskQueue) -> None:
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL)
        removed = queue.prune()
        if removed:
            print(f"[API] 回收过期任务 {removed} 个", flush=True)


def create_app(
    export_dir: str | Path | None = None,
    workers: int | None = None,
    ttl: float | None = None,
) -> FastAPI:
    """构建应用；`export_dir` / `workers` / `ttl` 缺省时取环境变量"""
    export_dir = Path(export_dir or os.getenv("LMAF_EXPORT_DIR", DEFAULT_EXPORT_DIR))
    workers = workers if workers is not None else int(os.getenv("LMAF_WORKERS", "2"))
    ttl = ttl if ttl is not None else float(os.getenv("LMAF_TASK_TTL", "3600"))

    queue = TaskQueue(workers=workers, ttl=ttl)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.runtime = Runtime(export_dir)
            app.state.error = None
        except Exception as exc:
            # 加载失败不阻断进程，由 /health 反映，推理接口返回 503
            app.state.runtime = None
            app.state.error = f"{type(exc).__name__}: {exc}"
            print(f"[API] 模型加载失败：{app.state.error}", flush=True)
        cleanup = asyncio.create_task(_cleanup_loop(queue))
        yield
        cleanup.cancel()

    app = FastAPI(title="LanmeiAntiFraud Serving", lifespan=lifespan)
    app.state.queue = queue

    @app.get("/api/v1/health")
    def health(request: Request) -> dict:
        runtime: Runtime | None = request.app.state.runtime
        return {
            "status": "ok",
            "model_loaded": runtime is not None,
            "error": request.app.state.error,
            "device": str(runtime.device) if runtime else None,
            "forward_backend": runtime.forward_backend if runtime else None,
            "attribution_backend": "eager" if runtime else None,
            "num_classes": runtime.num_classes if runtime else None,
        }

    @app.get("/api/v1/meta")
    def meta(request: Request) -> dict:
        runtime = _require_runtime(request)
        return {
            "num_classes": runtime.num_classes,
            "max_length": runtime.max_length,
            "labels": runtime.labels,
            "experts": runtime.experts,
        }

    @app.post("/api/v1/infer", status_code=202)
    def infer(payload: InferRequest, request: Request) -> dict:
        runtime = _require_runtime(request)
        if payload.target is not None and not 0 <= payload.target < runtime.num_classes:
            raise HTTPException(
                status_code=422,
                detail=f"target 需落在 [0, {runtime.num_classes}) 内",
            )
        queue: TaskQueue = request.app.state.queue
        ticket = queue.submit(lambda: runtime.infer(**payload.model_dump()))
        return {"ticket": ticket, "status": QUEUED}

    @app.get("/api/v1/infer/{ticket}")
    def task(ticket: str, request: Request) -> dict:
        snapshot = request.app.state.queue.get(ticket)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="ticket 不存在或已过期")
        return snapshot

    return app


def _require_runtime(request: Request) -> Runtime:
    runtime: Runtime | None = request.app.state.runtime
    if runtime is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    return runtime


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("LMAF_HOST", "0.0.0.0"),
        port=int(os.getenv("LMAF_PORT", "8000")),
    )


if __name__ == "__main__":
    main()