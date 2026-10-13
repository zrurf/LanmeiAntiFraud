from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

# 终态集合，仅这些任务会被 TTL 回收
TERMINAL = (DONE, FAILED)


@dataclass
class Task:
    ticket: str
    status: str
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    result: dict | None = None
    error: str | None = None

    def snapshot(self) -> dict:
        return {
            "ticket": self.ticket,
            "status": self.status,
            "created_at": _iso(self.created_at),
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "error": self.error,
            "result": self.result,
        }


def _iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


class TaskQueue:
    """线程池 + 内存任务表

    `submit` 接收零参可调用对象，其返回值作为任务结果；worker 内异常会被
    捕获并写入任务的 `error`，不影响其他任务
    """

    def __init__(self, workers: int, ttl: float, now: Callable[[], float] = time.time):
        self._executor = ThreadPoolExecutor(max_workers=max(1, workers))
        self._ttl = ttl
        self._now = now
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()

    def submit(self, func: Callable[[], dict]) -> str:
        ticket = uuid.uuid4().hex
        with self._lock:
            self._tasks[ticket] = Task(ticket=ticket, status=QUEUED, created_at=self._now())
        self._executor.submit(self._run, ticket, func)
        return ticket

    def _run(self, ticket: str, func: Callable[[], dict]) -> None:
        with self._lock:
            task = self._tasks.get(ticket)
            if task is None:
                return
            task.status = RUNNING
            task.started_at = self._now()
        try:
            result = func()
        except Exception as exc:
            with self._lock:
                task.status = FAILED
                task.error = f"{type(exc).__name__}: {exc}"
                task.finished_at = self._now()
            return
        with self._lock:
            task.status = DONE
            task.result = result
            task.finished_at = self._now()

    def get(self, ticket: str) -> dict | None:
        with self._lock:
            task = self._tasks.get(ticket)
            return task.snapshot() if task is not None else None

    def prune(self) -> int:
        """回收超过 TTL 的终态任务，返回回收数量"""
        now = self._now()
        with self._lock:
            stale = [
                ticket
                for ticket, task in self._tasks.items()
                if task.status in TERMINAL and now - task.created_at > self._ttl
            ]
            for ticket in stale:
                self._tasks.pop(ticket, None)
        return len(stale)