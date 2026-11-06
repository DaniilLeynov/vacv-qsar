from __future__ import annotations

import shutil
import sys
import threading
import time
from dataclasses import dataclass
from typing import TextIO

BAR_WIDTH = 24
FULL = "█"
EMPTY = "░"
RENDER_INTERVAL = 0.05
TICK_INTERVAL = 0.25


def human_size(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            if unit == "B":
                return f"{size:.0f} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


@dataclass
class Task:
    name: str
    fraction: float = 0.0
    pages: int = 0
    read_bytes: int = 0
    total_bytes: int | None = None
    done: bool = False
    status: str = ""
    extra: str = ""
    started_at: float = 0.0


class Progress:
    """Multi-line progress display with one bar per source."""

    def __init__(
        self,
        names: list[str],
        stream: TextIO | None = None,
        enabled: bool | None = None,
        min_interval: float = RENDER_INTERVAL,
    ):
        self.tasks = [Task(name=name) for name in names]
        self.stream = stream or sys.stderr
        if enabled is None:
            enabled = bool(getattr(self.stream, "isatty", lambda: False)())
        self.enabled = enabled
        self.min_interval = min_interval
        self._rendered = 0
        self._last_render = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, index: int) -> None:
        self.tasks[index].started_at = time.monotonic()
        self.tasks[index].extra = ""
        self._ensure_ticker()
        self._render(force=True)

    def page(self, index: int, number: int, total: int | None = None) -> None:
        task = self.tasks[index]
        task.extra = ""
        task.pages = number
        if total and total > 0:
            task.fraction = min(0.95, number / total)
        else:
            task.fraction = min(0.90, task.fraction + 0.1)
        self._render()

    def file_progress(
        self, index: int, read_bytes: int, total_bytes: int | None = None
    ) -> None:
        task = self.tasks[index]
        task.extra = ""
        task.read_bytes = read_bytes
        task.total_bytes = total_bytes
        if total_bytes and total_bytes > 0:
            task.fraction = min(0.99, read_bytes / total_bytes)
        self._render()

    def retry(self, index: int, attempt: int, status: int | None = None) -> None:
        task = self.tasks[index]
        task.extra = f"retry {attempt}"
        if status:
            task.extra += f" ({status})"
        self._render(force=True)

    def finish(self, index: int, status: str = "") -> None:
        task = self.tasks[index]
        task.done = True
        task.extra = ""
        task.fraction = 1.0
        task.status = status
        self._render(force=True)

    def close(self) -> None:
        self._stop.set()
        if self.enabled:
            self.stream.write("\n")
            self.stream.flush()

    def _ensure_ticker(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._tick, name="fetch-progress", daemon=True
        )
        self._thread.start()

    def _tick(self) -> None:
        while not self._stop.wait(TICK_INTERVAL):
            if any(not task.done for task in self.tasks):
                self._render(force=True)

    def _label(self, task: Task) -> str:
        parts = [task.name]
        if task.pages:
            parts.append(f"pages {task.pages}")
        if task.read_bytes:
            size = human_size(task.read_bytes)
            if task.total_bytes:
                size += f"/{human_size(task.total_bytes)}"
            parts.append(size)
        if not task.done and task.started_at:
            parts.append(f"{time.monotonic() - task.started_at:.1f}s")
        if task.extra:
            parts.append(task.extra)
        if task.status:
            parts.append(task.status)
        return " · ".join(parts)

    def _render(self, force: bool = False) -> None:
        if not self.enabled:
            return
        with self._lock:
            now = time.monotonic()
            if not force and now - self._last_render < self.min_interval:
                return
            self._last_render = now

            width = shutil.get_terminal_size(fallback=(80, 24)).columns
            limit = max(1, width - 1)

            lines = []
            for task in self.tasks:
                ratio = 1.0 if task.done else min(max(task.fraction, 0.0), 1.0)
                filled = int(round(ratio * BAR_WIDTH))
                bar = FULL * filled + EMPTY * (BAR_WIDTH - filled)
                line = f"[{bar}] {ratio * 100:3.0f}%  {self._label(task)}"
                lines.append(line[:limit])

            if self._rendered > 1:
                self.stream.write(f"\033[{self._rendered - 1}A")
            for position, line in enumerate(lines):
                self.stream.write("\r\033[K" + line)
                if position < len(lines) - 1:
                    self.stream.write("\n")
            self.stream.flush()
            self._rendered = len(lines)
