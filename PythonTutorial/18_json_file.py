#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 18_json_file.py
@Python Version: 3.12.13
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2026-09-30
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V0.1
@License: Apache License Version 2.0, January 2004
    Copyright 2026. All rights reserved.

@Description: Python 操作JSON格式数据, 同步保存json文件到本地
"""

"""jsonstore.py —— 生产级 JSON 文件读写封装。"""

# from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any
from collections.abc import Callable


class JsonStore:
    """线程安全的 JSON 文件存储，写入原子化，支持自定义类型扩展。

    用法::

        store = JsonStore("data/app.json", default_factory=dict)
        cfg = store.read()
        cfg["theme"] = "dark"
        store.write(cfg)
    """

    def __init__(
        self,
        path: str | Path,
        *,
        indent: int | None = 2,
        ensure_ascii: bool = False,
        encoding: str = "utf-8",
        default: Callable[[Any], Any] | None = None,
        default_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.indent = indent
        self.ensure_ascii = ensure_ascii
        self.encoding = encoding
        self.default = default
        self.default_factory = default_factory
        self._lock = threading.RLock()

    # ---------- 读 ----------
    def read(self) -> Any:
        with self._lock:
            try:
                text = self.path.read_text(encoding=self.encoding)
            except FileNotFoundError:
                if self.default_factory is not None:
                    return self.default_factory()
                raise

            if not text.strip():
                if self.default_factory is not None:
                    return self.default_factory()
                raise ValueError(f"文件为空: {self.path}")

            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{self.path}:{e.lineno}:{e.colno} JSON 解析失败: {e.msg}"
                ) from e

    # ---------- 写 ----------
    def write(self, data: Any, *, atomic: bool = True) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if atomic:
                self._atomic_write(data)
            else:
                with self.path.open("w", encoding=self.encoding, newline="\n") as f:
                    self._dump(data, f)

    def update(self, mutator: Callable[[Any], Any]) -> Any:
        """读取 → 修改 → 写回，整个过程持锁，避免并发丢更新。"""
        with self._lock:
            data = self.read()
            result = mutator(data)
            self.write(data if result is None else result)
            return data

    # ---------- 内部 ----------
    def _dump(self, data: Any, fp) -> None:
        json.dump(
            data,
            fp,
            ensure_ascii=self.ensure_ascii,
            indent=self.indent,
            default=self.default,
            allow_nan=False,
        )
        fp.write("\n")

    def _atomic_write(self, data: Any) -> None:
        fd, tmp_name = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp"
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding=self.encoding, newline="\n") as f:
                self._dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.path)
            self._fsync_dir()
        finally:
            tmp_path.unlink(missing_ok=True)

    def _fsync_dir(self) -> None:
        """同步目录项，保证 rename 本身也持久化（仅 POSIX 有意义）。"""
        if os.name != "posix":
            return
        try:
            dfd = os.open(self.path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)


# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    from datetime import datetime

    def encode(o):
        if isinstance(o, datetime):
            return o.isoformat()
        raise TypeError(f"不可序列化: {type(o).__name__}")

    store = JsonStore("data/app.json", default=encode, default_factory=dict)

    store.write(
        {
            "app": "demo",
            "locale": "zh-CN",
            "started_at": datetime.now(),
            "features": {"dark_mode": True},
        }
    )

    store.update(lambda cfg: cfg["features"].update({"beta": False}))
    print(store.read())
