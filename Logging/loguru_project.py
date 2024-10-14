#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: loguru_project.py
@Python Version: 3.12.1
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2024-10-14.
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V2.1
@License: Apache License Version 2.0, January 2004
    Copyright 2024. All rights reserved.

@Description: 利用 loguru 日志库进行二次封装, 便于项目中不同模块和函数重复使用
"""

import logging.handlers
import os
import logging
from functools import wraps
from time import perf_counter

from loguru import logger


class LoggerBase:
    def __init__(
        self,
        prefix="app",
        level="TRACE",
        log_dir="logs",
        max_size=20,
        retention="7 days",
    ):
        self.log_dir = log_dir
        self.prefix = prefix
        self.level = level
        self.max_size = max_size
        self.retention = retention

        self.rotation_ = f"{self.max_size} MB"
        self.encoding_ = f"utf-8"
        self.backtrace_ = True
        self.diagnose_ = True
        self.colorize_ = False
        self.enqueue_ = True
        self.catch_ = True

        self.format_ = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> "
            "| <magenta>{process}</magenta>:<yellow>{thread}</yellow> "
            "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<yellow>{line}</yellow> - <level>{message}</level>"
        )
        self.shared_config = {
            "level": self.level,
            "backtrace": self.backtrace_,
            "diagnose": self.diagnose_,
            "format": self.format_,
            "colorize": self.colorize_,
            "rotation": self.rotation_,
            "retention": self.retention,
            "encoding": self.encoding_,
            "enqueue": self.enqueue_,
            "catch": self.catch_,
        }

        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = self.createLogger()

    def createLogger(self):
        # 采用了层次式的日志记录方式，就是低级日志文件会记录比他高的所有级别日志，
        # 这样可以做到低等级日志最丰富，高级别日志更少更关键
        log_filepath = (
            f"{self.log_dir}/{self.prefix}_{self.level}_{{time:YYYY-MM-DD}}.log"
        )

        # ====== TRACE/DEBUG/INFO/SUCCESS/WARNING/ERROR/CRITICAL ======
        logger.add(
            sink=log_filepath,
            filter=lambda record: record["level"].no >= logger.level(self.level).no,
            **self.shared_config,
        )

        return logger

    def __getattr__(self, level: str):
        return getattr(self.logger, level)

    def log_decorator(self, msg="捕获到异常了, 请排查"):
        """
             日志装饰器，记录函数的名称、参数、返回值、运行时间和异常信息
        Args:
            logger: 日志记录器对象

        Returns:
            装饰器函数

        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.logger.info(f"-----------分割线-----------")
                self.logger.info(f"调用 {func.__name__} args: {args}; kwargs:{kwargs}")
                start = perf_counter()  # 开始时间
                try:
                    result = func(*args, **kwargs)
                    end = perf_counter()  # 结束时间
                    duration = end - start
                    self.logger.info(
                        f"{func.__name__} 返回结果：{result}, 耗时：{duration:4f}s"
                    )
                    return result
                except Exception as e:
                    self.logger.exception(f"{func.__name__}: {msg}")
                    self.logger.info(f"-----------分割线-----------")
                    # raise e

            return wrapper

        return decorator


# ==============================
if __name__ == "__main__":
    log = LoggerBase()
    log_communicate = LoggerBase(prefix="communicate", level="ERROR")
    log_camera = LoggerBase(prefix="camera", level="WARNING")

    @log.log_decorator("This is a unknown problem exception")
    def test_zero_division_error(a, b):
        return a / b

    log_communicate.info("PLC connect successfully")
    log_communicate.info("HOST-UP connect successfully")
    log_communicate.info("Corrector connect successfully")

    log_camera.info("the camera connect successfully")
    log_camera.error("the camera open NOT successfully")

    log.trace("TRACE message")
    log.error("错误信息")
    log.critical("严重错误信息")
    test_zero_division_error(1, 0)
    log.debug("调试信息")
    log.info("普通信息")
    log.success("成功信息")
    log.warning("警告信息")
