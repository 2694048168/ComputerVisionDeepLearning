#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: loguru_logger.py
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

import os
from functools import wraps
from time import perf_counter

from loguru import logger


class LoggerBase:
    """
    根据时间、文件大小切割日志
    """

    def __init__(self, log_dir="logs", max_size=20, retention="7 days"):
        self.log_dir = log_dir
        self.max_size = max_size
        self.retention = retention
        self.logger = self.configure_logger()

    def set_level(self, level):
        if self.logger is not None:
            self.logger.level(level)

    def configure_logger(self):
        """

        Returns:

        """
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)

        shared_config = {
            "level": "TRACE",
            "enqueue": True,
            "backtrace": True,
            # "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            # 格式里面添加了process和thread记录，方便查看多进程和线程程序
            "format": "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> "
            "| <magenta>{process}</magenta>:<yellow>{thread}</yellow> "
            "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<yellow>{line}</yellow> - <level>{message}</level>",
        }

        # 添加按照日期和大小切割的文件 handler
        logger.add(
            sink=f"{self.log_dir}/{{time:YYYY-MM-DD}}.log",
            rotation=f"{self.max_size} MB",
            retention=self.retention,
            **shared_config,
        )

        # 配置按照等级划分的文件 handler 和控制台输出
        logger.add(sink=self.get_log_path, **shared_config)

        return logger

    def get_log_path(self, message: str) -> str:
        """
        根据等级返回日志路径
        Args:
            message:

        Returns:

        """
        log_level = message.record["level"].name.lower()
        log_file = f"{log_level}.log"
        log_path = os.path.join(self.log_dir, log_file)

        return log_path

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

    @log.log_decorator("This is a unknown problem exception")
    def test_zero_division_error(a, b):
        return a / b

    log.trace("TRACE message")
    log.error("错误信息")
    log.critical("严重错误信息")
    test_zero_division_error(1, 0)
    log.debug("调试信息")
    log.info("普通信息")
    log.success("成功信息")
    log.warning("警告信息")
