#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 14_logger.py
@Python Version: 3.12.13
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2026-09-22
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V0.1
@License: Apache License Version 2.0, January 2004
    Copyright 2026. All rights reserved.

@Description: 现代 Python 日志记录功能
"""

import os
import sys
import logging
import logging.handlers
import threading
from datetime import datetime


class Logger:
    __instance = None

    def __init__(
        self,
        log_filename,
        log_folder="logs",
        level=logging.DEBUG,
        console=True,
        time_rotation=True,
        max_bytes=10 * 1024 * 1024,
        backup_count=50,
    ):
        """线程安全的同步日志记录器
        :param log_filename: 日志文件名
        :param log_folder: 日志文件文件夹, default="logs"
        :param level: 日志级别, DEBUG
        :param console: 是否在控制台输出, 默认为True
        :param time_rotation: 是否启用时间轮转, 默认为True(按天轮转)
        :param max_bytes: 日志文件最大字节数(用于大小轮转), 默认10MB
        :param backup_count: 保留的备份文件数量, 默认5个
        """
        self.locker = threading.Lock()
        file_basename, file_extension = os.path.splitext(os.path.basename(log_filename))
        self.logger = logging.getLogger(file_basename)
        self.logger.setLevel(level)

        # 清除现有处理器避免重复
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        os.makedirs(log_folder, exist_ok=True)
        log_file_ = (
            file_basename + "_" + datetime.now().strftime("%Y-%m-%d") + file_extension
        )
        log_filepath = os.path.join(log_folder, log_file_)
        if time_rotation:
            # 按天轮转，午夜切换，文件名添加日期后缀
            file_handler = logging.handlers.TimedRotatingFileHandler(
                log_filepath,
                when="midnight",
                interval=1,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.suffix = "%Y-%m-%d-%H-%M-%S"
        else:
            # 按大小轮转
            file_handler = logging.handlers.RotatingFileHandler(
                log_filepath,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )

        # 默认格式设置
        default_fmt = "[%(asctime)s]-[%(name)s][%(levelname)s]-[%(process)d:%(processName)s][%(thread)d:%(threadName)s]-[%(funcName)s:%(lineno)d]-[%(message)s]-%(pathname)s"
        file_handler.setFormatter(logging.Formatter(default_fmt))
        self.logger.addHandler(file_handler)

        if console:
            console_handler_stdout = logging.StreamHandler(sys.stdout)
            console_handler_stdout.setFormatter(logging.Formatter(default_fmt))
            self.logger.addHandler(console_handler_stdout)

    def GetLogger(self) -> logging.Logger:
        return self.logger

    def Debug(self, message) -> None:
        self.locker.acquire()
        self.logger.debug(message)
        self.locker.release()

    def Info(self, message) -> None:
        self.locker.acquire()
        self.logger.info(message)
        self.locker.release()

    def Warn(self, message) -> None:
        self.locker.acquire()
        self.logger.warning(message)
        self.locker.release()

    def Error(self, message) -> None:
        self.locker.acquire()
        self.logger.error(message)
        self.locker.release()

    def Critical(self, message) -> None:
        self.locker.acquire()
        self.logger.critical(message)
        self.locker.release()

    def Exception(self, message) -> None:
        self.locker.acquire()
        self.logger.exception(message)
        self.locker.release()

    @classmethod
    def GetInstance(cls) -> "Logger":
        if cls.__instance is None:
            cls.__instance = Logger(log_filename="default.log")
        return cls.__instance


# ---------------------------
if __name__ == "__main__":
    test_filename = "test.log"
    test_logger = Logger(test_filename)

    test_logger.Debug("This is a Debug message")
    test_logger.Info("This is a Info message")
    test_logger.Warn("This is a Warning message")
    test_logger.Error("This is a Error message")
    test_logger.Critical("This is a Critical message")

    Logger.GetInstance().Debug("Debug Message")
