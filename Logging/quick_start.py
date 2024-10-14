#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: quick_start.py
@Python Version: 3.12.1
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2024-10-14.
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V2.1
@License: Apache License Version 2.0, January 2004
    Copyright 2024. All rights reserved.

@Description: 
"""

import os
from loguru import logger

if __name__ == "__main__":
    # 调用 logger.add() 方法来配置文件输出
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, "log_file.log")
    logger.add(log_filepath)

    # 设置日志级别, 默认级别为 INFO
    logger.level("DEBUG")

    # 自定义日志格式, 调用 logger.add()方法并设置format 参数, 可以指定日志的格式
    logger.add(log_dir + "app.log", format="[{time:HH:mm:ss}] {level} - {message}")

    # 添加日志切割, 添加日志切割选项来分割文件，以便更好地管理和维护
    # 将日志文件每天切割为一个新文件
    logger.add(log_dir + "app.log", rotation="00:00")

    # Loguru 会自动添加时间戳、日志级别和日志消息内容，并将其输出到终端
    logger.info("this is a INFO level message")
    logger.warning("this is a WARNING level message")
    logger.error("this is a ERROR level message")
