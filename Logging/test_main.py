#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: test_main.py
@Python Version: 3.12.1
@Platform: PyTorch 2.0.0 + cu117
@Author: Wei Li (Ithaca)
@Date: 2024-08-06.
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V2.1
@License: Apache License Version 2.0, January 2004
    Copyright 2024. All rights reserved.

@Description: 
1. https://blog.csdn.net/pansaky/article/details/90710751
2. https://blog.csdn.net/Code_LT/article/details/140301560
3. https://blog.csdn.net/2301_80240808/article/details/134387521
"""

import logging
import logging.config


if __name__ == "__main__":
    logging.config.fileConfig("logging.conf")
    rootLogger = logging.getLogger()
    rootLogger.debug("This is root logger, debug message")

    appLogger = logging.getLogger("applog")
    appLogger.debug("This is root logger, debug message")

    var = "Ithaca"
    try:
        int(var)
    except Exception as exp:
        appLogger.exception(exp)
