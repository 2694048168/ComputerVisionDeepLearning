#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: log_test.py
@Python Version: 3.12.1
@Platform: PyTorch 2.0.0 + cu117
@Author: Wei Li (Ithaca)
@Date: 2024-08-05.
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V2.1
@License: Apache License Version 2.0, January 2004
    Copyright 2024. All rights reserved.

@Description: 
1. [python logging](https://docs.python.org/zh-cn/3/library/logging.html)
2. [logging format](https://docs.python.org/zh-cn/3/library/logging.html#logrecord-attributes)
3. [logging handler](https://docs.python.org/zh-cn/3/library/logging.html#logging.Handler)
"""

import sys
import logging


# 自定义Logger类 https://www.amd5.cn/atang_4863.html
# 基本的日志轮转方式，这个类是日志轮转的基类，后面日志按时间轮转，按大小轮转的类都继承于此
# https://blog.csdn.net/weixin_43790276/article/details/101944628


def test_basic():
    print("当前函数名称:", sys._getframe().f_code.co_name)

    logging.debug("this is a DEBUG log message")
    logging.info("this is a INFO log message")
    logging.warning("this is a WARN log message")
    logging.error("this is a ERROR log message")
    logging.critical("this is a CRITICAL log message")
    print("=======================================")


def test_level():
    print("当前函数名称:", sys._getframe().f_code.co_name)
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.DEBUG)
    logging.debug("this is a DEBUG log message")
    logging.info("this is a INFO log message")
    logging.warning("this is a WARN log message")
    logging.error("this is a ERROR log message")
    logging.critical("this is a CRITICAL log message")
    print("=======================================")


def test_format():
    print("当前函数名称:", sys._getframe().f_code.co_name)
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s]-[%(name)s][%(levelname)s]-[%(process)d:%(processName)s][%(thread)d:%(threadName)s]-[%(message)s]-%(pathname)s",
    )
    logging.debug("this is a DEBUG log message")
    logging.info("this is a INFO log message")
    logging.warning("this is a WARN log message")
    logging.error("this is a ERROR log message")
    logging.critical("this is a CRITICAL log message")
    print("=======================================")


def test_logfile():
    print("当前函数名称:", sys._getframe().f_code.co_name)
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s]-[%(name)s][%(levelname)s]-[%(process)d:%(processName)s][%(thread)d:%(threadName)s]-[%(message)s]-%(pathname)s",
        filename="log.txt",
        filemode="a",
    )
    logging.debug("this is a DEBUG log message")
    logging.info("this is a INFO log message")
    logging.warning("this is a WARN log message")
    logging.error("this is a ERROR log message")
    logging.critical("this is a CRITICAL log message")
    print("=======================================")


if __name__ == "__main__":
    # ======== 全局 logger ==========
    # step 1. print logger into console
    # test_basic()
    # test_level()
    # test_format()

    # step 2. logger into local file
    test_logfile()

    # ======== 自定义 logger ==========
    communicate_logger = logging.getLogger("communicate_logger")
    communicate_logger.error("this is log from [communicate_logger]")
    communicate_logger.setLevel(logging.WARNING)

    communicateFile_handler = logging.FileHandler("communicate_log.txt", mode="a")
    communicateFile_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s]-[%(name)s][%(levelname)s]-[%(process)d:%(processName)s][%(thread)d:%(threadName)s]-[%(message)s]-%(pathname)s"
        )
    )
    communicate_logger.addHandler(communicateFile_handler)
    communicate_logger.error("this is log from [communicate_logger]")
    communicate_logger.debug("this is log from [communicate_logger]")
    communicate_logger.info("this is log from [communicate_logger]")
    communicate_logger.warning("this is log from [communicate_logger]")
    communicate_logger.critical("this is log from [communicate_logger]")

    # step 3. logger 记录异常情况
    # ======== 自定义 logger ==========
    except_logger = logging.getLogger("except_logger")
    exceptFile_handler = logging.FileHandler("except_log.txt", mode="a")
    exceptFile_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s]-[%(name)s][%(levelname)s]-[%(process)d:%(processName)s][%(thread)d:%(threadName)s]-[%(message)s]-%(pathname)s"
        )
    )
    try:
        1 / 0
    except:
        except_logger.exception("Get exception")
