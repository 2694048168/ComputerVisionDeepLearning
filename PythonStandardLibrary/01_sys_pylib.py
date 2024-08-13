#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 01_sys_pylib.py
@Python Version: 3.12.1
@Platform: PyTorch 2.0.0 + cu117
@Author: Wei Li (Ithaca)
@Date: 2024-08-13.
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V2.1
@License: Apache License Version 2.0, January 2004
    Copyright 2024. All rights reserved.

@Description: 
"""

# Python 标准库 SYS
# 1. 针对 Python解释器相关的变量和方法
import sys


if __name__ == "__main__":
    # 1. 针对 Python解释器相关的变量和方法
    print(f"the current Python interpreter version: {sys.version}")

    # 当前操作系统能够表示的最大int(32bit/64bit)
    print(f"the current max-size for integer: {sys.maxsize}")

    # 当前 Python 模块搜索路径
    print(f"the current Path-search for interpreter: {sys.path}")

    # 依赖当前 Python 解释器
    print(f"the current platform: {sys.platform}")
    print(f"the Python copyright:\n {sys.copyright}")

    # 命令行参数列表
    print(f"the command-argv: {sys.argv}")

    # 默认编码
    print(f"the default encoding: {sys.getdefaultencoding()}")
    print(f"the default filesystem encoding: {sys.getfilesystemencoding()}")

    # 函数递归调用/堆栈次数限制
    print(f"the function recursive limit: {sys.getrecursionlimit()}")
    sys.setrecursionlimit(200)
    print(f"the function recursive limit: {sys.getrecursionlimit()}")

    # 程序退出状态
    # sys.exit(-1)
    sys.exit(0)
