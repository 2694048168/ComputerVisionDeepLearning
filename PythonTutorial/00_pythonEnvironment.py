#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 00_pythonEnvironment.py
@Python Version: 3.12.1(from the Latest Miniconda)
@Author: Wei Li (Ithaca)
@Email: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/
@Date: 2024-03-01
@copyright Copyright (c) 2024 Wei Li
@Description: 测试本地主机的 Python 环境配置
@Doc: https://2694048168.github.io/blog/#/PaperMD/python_env_ai
'''

import sys


# --------------------------
if __name__ == "__main__":
    # 使用命令行: python --version
    print("======== Python 版本信息 ========")
    print(sys.version)
    print("======== Python 主次版本信息 ========")
    print(sys.version_info)
