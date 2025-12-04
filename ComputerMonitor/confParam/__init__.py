#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: __init__.py
@Python Version: 3.12.1
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2025-08-15.
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V1.0
@License: Apache License Version 2.0, January 2004
    Copyright 2025. All rights reserved.

@Description: Python __init__.py 文件是将目录转换为 Python 包的关键
"""

# 可选：暴露公共接口
from .DataStruct import SystemParam
from .SystemParamConf import SystemParamConf

__all__ = ['SystemParam', 'SystemParamConf']
