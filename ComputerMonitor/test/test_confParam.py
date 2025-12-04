#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: test_confParam.py
@Python Version: 3.12.1
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2025-08-15.
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V1.0
@License: Apache License Version 2.0, January 2004
    Copyright 2025. All rights reserved.

@Description: 测试模块
"""

import os
import sys
# 将项目根目录添加到模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from confParam.SystemParamConf import SystemParamConf


# ---------------------------
if __name__ == "__main__":
    systemConf = SystemParamConf()
    print(f"The name of Software: {systemConf.params.software_name}")
    print(f"The Version of Software: {systemConf.params.software_version}")

    filepath = "./config/system_param.json"
    # systemConf.SerializeToFile(filepath)
    
    systemConf.DeserializeFromFile(filepath)
    print(f"The Version of Software: {systemConf.params.GetVersion()}")
