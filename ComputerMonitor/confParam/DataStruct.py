#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: DataStruct.py
@Python Version: 3.12.1
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2025-08-15.
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V1.0
@License: Apache License Version 2.0, January 2004
    Copyright 2025. All rights reserved.

@Description: 数据结构
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class SystemParam:
    """自定义结构体类"""
    software_name: str = "ComputerMonitor" # 软件名称
    software_description: str = "The Computer Performance Monitor for Industrial Vision" # 软件简要描述
    major_version: int = 1 # 软件主版本号
    minor_version: int = 0 # 软件次版本号
    patch_version: int = 0 # 软件修订号
    timestamp: str = datetime.now().strftime("%Y%m%d-%H%M%S")
    software_version: str = f"{software_name}-V{major_version}.{minor_version}.{patch_version}-{timestamp}"
    author: str = "WeiLi(Ithaca)" # 开发者
    maintainer: str = "WeiLi(Ithaca)" # 维护者
    company: str = "WeiLi-Ithaca.com" # 公司
    copyright: str = "Copyright @ 2025 WeiLi(Ithaca). All Rights Reserved." # 版权
    # 使用 default_factory 初始化可变字段
    # tags: List[str] = field(default_factory=list)
    # 使用 default_factory 初始化字典
    # metadata: Dict[str, Union[str, int, float]] = field(default_factory=dict)

    def GetVersion(self) -> str:
        """软件版本信息"""
        self.software_version = f"{self.software_name}-V{self.major_version}.{self.minor_version}.{self.patch_version}-{self.timestamp}"
        return self.software_version
