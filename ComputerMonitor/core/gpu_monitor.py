#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: gpu_monitor.py
@Python Version: 3.12.1
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2025-08-11.
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V1.0
@License: Apache License Version 2.0, January 2004
    Copyright 2025. All rights reserved.

@Description: 
"""

import wmi


class GPUMonitor:
    def __init__(self):
        self.wmi = wmi.WMI()
    
    def get_gpu_utilization(self):
        """获取GPU利用率"""
        try:
            gpu_info = self.wmi.Win32_VideoController()[0]
            if hasattr(gpu_info, "LoadPercentage"):
                return gpu_info.LoadPercentage
        except Exception as e:
            print(f"获取GPU利用率失败: {e}")
        return 0
