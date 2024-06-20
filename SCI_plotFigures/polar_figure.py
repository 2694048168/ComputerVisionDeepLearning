#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: polar_figure.py
@Python Version: 3.12.1
@Platform: PyTorch 2.0.0 + cu117
@Author: Wei Li (Ithaca)
@Date: 2024-06-20.
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V2.1
@License: Apache License Version 2.0, January 2004
    Copyright 2024. All rights reserved.

@Description: 
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------
if __name__ == "__main__":
    # 设置风格
    sns.set_theme(style="white")

    # 数据准备
    theta = np.linspace(0, 2 * np.pi, 100)
    r = np.abs(np.sin(theta) * np.cos(theta))

    # 创建极坐标图
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.plot(theta, r, color="b", linewidth=2)

    # 添加装饰
    plt.title("Polar Plot", fontsize=15)
    # plt.show()
    plt.tight_layout()
    plt.savefig("images/polar_figure.png", dpi=600)
