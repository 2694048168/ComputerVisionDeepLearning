#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: twin_figure.py
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

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ----------------------------
if __name__ == "__main__":
    # 设置风格
    sns.set_theme(style="whitegrid")

    # 生成数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # 创建图表
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax2 = ax1.twinx()
    ax1.plot(x, y1, "g-")
    ax2.plot(x, y2, "b-")

    # 添加装饰
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Sine", color="g")
    ax2.set_ylabel("Cosine", color="b")
    plt.title("Dual Axis Plot")
    # plt.show()
    plt.tight_layout()
    plt.savefig("images/twin_figure.png", dpi=600)
