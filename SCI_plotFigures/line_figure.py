#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: line_figure.py
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


# -------------------------
if __name__ == "__main__":
    # 设置风格
    sns.set_theme(style="whitegrid")

    # 生成数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, label="Sine Wave", color="b", linewidth=2)
    plt.plot(x, y2, label="Cosine Wave", color="r", linestyle="--", linewidth=2)

    # 添加装饰
    plt.fill_between(x, y1, y2, color="gray", alpha=0.1)
    plt.title("Line Plot", fontsize=15)
    plt.xlabel("X-axis", fontsize=12)
    plt.ylabel("Y-axis", fontsize=12)
    plt.legend()
    # plt.show()
    plt.tight_layout()
    plt.savefig("images/line_figure.png", dpi=600)
