#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 3DScatter_figure.py
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
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# -------------------------
if __name__ == "__main__":
    # 设置风格
    sns.set_theme(style="white")

    # 数据准备
    x = np.random.standard_normal(100)
    y = np.random.standard_normal(100)
    z = np.random.standard_normal(100)

    # 创建3D散点图
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(x, y, z, c=z, cmap="viridis")

    # 添加装饰
    fig.colorbar(scatter)
    plt.title("3D Scatter Plot", fontsize=15)
    # plt.show()
    plt.tight_layout()
    plt.savefig("images/3DScatter_figure.png", dpi=600)
