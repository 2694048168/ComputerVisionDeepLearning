#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: contour_figure.py
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


# ---------------------------
if __name__ == "__main__":
    # 设置风格
    sns.set_theme(style="white")

    # 数据准备
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    # 创建等高线图
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(X, Y, Z, cmap="coolwarm", levels=20)
    plt.colorbar(contour)

    # 添加装饰
    plt.title("Contour Plot", fontsize=15)
    # plt.show()
    plt.tight_layout()
    plt.savefig("images/contour_figure.png", dpi=600)
