#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: heatmap_figure.py
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


# --------------------------
if __name__ == "__main__":
    # 生成数据
    data = np.random.rand(10, 12)

    # 创建热图
    plt.figure(figsize=(10, 6))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="coolwarm")

    # 添加装饰
    plt.title("Heatmap", fontsize=15)
    # plt.show()
    plt.tight_layout()
    plt.savefig("images/heatmap_figure.png", dpi=600)
