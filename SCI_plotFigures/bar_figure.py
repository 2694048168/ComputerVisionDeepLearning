#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: bar_figure.py
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


# ---------------------------
if __name__ == "__main__":
    # 设置风格
    sns.set_theme(style="whitegrid")

    # 生成数据
    categories = ["A", "B", "C", "D"]
    values1 = [5, 7, 8, 6]
    values2 = [3, 4, 5, 2]

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    bar1 = ax.bar(categories, values1, label="Group 1")
    bar2 = ax.bar(categories, values2, bottom=values1, label="Group 2")

    # 添加装饰
    ax.set_title("Stacked Bar Chart", fontsize=15)
    ax.set_xlabel("Categories", fontsize=12)
    ax.set_ylabel("Values", fontsize=12)
    ax.legend()

    # 添加数值标签
    for rect in bar1 + bar2:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    # plt.show()
    plt.tight_layout()
    plt.savefig("images/bar_figure.png", dpi=600)
