#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: spider_figure.py
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
    sns.set_theme(style="whitegrid")

    # 数据准备
    labels = np.array(["A", "B", "C", "D", "E"])
    stats = [10, 20, 30, 40, 50]
    stats2 = [30, 10, 20, 30, 40]

    # 创建蜘蛛图
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    stats = np.concatenate((stats, [stats[0]]))
    stats2 = np.concatenate((stats2, [stats2[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, stats, color="blue", alpha=0.25, label="Group 1")
    ax.plot(angles, stats, color="blue", linewidth=2)
    ax.fill(angles, stats2, color="red", alpha=0.25, label="Group 2")
    ax.plot(angles, stats2, color="red", linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.grid(True)

    # 添加标题和图例
    plt.title("Enhanced Spider Chart", size=20, color="black", y=1.1)
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    # plt.show()
    plt.tight_layout()
    plt.savefig("images/spider_figure.png", dpi=600)
