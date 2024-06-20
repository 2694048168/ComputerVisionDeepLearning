#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: Facet_grid_figure.py
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

import seaborn as sns
import matplotlib.pyplot as plt


# -------------------------
if __name__ == "__main__":
    # 生成数据
    tips = sns.load_dataset("tips")

    # 创建图表
    g = sns.FacetGrid(tips, col="time", row="smoker", margin_titles=True)
    g.map(sns.scatterplot, "total_bill", "tip", alpha=0.7)
    g.add_legend()

    # 添加装饰
    plt.suptitle("Facet Grid", y=1.02, fontsize=15)
    # plt.show()
    plt.tight_layout()
    plt.savefig("images/Facet_grid_figure.png", dpi=600)
