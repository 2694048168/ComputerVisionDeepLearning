#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: pair_figure.py
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


# --------------------------
if __name__ == "__main__":
    # 生成数据
    iris = sns.load_dataset("iris")

    # 创建图表
    sns.pairplot(iris, hue="species", palette="muted")
    plt.suptitle("Pair Plot", y=1.02, fontsize=15)
    # plt.show()
    plt.tight_layout()
    plt.savefig("images/pair_figure.png", dpi=600)
