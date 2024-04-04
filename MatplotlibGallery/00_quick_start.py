#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 00_quick_start.py
@Python Version: 3.12.1(from the Latest Miniconda)
@Author: Wei Li (Ithaca)
@Email: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/
@Date: 2024-04-03
@copyright Copyright (c) 2024 Wei Li
@Description: 快速使用 matplotlib 库

https://www.bilibili.com/video/BV1Pe4y1R79d/
'''

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# --------------------------
if __name__ == "__main__":
    X = np.linspace(0, 10 * np.pi, 1000)
    Y = np.cos(X)

    fig, ax = plt.subplots()
    ''' fmt = '[marker][line][color]' '''
    ax.plot(X, Y, color="red", marker="o", linestyle="solid", 
            linewidth=1, markersize=1)

    save_path = "./images/"
    os.makedirs(save_path, exist_ok=True)
    # filepath = save_path + "figure.pdf"
    filepath = save_path + "figure.png"
    # filepath = save_path + "figure.bmp" # error
    '''supported formats: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba,
    svg, svgz, tif, tiff, webp '''
    fig.savefig(filepath)
    plt.show()
