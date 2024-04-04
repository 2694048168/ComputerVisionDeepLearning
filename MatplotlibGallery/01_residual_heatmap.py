#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 01_residual_heatmap.py
@Python Version: 3.12.1(from the Latest Miniconda)
@Author: Wei Li (Ithaca)
@Email: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/
@Date: 2024-04-03
@copyright Copyright (c) 2024 Wei Li
@Description: 数字图像处理中重建图像的主观结果对比, 残差图 + 热力颜色增强表示 
'''

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
#从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
import cv2 as cv


# --------------------------
if __name__ == "__main__":
    filepath = "./images/butterfly_HR_x4.png"
    img = cv.imread(filepath, flags=0)

    img_blur = cv.GaussianBlur(img, (5, 5), sigmaX=0.5)

    plt.imshow(img - img_blur)
    plt.xlim(0, 255)
    plt.ylim(0, 255)
    # plt.gca().xaxis.set_major_locator(MultipleLocator(10))
    # plt.gca().yaxis.set_major_locator(MultipleLocator(10))
    cax = plt.axes((0.85, 0.1, 0.075, 0.8))
    plt.colorbar(cax=cax)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    
    save_path = "./images/"
    os.makedirs(save_path, exist_ok=True)
    filepath = save_path + "figure_heatmap.png"
    plt.savefig(filepath)
    plt.show()
