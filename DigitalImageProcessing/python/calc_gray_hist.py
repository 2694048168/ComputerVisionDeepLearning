#!/usr/bin/env python3
# encoding: utf-8


"""
@Funciton: 计算灰度直方图(归一化直方图，概率直方图)
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-09-15
"""

import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def calcGrayHist(image):
    """计算图像的灰度直方图。

    Args:
        image ([type]): 单通道的灰度图像,图像深度为 8 bit

    Returns:
        一维 ndarray : 灰度图像的直方图，每一个灰度级对应的像素个数
    """
    rows, cols = image.shape
    grayHist = np.zeros([256], np.uint64)
    for idx_row in range(rows):
        for idx_col in range(cols):
            grayHist[image[idx_row][idx_col]] += 1
    
    return grayHist


# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or dictory.")
        
        # 1. 手动计算直方图        
        grayHist = calcGrayHist(image)
        # 可视化灰度直方图
        x_range = range(256)
        plt.plot(x_range, grayHist, linewidth=2, color="black")
        y_maxValue = np.max(grayHist)
        plt.axis([0, 255, 0, y_maxValue])
        plt.xlabel("Gray Level")
        plt.ylabel("Number of Pixels")
        plt.show()
        plt.close()

        # 2. 利用 matplotlib 计算直方图
        rows, cols = image.shape
        # 二维矩阵转换为一维数组
        pixelSequence = image.reshape([rows*cols, ])
        numberBins = 256 # 灰度等级
        histgram, bins, patch = plt.hist(pixelSequence, numberBins, facecolor="black", histtype="bar")
        plt.xlabel(u"GrayLevel")
        plt.ylabel(u"Number of Pixels")
        plt.axis([0, 255, 0, np.max(histgram)])
        plt.show()
        plt.close()

    else:
        print(f"Usage: python histogram imageFile.")
    