#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 06_imageHistogram.py
@Python Version: 3.12.1
@Author: Wei Li (Ithaca)
@Email: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/
@Date: 2024-03-07
@copyright Copyright (c) 2024 Wei Li
@Description: 利用 OpenCV 和 Numpy 库进行图像直方图统计和显示
'''

import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def calcGrayHist(image):
    """计算图像的灰度直方图

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


def drawColorHist(image_filepath):
    """计算彩色图像的BGR三个通道的直方图
        并使用 matplotlib 绘图库进行绘制统计的结果

    Args:
        image_filepath (_type_): 图像文件路径
    """
    image = cv.imread(image_filepath, flags=cv.IMREAD_UNCHANGED)
    colors = ['blue','green','red'] # OpenCV color: BGR
    for idx in range(len(colors)):
        hist_total, hist_edges = np.histogram(image[:,:,idx].ravel(), bins=256,range=(0,256))
        plt.plot(0.5*(hist_edges[:-1]+hist_edges[1:]),hist_total,label=colors[idx],color=colors[idx])
    plt.show()


# --------------------------
if __name__ == "__main__":
    if len(sys.argv) < 1:
        print(f"Error: Please enter the image filepath.")
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

    # 彩色图像的直方图统计
    filepath = sys.argv[1]
    drawColorHist(filepath)
