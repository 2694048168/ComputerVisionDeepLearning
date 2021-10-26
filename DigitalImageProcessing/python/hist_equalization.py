#!/usr/bin/env python3
# encoding: utf-8


"""
@Funciton: 灰度图像的直方图均衡化(全局直方图均衡化)
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-09-16
"""

import sys
import math
import cv2 as cv
import numpy as np


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


def equalizeHist(image):
    """全局直方图均衡化。

    Args:
        image (ndarray): 矩阵形式的输入图像

    Returns:
        [ndarray]: 矩阵形式的经过直方图均衡化后的输出图像
    """
    # 对于直方图均衡化的实现主要分四个步骤:
    # 1. 计算图像的灰度直方图。
    rows, cols = image.shape
    grayHist = calcGrayHist(image)
    # 2. 计算灰度直方图的累加直方图。
    zeroCumuMoment = np.zeros([256], np.uint32)
    for p in range(256):
        if p == 0:
            zeroCumuMoment[p] = grayHist[0]
        else:
            zeroCumuMoment[p] = zeroCumuMoment[p-1] + grayHist[p]
    # 3. 根据累加直方图和直方图均衡化原理得到输入灰度级和输出灰度级之间的映射关系。
    output_q = np.zeros([256], np.uint8)
    cofficient = 256.0 / (rows * cols)
    for p in range(256):
        q = cofficient * float(zeroCumuMoment[p]) - 1
        if q >= 0:
            output_q[p] = math.floor(q)
        else:
            output_q[p] = 0
    # 4. 根据第三步得到的灰度级映射关系，循环得到输出图像的每一个像素的灰度级。
    equalizeHistImage = np.zeros(image.shape, np.uint8)
    for idx_row in range(rows):
        for idx_col in range(cols):
            equalizeHistImage[idx_row][idx_col] = output_q[image[idx_row][idx_col]]

    return equalizeHistImage


# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or dictory.")
        cv.imshow("OriginImage", image)

        # 1. 直方图均衡化
        equalHistImg = equalizeHist(image)
        cv.imshow("EqualizeHistImage", equalHistImg)

        # 2. 自适应直方图均衡化 ----> 限制对比度的自适应直方图均衡化
        # 构建 CLAHE 对象
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8));
        #  限制对比度的自适应直方图均衡化
        dst_contrastLimit = clahe.apply(image)
        cv.imshow("ContrastLimitImage", dst_contrastLimit)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print(f"Usage: python histogram imageFile.")
