#!/usr/bin/env python3
# encoding: utf-8


"""
@Funciton: 双边滤波; 非常耗时，提出双边滤波的快速算法
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-09-25
"""

import sys
import math
import cv2 as cv
import numpy as np


def getClosenessWeight(sigma_g, H, W):
    """构建空间权重模板"""
    r, c = np.mgrid[0:H:1, 0:W:1]
    r -= (H - 1) // 2
    c -= (W - 1) // 2
    closeWeight = np.exp(-0.5*(np.power(r, 2) + np.power(c, 2)) / math.pow(sigma_g, 2))
    return closeWeight


def bilateralFilterGray(image, H, W, sigma_g, sigma_d):
    """双边滤波

    Args:
        image (ndarray): 输入单通道图像, 灰度级范围[0，1]
        H ([int]): 权重模板的高
        W ([int]): 权重模板的宽
        sigma_g ([float]): 空间距离权重模板的标准差，大于 1
        sigma_d ([float]): 相似性权重模板的标准差， 小于 1

    Returns:
        [ndarray]: 双边滤波结果图像, 浮点型矩阵
    """
    closenessWeight = getClosenessWeight(sigma_g, H, W)
    center_H = (H - 1) // 2
    center_W = (W - 1) // 2
    rows, cols = image.shape
    bilateral_filter_gray_image = np.zeros(image.shape, np.float32)
    for r in range(rows):
        for c in range(cols):
            pixel = image[r][c]
            # 边界判断
            rTop = 0 if r-center_H < 0 else r-center_H
            rBottom = rows-1 if r+center_H > rows-1 else r+center_H
            cLeft = 0 if c-center_W < 0 else c-center_W
            cRight = cols-1 if c+center_W > cols-1 else c+center_W
            # 权重模板作用区域
            region = image[rTop:rBottom+1, cLeft:cRight+1]
            # 构建灰度值相似性的权重因子
            similarityWeightTemp = np.exp(-0.5*np.power(region-pixel, 2.0)/math.pow(sigma_d, 2))
            closenessWeightTemp = closenessWeight[rTop-r+center_H:rBottom-r+center_H+1, cLeft-c+center_W:cRight-c+center_W+1]
            # 两个权重模板点乘
            weightTemp = similarityWeightTemp * closenessWeightTemp
            # 归一化权重模板
            weightTemp = weightTemp / np.sum(weightTemp)
            # 权重模板和对应的邻域值相乘求和
            bilateral_filter_gray_image[r][c] = np.sum(region * weightTemp)

    return bilateral_filter_gray_image
    

# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or dictory.")
        cv.imshow("OriginImage", image)

        # [0-255] ----> [0-1]
        image = image/255.0

        # 1. 双边滤波
        bilateral_filter_gray_image = bilateralFilterGray(image, 33, 33, 19, 0.2)
        bilateral_filter_gray_image = (bilateral_filter_gray_image * 255).astype(np.uint8)
        cv.imshow("bilateral_filter", bilateral_filter_gray_image)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print(f"Usage: python histogram imageFile.")
