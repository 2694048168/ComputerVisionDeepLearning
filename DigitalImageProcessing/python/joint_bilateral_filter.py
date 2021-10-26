#!/usr/bin/env python3
# encoding: utf-8


"""
@Funciton: 联合双边滤波
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
    closeWeight = np.exp(-0.5*(np.power(r, 2) +
                         np.power(c, 2)) / math.pow(sigma_g, 2))
    return closeWeight


def jointBilateralFilter(image, H, W, sigma_g, sigma_d, borderType=cv.BORDER_DEFAULT):
    """联合双边滤波

    Args:
        image (ndarray): 输入单通道图像, 灰度级范围[0，255]
        H ([int]): 权重模板的高
        W ([int]): 权重模板的宽
        sigma_g ([float]): 空间距离权重模板的标准差，大于 1
        sigma_d ([float]): 相似性权重模板的标准差， 小于 1

    Returns:
        [ndarray]: 联合双边滤波结果图像, 浮点型矩阵
    """
    center_H = (H - 1) // 2
    center_W = (W - 1) // 2
    rows, cols = image.shape
    # 空间距离权重
    closenessWeight = getClosenessWeight(sigma_g, H, W)
    # 对原图进行高斯平滑
    image_gauss = cv.GaussianBlur(image, (W, H), sigma_g)
    # 对原图和高斯平滑结果进行边界扩充
    image_padding = cv.copyMakeBorder(
        image, center_H, center_H, center_W, center_W, borderType)
    image_gauss_padding = cv.copyMakeBorder(
        image_gauss, center_H, center_H, center_W, center_W, borderType)

    joint_bilateral_filter_image = np.zeros(image.shape, np.float32)
    i, j = 0, 0
    for r in range(center_H, center_H+rows, 1):
        for c in range(center_W, center_W+cols, 1):
            pixel = image_gauss_padding[r][c]
            # 当前位置的邻域
            rTop, rBottom = r - center_H, r + center_H
            cLeft, cRight = c - center_W, c + center_W
            # 从扩充的高斯平滑图像中截取该邻域，用于构建相似性权重
            region = image_gauss_padding[rTop:rBottom+1, cLeft:cRight+1]
            # 构建灰度值相似性的权重因子
            similarityWeight = np.exp(-0.5*np.power(region - pixel, 2.0)/math.pow(sigma_d, 2.0))
            # 两个权重模板点乘
            weight = closenessWeight * similarityWeight
            # 归一化权重模板
            weight = weight / np.sum(weight)
            # 权重模板和对应的邻域值相乘求和
            joint_bilateral_filter_image[i][j] = np.sum(
                image_padding[rTop:rBottom+1, cLeft:cRight+1] * weight)
            j += 1
        j = 0
        i += 1

    return joint_bilateral_filter_image


# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        print(image.shape)
        if image is None:
            print(f"Error: no such file or dictory.")
        cv.imshow("OriginImage", image)

        # convert data type uint8 to float64
        image = image.astype(np.float64)
        # 1. 联合双边滤波
        joint_bilateral_filter_image = jointBilateralFilter(image, 5, 5, 7, 2)
        # convert data type to show or save
        joint_bilateral_filter_image = np.round(joint_bilateral_filter_image).astype(np.uint8)
        cv.imshow("joint_bilateral_filter", joint_bilateral_filter_image)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print(f"Usage: python histogram imageFile.")
