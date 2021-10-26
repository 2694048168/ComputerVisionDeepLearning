#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 高斯差分边缘检测(接近高斯拉普拉斯算子) —— 计算量减少
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-10-17
"""

import sys
from scipy import signal
import numpy as np
import cv2 as cv


def gaussConv(image, size, sigma):
    """函数 gaussConv 实现非归一化的高斯卷积

    Args:
        image ([ndarray]): [description]
        size ([tuple]): 卷积核的大小，二元元组，(高，宽)
        sigma ([float]): 高斯函数的标准差

    Returns:
        [ndarray]: 高斯卷积结果
    """
    H, W = size
    # 构建水平方向上非归一化的高斯卷积核
    _, x_col = np.mgrid[0:1, 0:W]
    x_col = x_col - (W - 1) / 2
    x_kernel = np.exp(-np.power(x_col, 2.0))
    img_xk = signal.convolve2d(image, x_kernel, "same", "symm", 0)

    # 构造垂直方向非归一化的高斯卷积核
    y_row, _ = np.mgrid[0:H, 0:1]
    y_row = y_row - (H - 1) / 2
    y_kernel = np.exp(-np.power(y_row, 2.0))
    img_xk_yk = signal.convolve2d(img_xk, y_kernel, "same", "symm", 0)
    
    img_xk_yk = img_xk_yk * 1.0/(2 * np.pi * pow(sigma, 2.0))

    return img_xk_yk


def DiffGuassian(image, size, sigma, k=1.1):
    # 标准差为 sigma 的非归一化高斯卷积核
    img_gauss_kernel_1 = gaussConv(image, size, sigma)

    # 标准差为 k*sigma 的非归一化高斯卷积核
    img_gauss_kernel_k = gaussConv(image, size, k*sigma)

    # 两个高斯卷积的差分
    diff_guass = img_gauss_kernel_k - img_gauss_kernel_1
    diff_guass = diff_guass / (pow(sigma, 2.0)*(k - 1))

    return diff_guass


# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or dictory.")
        cv.imshow("OriginImage",image)

        # -------- Difference of Guassian Operator --------
        sigma = 0.2
        k = 1.1
        size = (3, 3)
        img_diff_gauss = DiffGuassian(image, size, sigma, k)

        # 1. 二值化处理
        edge = np.copy(img_diff_gauss)
        edge[edge>0] = 255
        edge[edge<=0] = 0
        edge = edge.astype(np.uint8)
        cv.imshow("edge_binary", edge)

        # 2. 抽象化处理
        asbstraction_img = -np.copy(img_diff_gauss)
        asbstraction_img = asbstraction_img.astype(np.float32)
        asbstraction_img[asbstraction_img>=0] = 1.0
        asbstraction_img[asbstraction_img<0] = 1.0 + np.tanh(asbstraction_img[asbstraction_img<0])
        cv.imshow("abstraction_edge", asbstraction_img)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print("Usge: python.py imageFile")
