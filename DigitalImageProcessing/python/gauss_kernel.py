#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 计算高斯卷积算子；高斯卷积核的可分离；高斯卷积核进行图像平滑(模糊)
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-09-17
"""

import math
import sys
from scipy import signal
import numpy as np
import cv2 as cv


def getGaussKernel(sigma, H, W):
    """计算高斯卷积算子。

    Args:
        sigma (float): 高斯分布的标准差 sigma
        H (int): 高斯核的大小，奇数
        W (int): 高斯核的大小，奇数

    Returns:
        [ndarray]: 高斯卷积核算子
    """
    # 1. 构建高斯矩阵
    # -------------------------
    gaussMatrix = np.zeros([H, W], np.float32)
    # 计算中心点
    centerH = (H - 1) / 2
    centerW = (W - 1) / 2
    for idx_row in range(H):
        for idx_col in range(W):
            norm2 = math.pow(idx_row - centerH, 2) + math.pow(idx_col - centerW, 2)
            gaussMatrix[idx_row][idx_col] = math.exp(-norm2 / (2 *math.pow(sigma, 2)))
    # 利用 Numpy 进行简化
    # gaussMatrix = np.exp(-0.5 * (np.power(idx_row) + np.power(idx_col)) / math.pow(sigma, 2))
    # -------------------------

    # 2. 计算高斯矩阵的和
    sumGM = np.sum(gaussMatrix)
    # 3. 归一化高斯矩阵， 即得到高斯卷积算子
    gaussKernel = gaussMatrix / sumGM

    return gaussKernel


def gaussBlur(image, sigma, H, W, mode, _boudary="fill", _fillvalue=0):
    # 水平方向高斯卷积核
    gaussKernel_x = cv.getGaussianKernel(sigma, W, cv.CV_64F)
    # 转置
    gaussKernel_x = np.transpose(gaussKernel_x)
    # 图像矩阵与水平高斯核卷积
    gaussBlur_x = signal.convolve2d(image, gaussKernel_x, mode=mode, boundary=_boudary, fillvalue=_fillvalue)
    # 构建垂直方向的高斯卷积核
    gaussKernel_y = cv.getGaussianKernel(sigma, H, cv.CV_64F)
    # 与垂直方向上的高斯核卷积
    gaussKernel_xy = signal.convolve2d(gaussBlur_x, gaussKernel_y, mode=mode, boundary=_boudary, fillvalue=_fillvalue)

    return gaussKernel_xy

    
# --------------------------
if __name__ == "__main__":
    # 1. 计算高斯卷积算子
    sigma = 1.2
    H, W = 3, 3
    gaussKernel = getGaussKernel(sigma=sigma, H=H, W=W)
    print(f"The gauss kernel of 3x3 is:\n{gaussKernel}")

    # 2. 高斯卷积算子是可分离的，只需要一个方向(垂直或者水平)的 API即可
    gauss_kernel_v = cv.getGaussianKernel(3, 1.2, cv.CV_64F)  # 垂直方向
    print(f"The vertical direction of Gaussian Kernel:\n{gauss_kernel_v}")
    # 对垂直方向的 kernel 进行转置
    print(f"The horizontal direction of Gaussian Kernel:\n{gauss_kernel_v.T}")
    # 验证一下，高斯卷积算子是可分离的
    
    # 3. 利用 高斯卷积核进行图像平滑
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or directory.")
            sys.exit()
        cv.imshow("OriginImage", image)
        blurImage = gaussBlur(image, 5, 51, 51, "same")
        # 对 blurImage 进行灰度级显示，float to uint8
        blurImage = (np.round(blurImage)).astype(np.uint8)
        cv.imshow("GaussBlurImage", blurImage)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print(f"Usage: python gaussBlur imageFile.")
