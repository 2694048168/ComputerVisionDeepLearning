#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 高斯拉普拉斯算子 —— 先二维高斯平滑处理，后进行拉普拉斯微分算子  —— 可分离高斯拉普拉斯卷积核
            拉普拉斯算子对噪声很敏感，使用首先应对图像进行高斯平滑，然后再与拉普拉斯算子卷积，最后得到二值化边缘图。
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-10-17
"""

import sys
from scipy import signal
import numpy as np
import cv2 as cv


def createLaplaceGaussianKernel(sigma, size):
    """构建高斯拉普拉斯卷积核

    Args:
        sigma ([float]): 高斯函数的标准差
        size ([tuple]): 高斯核的大小，奇数

    Returns:
        [ndarray]: 高斯拉普拉斯卷积核
    """
    H, W = size
    r, c = np.mgrid[0:H:1, 0:W:1]
    r = r - (H - 1) / 2
    c = c - (W - 1) / 2

    sigma2 = pow(sigma, 2.0)
    norm2 = np.power(r, 2.0) + np.power(c, 2.0)
    LoGKernel = (norm2 / sigma2 - 2)*np.exp(-norm2 / (2 * sigma2))

    return LoGKernel


def LaplaceGaussianOperator(image, sigma, size, _boundary="symm", _fillvalue=0):
    # Laplace of Guassian convolution kernel
    laplace_gaussian_kernel = createLaplaceGaussianKernel(sigma=sigma, size=size)

    img_laplace_gaussian_conv = signal.convolve2d(image, laplace_gaussian_kernel, mode="same", boundary=_boundary, fillvalue=_fillvalue)

    return img_laplace_gaussian_conv


# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or dictory.")
        cv.imshow("OriginImage",image)

        # -------- Laplace of Guassian Operator --------
        img_laplce_gaussian_conv = LaplaceGaussianOperator(image, 1, (7, 7))

        # 阈值化处理获取二值图
        edge_binary = np.copy(img_laplce_gaussian_conv)
        edge_binary[edge_binary>0] = 255
        edge_binary[edge_binary<=0] = 0
        edge_binary = edge_binary.astype(np.uint8)
        cv.imshow("EdgeBinary", edge_binary)

        # 反色处理，以黑色显示边缘
        edge_black_binary = 255 - edge_binary
        cv.imshow("EdgeBinaryBlack", edge_black_binary)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print("Usge: python.py imageFile")
