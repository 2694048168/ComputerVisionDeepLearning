#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: Marr-Hildreth 边缘检测 基于 高斯差分算子核高斯拉普拉斯算子
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


def zero_cross_default(imgDiffGuass):
    zero_cross = np.zeros(imgDiffGuass.shape, np.uint8)
    rows, cols = imgDiffGuass.shape
    for row_idx in range(1, rows - 1):
        for col_idx in range(1, cols - 1):
            # 左/右方向
            if imgDiffGuass[row_idx][col_idx -1]*imgDiffGuass[row_idx][col_idx + 1] < 0:
                zero_cross[row_idx][col_idx] = 255
                continue

            # 上/下方向
            if imgDiffGuass[row_idx -1][col_idx]*imgDiffGuass[row_idx+1][col_idx] < 0:
                zero_cross[row_idx][col_idx] = 255
                continue

            # 左上/右下方向
            if imgDiffGuass[row_idx -1][col_idx-1]*imgDiffGuass[row_idx+1][col_idx+1] < 0:
                zero_cross[row_idx][col_idx] = 255
                continue

            # 右上/左下方向
            if imgDiffGuass[row_idx -1][col_idx+1]*imgDiffGuass[row_idx+1][col_idx-1] < 0:
                zero_cross[row_idx][col_idx] = 255
                continue

    return zero_cross


def zero_cross_mean(imgDiffGuass):
    zero_cross = np.zeros(imgDiffGuass.shape, np.uint8)
    rows, cols = imgDiffGuass.shape
    # 存储左上、右上、左下、右下方向的均值
    fourMean = np.zeros(4, np.float32)
    for row_idx in range(1, rows - 1):
        for col_idx in range(1, cols - 1):
            # 左上方向的均值
            leftTopMean = np.mean(imgDiffGuass[row_idx-1:row_idx+1, col_idx-1:col_idx+1])
            fourMean[0] = leftTopMean

            # 右上方向的均值
            rightTopMean = np.mean(imgDiffGuass[row_idx-1:row_idx+1, col_idx:col_idx+2])
            fourMean[1] = rightTopMean

            # 左下方向的均值
            leftBottomMean = np.mean(imgDiffGuass[row_idx:row_idx+2, col_idx-1:col_idx+1])
            fourMean[2] = leftBottomMean

            # 右下方向的均值
            rightBottomMean = np.mean(imgDiffGuass[row_idx:row_idx+2, col_idx:col_idx+2])
            fourMean[3] = rightBottomMean

            if (np.min(fourMean)*np.max(fourMean) < 0):
                zero_cross[row_idx][col_idx] = 255

    return zero_cross


def Marr_Hildreth(image, size, sigma, k=1.1, crossType="ZERO_CROSS_DEFAULT"):
    # 高斯差分
    imgDiffGauss = DiffGuassian(image, size, sigma, k)

    # 过零点
    if crossType == "ZERO_CROSS_DEFAULT":
        zero_cross = zero_cross_default(imgDiffGauss)
    elif crossType == "ZERO_CROSS_MEAN":
        zero_cross = zero_cross_mean(imgDiffGauss)
    else:
        print("Not implementation")

    return zero_cross
            

# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or dictory.")
        cv.imshow("OriginImage",image)

        # -------- Marr-Hildreth 边缘检测 --------
        edge = Marr_Hildreth(image, (7, 7), 1, 1.1, "ZERO_CROSS_DEFAULT")
        cv.imshow("Marr-Hildreth_Edge", edge)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print("Usge: python.py imageFile")
