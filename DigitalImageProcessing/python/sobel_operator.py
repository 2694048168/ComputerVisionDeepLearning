#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: Sobel 算子进行边缘检测 —— 可分离卷积核
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-10-14
"""

import sys
import math
from scipy import signal
import numpy as np
import cv2 as cv


def PascalSmooth(n):
    """函数 PascalSmooth 返回 n 阶的非归一化的高斯平滑算子，
    即指数为 n-1 的二项式展开式的系数，
    其中对于阶乘的实现，利用 Python 的函数包 math 中的 factorial，其参数 n 为奇数。

    Args:
        n ([int]): 高斯卷积算子的阶数(奇数)

    Returns:
        [array]: 高斯卷积算子中的系数，即用于 Soble 算子中平滑核参数
    """
    pascalSmooth = np.zeros([1, n], np.float32)
    for idx in range(n):
        pascalSmooth[0][idx] = math.factorial(n - 1) / math.factorial(idx) * math.factorial(n - 1 - idx)

    return pascalSmooth


def PascalDiff(n):
    """函数 PascalDiff 返回 n 阶差分算子，完成 Sobel 在方向上的差分操作

    Args:
        n ([int]): Sobel 进行 n 阶差分

    Returns:
        [array]: Soble n 阶差分结果
    """
    pascalDiff = np.zeros([1, n], np.float32)
    pascalSmooth_previous = PascalSmooth(n - 1)
    for idx in range(n):
        if idx == 0:
            # 恒等于 1
            pascalDiff[0][idx] = pascalSmooth_previous[0][idx]
        elif idx == n - 1:
            # 恒等于 -1
            pascalDiff[0][idx] = -pascalSmooth_previous[0][idx - 1]
        else:
            pascalDiff[0][idx] = pascalSmooth_previous[0][idx] - pascalSmooth_previous[0][idx - 1]

    return pascalDiff


def GetSobelKernel(n):
    """ PascalSmooth 返回的平滑算子和 PascalDiff 返回的差分算子进行 full 卷积，
    就可以得到完整的水平方向和垂直方向上的 nxn 的 Sobel 算子。
    注意，真正在进行 Sobel 卷积时，这一步是多余的，直接通过卷积的分离性就可以完成 Sobel 卷积,
    这里只是为了得到完整的 Sobel 核,通过定义函数 getSobelKernel 来实现, 返回值包括水平方向和垂直方向上的 Sobel 核。

    Args:
        n ([int]): n 阶 Sobel 算子

    Returns:
        [array]: 水平方向和垂直方向的 Sobel 卷积核
    """
    pascalSmoothKernel = PascalSmooth(n)
    pascalDiffKernel = PascalDiff(n)

    # 水平方向上卷积核
    sobelKernel_x = signal.convolve2d(pascalSmoothKernel.transpose(), pascalDiffKernel, mode="full")

    # 垂直方向上卷积核
    sobelKernel_y = signal.convolve2d(pascalSmoothKernel, pascalDiffKernel.transpose(), mode="full")

    return (sobelKernel_x, sobelKernel_y)


def SobelOperator(image, n):
    """ 构建了 Sobel 平滑算子和差分算子后，通过这两个算子来完成图像矩阵与 Sobel 算子的 same 卷积，
    函数 SobelOperator 实现该功能: 
        图像矩阵先与垂直方向上的平滑算子卷积得到的卷积结果，
        再与水平方向上的差分算子卷积，
        这样就得到了图像矩阵与sobel_x 核的卷积。
        与该过程类似,图像矩阵先与水平方向上的平滑算子卷积得到的卷积结果,
        再与垂直方向上的差分算子卷积,
        这样就得到了图像矩阵与 sobel_y 核的卷积。

    Args:
        image ([ndarray]): 进行 Sobel 算子的原始输入图像
        n ([int]): 进行 Sobel 算子的阶数

    Returns:
        [ndarray]: 水平方向上的 Sobel 卷积结果；垂直方向上的卷积结果
    """
    pascalSmoothKernel = PascalSmooth(n)
    pascalDiffKernel = PascalDiff(n)

    # -------- 与水平方向上 Sobel 卷积核进行卷积 --------
    # 可分离卷积核 1. 先进行垂直方向的平滑
    img_sobel_x = signal.convolve2d(image, pascalSmoothKernel.transpose(), mode="same")
    # 可分离卷积核 2. 再进行水平方向的差分
    img_sobel_x = signal.convolve2d(img_sobel_x, pascalDiffKernel, mode="same")

    # -------- 与水平方向上 Sobel 卷积核进行卷积 --------
    # 可分离卷积核 1. 先进行垂直方向的平滑
    img_sobel_y = signal.convolve2d(image, pascalSmoothKernel, mode="same")
    # 可分离卷积核 2. 再进行水平方向的差分
    img_sobel_y = signal.convolve2d(img_sobel_x, pascalDiffKernel.transpose(), mode="same")

    return img_sobel_x, img_sobel_y


# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or dictory.")
        cv.imshow("OriginImage",image)

        # 注意区分边缘强度和边缘强度的灰度级显示
        img_soble_x, img_soble_y = SobelOperator(image, 3)

        # 计算绝对值，获取水平方向和垂直方向的边缘强度
        abs_img_soble_x = np.abs(img_soble_x)
        abs_img_soble_y = np.abs(img_soble_y)

        # 水平方向和垂直方向的边缘强度的灰度级显示
        edge_x = abs_img_soble_x.copy()
        edge_y = abs_img_soble_y.copy()
        # 将大于 255 直接进行饱和操作
        edge_x[edge_x>255] = 255
        edge_y[edge_y>255] = 255
        # 数据类型转换
        edge_x = edge_x.astype(np.uint8)
        edge_y = edge_y.astype(np.uint8)
        cv.imshow("edge_x", edge_x)
        cv.imshow("edge_y", edge_y)

        # 根据 sobel 两个卷积结果, 计算最终的边缘强度
        # 计算最终的边缘强度有多种方式, 采用 平方和开方方式
        edge = np.sqrt(np.power(img_soble_x, 2.0) + np.power(img_soble_y, 2.0))

        # 边缘轻度的灰度级显示
        # Sobel 边缘检测，将边缘强度大于 255 的值直接截断为 255，这样得到的边缘有可能不够平滑
        # edge[edge>255] = 255
        # edge = edge.astype(np.uint8)
        # cv.imshow("edge_255", edge)

        # 另一种方式，对所得到的边缘强度进行直方图正规化处理或者归一化处理。
        # 对边缘强度进行归一化处理得到边缘强度的灰度级显示，如果得到的对比度较低，还可以通过伽马变换进行对比度增强
        edge = edge / np.max(edge)
        edge = np.power(edge, 1)
        edge *= 255
        edge = edge.astype(np.uint8)
        cv.imshow("Soble_scale", edge)
        # cv.imwrite("./image/Soble_scale.png", edge)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print("Usge: python.py imageFile")
