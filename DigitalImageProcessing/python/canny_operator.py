#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: Canny 边缘检测算子 - 解决边缘梯度方向的信息问题和阈值处理问题
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-10-16
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


def non_maximum_suppression_default(dx, dy):
    """函数 non_maximum_suppression_default 实现非极大值抑制的默认计算万式，
    需要汪意的是，在函数实现中最好利用 dx 和dy 的平方和开方的方式来衡量边缘强度。

    Args:
        dx ([ndarray]): dx 代表图像矩阵与 sobel_x 或 prewitt_x 的卷枳，即与水平方向差分算子卷积结果
        dy ([ndarray]): dy 代表图像矩阵与 sobel_y 或 prewitt_y 的卷积，即与垂直方向差分算子卷积结果

    Returns:
        [ndarray]: 通过非极大值抑制后的边缘强度图
    """
    # 边缘强度
    edgeMagnitude = np.sqrt(np.power(dx, 2.0) + np.power(dy, 2.0))

    rows, cols = dx.shape
    # 梯度方向
    gradientDirection = np.zeros((rows, cols))
    # 边缘强度非极大值抑制
    edgeMagnitude_nonMaxSuppression = np.zeros((rows, cols))

    # 对每一个位置进行非极大值抑制的处理，非极大值抑制操作返回的仍然是一个矩阵
    # 因为卷积运算采用的是补零操作，导致所得到的 magnitude 产生了额外的边缘响应
    # 如果采用的是以边界为对称的边界扩充方式，那么卷积结果的边界全是 0。在非极大值抑制这一步中，对边界不做任何处理
    for row_idx in range(1, rows - 1):
        for col_idx in range(1, cols - 1):
            # angle (用于表征或者度量梯度方向的量) 的范围 [0, 180] [-180, 0]
            angle = math.atan2(dy[row_idx][col_idx], dx[row_idx][col_idx]) / math.pi * 180
            gradientDirection[row_idx][col_idx] = angle

            # 左/右方向
            if (abs(angle) < 22.5 or abs(angle) > 157.5):
                if (edgeMagnitude[row_idx][col_idx] > edgeMagnitude[row_idx][col_idx - 1] and edgeMagnitude[row_idx][col_idx] > edgeMagnitude[row_idx][col_idx + 1]):
                    edgeMagnitude_nonMaxSuppression[row_idx][col_idx] = edgeMagnitude[row_idx][col_idx]
            
            # 左上/右下方向
            if (angle >= 22.5 and angle < 67.5 or (-angle > 112.5 and -angle <= 157.5)):
                if (edgeMagnitude[row_idx][col_idx] > edgeMagnitude[row_idx - 1][col_idx - 1] and edgeMagnitude[row_idx][col_idx] > edgeMagnitude[row_idx + 1][col_idx + 1]):
                    edgeMagnitude_nonMaxSuppression[row_idx][col_idx] = edgeMagnitude[row_idx][col_idx]

            # 上/下方向
            if (abs(angle >= 67.5) and abs(angle) <= 112.5):
                if (edgeMagnitude[row_idx][col_idx] > edgeMagnitude[row_idx - 1][col_idx] and edgeMagnitude[row_idx][col_idx] > edgeMagnitude[row_idx + 1][col_idx]):
                    edgeMagnitude_nonMaxSuppression[row_idx][col_idx] = edgeMagnitude[row_idx][col_idx]

            # 右上/左下方向
            if (angle > 112.5 and angle <= 157.5 or (-angle >= 22.5 and -angle < 67.5)):
                if (edgeMagnitude[row_idx][col_idx] > edgeMagnitude[row_idx - 1][col_idx + 1] and edgeMagnitude[row_idx][col_idx] > edgeMagnitude[row_idx + 1][col_idx - 1]):
                    edgeMagnitude_nonMaxSuppression[row_idx][col_idx] = edgeMagnitude[row_idx][col_idx]

    return edgeMagnitude_nonMaxSuppression


def non_maximum_suppression_interpolation(dx, dy):
    """函数 non_maximum_suppression_interpolation 实现非极大值抑制的插值万式，
    需要汪意的是，在函数实现中最好利用 dx 和dy 的平方和开方的方式来衡量边缘强度。

    Args:
        dx ([ndarray]): dx 代表图像矩阵与 sobel_x 或 prewitt_x 的卷枳，即与水平方向差分算子卷积结果
        dy ([ndarray]): dy 代表图像矩阵与 sobel_y 或 prewitt_y 的卷积，即与垂直方向差分算子卷积结果

    Returns:
        [ndarray]: 通过非极大值抑制后的边缘强度图
    """

    # 边缘强度
    edgeMagnitude = np.sqrt(np.power(dx, 2.0) + np.power(dy, 2.0))

    rows, cols = dx.shape
    # 梯度方向
    gradientDirection = np.zeros((rows, cols))
    # 边缘强度非极大值抑制
    edgeMagnitude_nonMaxSuppression = np.zeros((rows, cols))

    # 对每一个位置进行非极大值抑制的处理，非极大值抑制操作返回的仍然是一个矩阵
    # 因为卷积运算采用的是补零操作，导致所得到的 magnitude 产生了额外的边缘响应
    # 如果采用的是以边界为对称的边界扩充方式，那么卷积结果的边界全是 0。在非极大值抑制这一步中，对边界不做任何处理
    for row_idx in range(1, rows - 1):
        for col_idx in range(1, cols - 1):
            if dy[row_idx][col_idx] == 0 and dx[row_idx][col_idx] == 0:
                continue

            # angle (用于表征或者度量梯度方向的量) 的范围 [0, 180] [-180, 0]
            angle = math.atan2(dy[row_idx][col_idx], dx[row_idx][col_idx]) / math.pi * 180
            gradientDirection[row_idx][col_idx] = angle

            # 左上方和上方的插值，右下方和下方的插值
            if (angle > 45 and angle <= 90) or (angle > -135 and angle <= -90):
                ratio = dx[row_idx][col_idx] / dy[row_idx][col_idx]
                leftTop_top = ratio * edgeMagnitude[row_idx - 1][col_idx - 1] + (1 - ratio) * edgeMagnitude[row_idx - 1][col_idx]
                rightBottom_bottom = (1 - ratio) * edgeMagnitude[row_idx + 1][col_idx] + ratio*edgeMagnitude[row_idx + 1][col_idx + 1]

                if edgeMagnitude[row_idx][col_idx] > leftTop_top and edgeMagnitude[row_idx][col_idx] > rightBottom_bottom:
                    edgeMagnitude_nonMaxSuppression[row_idx][col_idx] = edgeMagnitude[row_idx][col_idx]

            # 右上方和上方的插值，左下方和下方的插值
            if (angle > 90 and angle <= 135) or (angle > -90 and angle <= -45):
                ratio = abs(dx[row_idx][col_idx] / dy[row_idx][col_idx])
                rightTop_top = ratio * edgeMagnitude[row_idx - 1][col_idx + 1] + (1 - ratio) * edgeMagnitude[row_idx - 1][col_idx]
                leftBottom_bottom = ratio * edgeMagnitude[row_idx + 1][col_idx - 1] + (1 - ratio) * edgeMagnitude[row_idx + 1][col_idx]

                if edgeMagnitude[row_idx][col_idx] > rightTop_top and edgeMagnitude[row_idx][col_idx] > leftBottom_bottom:
                    edgeMagnitude_nonMaxSuppression[row_idx][col_idx] = edgeMagnitude[row_idx][col_idx]

            # 左上方和左方的插值，右下方和右方的插值
            if (angle > 0 and angle <= 45) or (angle > -180 and angle <= -135):
                ratio = dx[row_idx][col_idx] / dy[row_idx][col_idx]
                leftTop_left = ratio * edgeMagnitude[row_idx - 1][col_idx - 1] + (1 - ratio) * edgeMagnitude[row_idx][col_idx - 1]
                rightBottom_right = ratio * edgeMagnitude[row_idx + 1][col_idx + 1] + (1 - ratio) * edgeMagnitude[row_idx][col_idx + 1]

                if edgeMagnitude[row_idx][col_idx] > rightBottom_right and edgeMagnitude[row_idx][col_idx] > leftTop_left:
                    edgeMagnitude_nonMaxSuppression[row_idx][col_idx] = edgeMagnitude[row_idx][col_idx]

            # 右上方和右方的插值，左下方和左方的插值
            if (angle > 135 and angle <= 180) or (angle > -45 and angle <= -0):
                ratio = abs(dx[row_idx][col_idx] / dy[row_idx][col_idx])
                rightTop_right = ratio * edgeMagnitude[row_idx - 1][col_idx + 1] + (1 - ratio) * edgeMagnitude[row_idx][col_idx + 1]
                leftBottom_left = ratio * edgeMagnitude[row_idx + 1][col_idx - 1] + (1 - ratio) * edgeMagnitude[row_idx][col_idx - 1]

                if edgeMagnitude[row_idx][col_idx] > rightTop_right and edgeMagnitude[row_idx][col_idx] > leftBottom_left:
                    edgeMagnitude_nonMaxSuppression[row_idx][col_idx] = edgeMagnitude[row_idx][col_idx]

    return edgeMagnitude_nonMaxSuppression


def checkInRanger(row_idx, col_idx, rows, cols):
    """判断一个点的坐标时候在图像范围内

    Args:
        row_idx ([int]): 行坐标索引
        col_idx ([int]): 列坐标索引
        rows ([int]): 行数
        cols ([int]): 列数

    Returns:
        [bool]: 该点的坐标是否在图像范围内
    """
    if  row_idx >= 0 and row_idx < rows and col_idx >= 0 and col_idx < cols:
        return True
    else:
        return False


def trace(edgeMag_nonMaxSup, edge, lowerThresh, row_idx, col_idx, rows, cols):
    # 大于高阈值的点认为是边缘
    if edge[row_idx][col_idx] == 0:
        edge[row_idx][col_idx] = 255
        for idx in range(-1, 2):
            for index in range(-1, 2):
                if checkInRanger(row_idx+idx, col_idx+index, rows, cols) and edgeMag_nonMaxSup[row_idx+idx][col_idx+index] >= lowerThresh:
                    trace(edgeMag_nonMaxSup, edge, lowerThresh, row_idx+idx, col_idx+index, rows, cols)


def hysteresisThreshold(edge_nonMaxSup, lowerThresh, upperThresh):
    rows, cols = edge_nonMaxSup.shape
    edge = np.zeros((rows, cols), np.uint8)
    for row_idx in range(1, rows - 1):
        for col_idx in range(1, cols - 1):
            # 大于高阈值的点设置为边缘点，而且以该点作为起始点延长边缘
            if edge_nonMaxSup[row_idx][col_idx] >= upperThresh:
                trace(edge_nonMaxSup, edge, lowerThresh, row_idx, col_idx, rows, cols)

            # 小于低阈值的点直接删掉
            if edge_nonMaxSup[row_idx][col_idx] < lowerThresh:
                edge[row_idx][col_idx] = 0

    return edge


# --------------------------
if __name__ == "__main__":
    """实现非极大值抑制和滞后阈值处理，实现 Canny 边缘检测，
    其中分别显示 Sobel 边缘强度的灰度级、非极大值抑制后的灰度级、滞后阈值处理后的二值化边缘图
    对于非极大值抑制使用的是第一种默认方式，可以换成插值方式。
    """
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or dictory.")
        cv.imshow("OriginImage",image)

        # -------- Canny 边缘检测 --------
        # step 1, Sobel kernel convolution
        img_sobel_x, img_sobel_y = SobelOperator(image, 3)

        # step 2, Edge strength
        edge = np.sqrt(np.power(img_sobel_x, 2.0) + np.power(img_sobel_y, 2.0))
        edge[edge>255] = 255
        edge = edge.astype(np.uint8)
        cv.imshow("edgeStrenth", edge)

        # step 3, Non-maximum suppression
        edgeMag_nonMaxSuppression = non_maximum_suppression_default(img_sobel_x, img_sobel_y)
        edgeMag_nonMaxSuppression[edgeMag_nonMaxSuppression>255] = 255
        edgeMag_nonMaxSuppression = edgeMag_nonMaxSuppression.astype(np.uint8)
        cv.imshow("edgeNonMaxSup", edgeMag_nonMaxSuppression)

        # step 4, Dual threshold hysteresis processing
        edge = hysteresisThreshold(edgeMag_nonMaxSuppression, 60, 180)
        lowerThresh = 40
        upperThresh = 150
        cv.imshow("CannyImage", edge)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print("Usge: python.py imageFile")
