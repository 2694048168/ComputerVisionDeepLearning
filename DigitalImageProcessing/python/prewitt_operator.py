#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: Prewitt Operator(Algorithm) —— 可分离卷积核
           Prewitt算子均是可分离的，为了减少耗时，
        在代码实现中, 利用卷积运算的结合律先进行水平方向上的平滑，再进行垂直方向上的差分，
        或者先进行垂直方向上的平滑，再进行水平方向上的差分
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-10-13
"""

import sys
from scipy import signal
import numpy as np
import cv2 as cv


def PrewittOperator(image, _boundary="symm"):
    """Prewitt 可分离边缘检测算子

    Args:
        image ([ndarray]): 原始输入图像
        _boundary (str, optional): 边界填充模式. Defaults to "symm".

    Returns:
        [tuple]: Prewitt 算子输出结果，元组形式
    """
    # 1 垂直方向的均值平滑
    kernel_smooth_y = np.array([[1], [1], [1]], np.float32)
    img_conv_prewitt_x = signal.convolve2d(image, kernel_smooth_y, mode="same", boundary=_boundary)
    # 2 水平方向的差分操作
    diff_x = np.array([[1, 0, -1]], np.float32)
    img_conv_prewitt_x = signal.convolve2d(image, diff_x, mode="same", boundary=_boundary)

    # 1 水平方向的均值平滑
    kernel_smooth_x = np.array([[1, 1, 1]], np.float32)
    img_conv_prewitt_y = signal.convolve2d(image, kernel_smooth_x, mode="same", boundary=_boundary)
    # 2 垂直方向的差分操作
    diff_y = np.array([[1], [0], [-1]], np.float32)
    img_conv_prewitt_y = signal.convolve2d(img_conv_prewitt_y, diff_y, mode="same", boundary=_boundary)

    return (img_conv_prewitt_x, img_conv_prewitt_y)


# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or dictory.")
        cv.imshow("OriginImage",image)

        # 注意区分边缘强度和边缘强度的灰度级显示
        img_prewitt_x, img_prewitt_y = PrewittOperator(image)

        # 计算绝对值，获取水平方向和垂直方向的边缘强度
        abs_img_prewitt_x = np.abs(img_prewitt_x)
        abs_img_prewitt_y = np.abs(img_prewitt_y)

        # 水平方向和垂直方向的边缘强度的灰度级显示
        edge_x = abs_img_prewitt_x.copy()
        edge_y = abs_img_prewitt_y.copy()
        # 将大于 255 直接进行饱和操作
        edge_x[edge_x>255] = 255
        edge_y[edge_y>255] = 255
        # 数据类型转换
        edge_x = edge_x.astype(np.uint8)
        edge_y = edge_y.astype(np.uint8)
        cv.imshow("edge_x", edge_x)
        cv.imshow("edge_y", edge_y)

        # 根据 prewitt 两个卷积结果, 计算最终的边缘强度
        # 计算最终的边缘强度有多种方式, 采用 插值法
        edge = 0.5*abs_img_prewitt_x + 0.5*abs_img_prewitt_y
        # 边缘轻度的灰度级显示
        edge[edge>255] = 255
        edge = edge.astype(np.uint8)
        cv.imshow("edge", edge)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print("Usge: python.py imageFile")
