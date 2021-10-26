#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: Robinson 边缘检测算子 —— 四面八方的进行差分(梯度信息，边缘强度)
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-10-16
"""

import sys
from scipy import signal
import numpy as np
import cv2 as cv


def RobinsonOperator(image, _boundary="fill", _fillvalue=0):
    """函数 RobinsonOperator 实现图像与每一个核的卷积，
    然后取绝对值作为对应方向上的边缘强度的量化, 在所有这8个方向上的对应位置取最大值作为最后输出的边缘强度

    Args:
        image (ndarray): 输入原始图像
        _boundary (str, optional): 边界填充类型. Defaults to "fill".
        _fillvalue (int, optional): 边界填充数值. Defaults to 0.

    Returns:
        [ndarray]: 输入图像对应的边缘强度
    """
    # 8 个方向的边缘强度
    list_edge = []

    # 1. image and kernel_1 convolution, 对卷积结果取绝对值，获得边缘强度
    kernel_1 = np.array([[1, 1, 1], [1, -2, 1], [-1, -1, -1]])
    img_kernel_1 = signal.convolve2d(image, kernel_1, mode="same", boundary=_boundary, fillvalue=_fillvalue)
    list_edge.append(np.abs(img_kernel_1))

    # 2. image and kernel_1 convolution, 对卷积结果取绝对值，获得边缘强度
    kernel_2 = np.array([[1, 1,1], [-1, -2, 1], [-1, -1, 1]])
    img_kernel_2 = signal.convolve2d(image, kernel_2, mode="same", boundary=_boundary, fillvalue=_fillvalue)
    list_edge.append(np.abs(img_kernel_2))

    # 3. image and kernel_1 convolution, 对卷积结果取绝对值，获得边缘强度
    kernel_3 = np.array([[-1, 1, 1], [-1, -2, 1], [-1, 1, 1]])
    img_kernel_3 = signal.convolve2d(image, kernel_3, mode="same", boundary=_boundary, fillvalue=_fillvalue)
    list_edge.append(np.abs(img_kernel_3))

    # 4. image and kernel_1 convolution, 对卷积结果取绝对值，获得边缘强度
    kernel_4 = np.array([[-1, -1, -1], [-1, -2, -1], [1, 1, 1]])
    img_kernel_4 = signal.convolve2d(image, kernel_4, mode="same", boundary=_boundary, fillvalue=_fillvalue)
    list_edge.append(np.abs(img_kernel_4))

    # 5. image and kernel_1 convolution, 对卷积结果取绝对值，获得边缘强度
    kernel_5 = np.array([[-1, -1, -1], [1, -2, 1], [1, 1, 1]])
    img_kernel_5 = signal.convolve2d(image, kernel_5, mode="same", boundary=_boundary, fillvalue=_fillvalue)
    list_edge.append(np.abs(img_kernel_5))

    # 6. image and kernel_1 convolution, 对卷积结果取绝对值，获得边缘强度
    kernel_6 = np.array([[1, -1, -1], [1, -2, -1], [1, 1, 1]])
    img_kernel_6 = signal.convolve2d(image, kernel_6, mode="same", boundary=_boundary, fillvalue=_fillvalue)
    list_edge.append(np.abs(img_kernel_6))

    # 7. image and kernel_1 convolution, 对卷积结果取绝对值，获得边缘强度
    kernel_7 = np.array([[1, 1, -1], [1, -2, -1], [1, 1, -1]])
    img_kernel_7 = signal.convolve2d(image, kernel_7, mode="same", boundary=_boundary, fillvalue=_fillvalue)
    list_edge.append(np.abs(img_kernel_7))

    # 8. image and kernel_1 convolution, 对卷积结果取绝对值，获得边缘强度
    kernel_8 = np.array([[1, 1, 1], [1, -2, -1], [1, -1, -1]])
    img_kernel_8 = signal.convolve2d(image, kernel_8, mode="same", boundary=_boundary, fillvalue=_fillvalue)
    list_edge.append(np.abs(img_kernel_8))

    # 对 8 个方向的边缘强度，在对应位置取最大值作为图像最后的边缘强度
    edge = list_edge[0]
    for idx in range(len(list_edge)):
        edge = edge * (edge >= list_edge[idx] + list_edge[idx]*(edge < list_edge[idx]))

    return edge


# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or dictory.")
        cv.imshow("OriginImage",image)

        # 边缘强度
        edge = RobinsonOperator(image, _boundary="symm")
        # 边缘强度的灰度级显示
        edge[edge>255] = 255
        edge = edge.astype(np.uint8)
        cv.imshow("edge", edge)

        # 简单的素描效果，对 edge 进行反色处理
        pencil_sketch = 255 - edge
        cv.imshow("PencilSketch", pencil_sketch)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print("Usge: python.py imageFile")
