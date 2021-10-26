#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 基于方向差分卷积核进行卷积操作——Roberts Operator(Algorithm)
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-10-13
"""

import sys
from scipy import signal
import numpy as np
import cv2 as cv


def RobertsOperator(image, _boundary="fill", _fillvalue=0):
    """计算 Roberts 算子卷积后的结果图像

    Args:
        image ([ndarray]): 原始输入图像
        _boundary (str, optional): 边界填充模式. Defaults to "fill".
        _fillvalue (int, optional): 填充值. Defaults to 0.

    Returns:
        [tuple]: Roberts 算子进行卷积后的两个结果图像，元组形式返回
    """
    H_img, W_img = image.shape[0:2]
    H_kernel, W_kernel = 2, 2

    # 第一个卷积核以及锚(anchor)点位置
    kernel_1 = np.array([[1, 0], [0, -1]], np.float32)
    anchor_1_row, anchor_1_col = 0, 0
    # 计算 full 卷积
    img_conv_full_1 = signal.convolve2d(image, kernel_1, mode="full", boundary=_boundary, fillvalue=_fillvalue)
    # 截取 full 结果以获取 same 卷积
    img_conv_same_1 = img_conv_full_1[H_kernel-anchor_1_row-1:H_img+H_kernel-anchor_1_row-1, W_kernel-anchor_1_col-1:W_img+W_kernel-anchor_1_col-1] 

    # 第二个卷积核以及锚(anchor)点位置
    kernel_2 = np.array([[0, 1], [-1, 0]], np.float32)
    anchor_2_row, anchor_2_col = 0, 1
    # 计算 full 卷积
    img_conv_full_2 = signal.convolve2d(image, kernel_2, mode="full", boundary=_boundary, fillvalue=_fillvalue)
    # 根据锚点位置截取 full 卷积获取 same 卷积
    img_conv_same_2 = img_conv_full_2[H_kernel-anchor_2_row-1:H_img+H_kernel-anchor_2_row-1, W_kernel-anchor_2_col-1:W_img+W_kernel-anchor_2_col-1]

    return (img_conv_same_1, img_conv_same_2)


# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or dictory.")
        cv.imshow("OriginImage",image)

        # 卷积的边界填充模式选择 symm
        img_conv_1, img_conv_2 = RobertsOperator(image, "symm")

        # 在 45 度 方向上的边缘强度的灰度级显示【区分边缘强度和边缘强度的灰度级显示的区别】
        img_conv_1 = np.abs(img_conv_1)
        edge_45 = img_conv_1.astype(np.uint8)
        cv.imshow("edge_45", edge_45)

        # 在 135 度 方向上的边缘强度的灰度级显示【区分边缘强度和边缘强度的灰度级显示的区别】
        img_conv_2 = np.abs(img_conv_2)
        edge_135 = img_conv_2.astype(np.uint8)
        cv.imshow("edge_135", edge_135)

        # 利用平方和的开方赖衡量最后的输出的边缘
        edge = np.sqrt(np.power(img_conv_1, 2.0) + np.power(img_conv_2, 2.0))
        edge = np.round(edge)
        edge[edge>255] = 255
        edge = edge.astype(np.uint8)
        cv.imshow("edge", edge)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print("Usge: python.py imageFile")
