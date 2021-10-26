#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 拉普拉斯二维微分算子 —— 不可分离的单独一个卷积
            拉普拉斯算子对噪声很敏感，使用首先应对图像进行高斯平滑，然后再与拉普拉斯算子卷积，最后得到二值化边缘图。
            水墨效果的边缘图，该边缘图也在某种程度上体现了边缘强度。
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-10-17
"""

import sys
from scipy import signal
import numpy as np
import cv2 as cv


def LaplaceOperator(image, _boundary="fill", _fillvalue=0):
    # laplace convolution kernel
    laplace_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32)
    # laplace_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32)

    img_laplace_conv = signal.convolve2d(image, laplace_kernel, mode="same", boundary=_boundary, fillvalue=_fillvalue)

    return img_laplace_conv


# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or dictory.")
        cv.imshow("OriginImage",image)

        # -------- Laplace Operator --------
        img_laplce_conv = LaplaceOperator(image, "symm")

        # case 1, 阈值化处理
        thresholdEdge = np.copy(img_laplce_conv)
        thresholdEdge[thresholdEdge>0] = 255
        thresholdEdge[thresholdEdge<0] = 0
        thresholdEdge = thresholdEdge.astype(np.uint8)
        cv.imshow("ThresholdEdge", thresholdEdge)

        # case 2, 抽象化处理(水墨画效果)
        asbstractionEdge = np.copy(img_laplce_conv)
        asbstractionEdge = asbstractionEdge.astype(np.float32)
        asbstractionEdge[asbstractionEdge>=0] = 1.0
        asbstractionEdge[asbstractionEdge<0] = 1.0 + np.tanh(asbstractionEdge[asbstractionEdge<0])
        cv.imshow("AsbstractionEdge", asbstractionEdge)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print("Usge: python.py imageFile")
