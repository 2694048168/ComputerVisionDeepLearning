#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: Scharr 边缘检测算子 —— 不可分离卷积核
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-10-16
"""

import sys
from scipy import signal
import numpy as np
import cv2 as cv


def ScharrOperator(image, _boundary="symm"):
    # image 和 scharr_x 在水平方向卷积，反映垂直方向的边缘强度
    scharr_x = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]], np.float32)
    img_convX = signal.convolve2d(image, scharr_x, mode="same", boundary=_boundary)

    # image 和 scharr_y 在垂直方向卷积，反映水平方向的边缘强度
    scharr_y = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], np.float32)
    img_convY = signal.convolve2d(image, scharr_y, mode="same", boundary=_boundary)

    return (img_convX, img_convY)


# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or dictory.")
        cv.imshow("OriginImage",image)

        img_convX, img_convY= ScharrOperator(image)

        # image 和 scharr_x 在水平方向卷积结果，反映垂直方向的边缘强度
        img_convX = np.abs(img_convX)
        edge_vertical = img_convX.astype(np.uint8)
        cv.imshow("edge_vertical", edge_vertical)

        # image 和 scharr_y 在垂直方向卷积结果，反映水平方向的边缘强度
        img_convY = np.abs(img_convY)
        edge_horizontal = img_convY.astype(np.uint8)
        cv.imshow("edge_horizontal", edge_horizontal)

        # 利用平方和的开方赖衡量最后的输出的边缘
        edge = np.sqrt(np.power(img_convX, 2.0) + np.power(img_convY, 2.0))
        edge = np.round(edge)
        edge[edge>255] = 255
        edge = edge.astype(np.uint8)
        cv.imshow("edge", edge)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print("Usge: python.py imageFile")
