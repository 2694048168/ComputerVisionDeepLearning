#!/usr/bin/env python3
# encoding: utf-8


"""
@Funciton: 已知原始坐标和变换后的坐标，通过方程方法计算投影矩阵，利用 投影矩阵完成投影变换
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-09-14
"""

import sys
import numpy as np
import cv2 as cv


# --------------------------
if __name__ == "__main__":
    # 1. 方程方法
    src = np.array([[0, 0], [200, 0], [0, 200], [200, 200]], np.float32)
    dst = np.array([[100, 20], [200, 20], [50, 70], [250, 70]], np.float32)
    perspective_transform_matrix = cv.getPerspectiveTransform(src=src, dst=dst)
    print(
        f"The affine transform maxtrix type is\n {type(perspective_transform_matrix)}")
    print(
        f"The affine transform maxtrix data type is\n {perspective_transform_matrix.dtype}")
    print(f"The affine transform maxtrix is\n {perspective_transform_matrix}")

    # 2. 利用投影矩阵完成投影变换
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        cv.imshow("Origin_Image", image)

        # 原始图像的高和宽
        h, w = image.shape
        # 原始坐标点和投影变换后的坐标点
        src = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]], np.float32)
        dst = np.array([[50, 50], [w/3, 50], [50, h-1], [w-1, h-1]], np.float32)
        # compute perspective transform matrix
        p = cv.getPerspectiveTransform(src, dst)
        # 利用投影矩阵完成投影变换
        r = cv.warpPerspective(image, p, (w, h), borderValue=125)
        cv.imshow("Perspective_Image", r)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print(f"Usage: python warpPerceptive image.")
