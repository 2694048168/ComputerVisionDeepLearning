#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 已知仿射变换矩阵，利用插值方法完成图像的几何变换(空间域操作)
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-09-14
"""

import sys
import numpy as np
import cv2 as cv


# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 利用命令行参数读取图像为灰度图像
        image = cv.imread(sys.argv[1], 0)
        cv.imshow("OriginImage", image)
    else:
        print(f"Usage: python warpAffine of OpenCV image.")

    # 原始图像的分辨率 H,W
    h, w = image.shape[:2]
    # 仿射变换矩阵，缩小 2 倍
    affine_matrix_1 = np.array([[0.5, 0, 0], [0, 0.5, 0]], np.float32)
    # 利用 OpenCV 提供的 API，函数原型和参数含义查看 API 接口即可
    dst_img_1 = cv.warpAffine(image, affine_matrix_1, (w, h), borderValue=125)
    cv.imshow("Scale_X2", dst_img_1)

    # 仿射变换矩阵，先缩小 2 倍，然后再平移
    affine_matrix_2 = np.array([[0.5, 0, w/4], [0, 0.5, h/4]], np.float32)
    dst_img_2 = cv.warpAffine(image, affine_matrix_2, (w, h), borderValue=125)
    cv.imshow("Scale_X2_translation", dst_img_2)

    # 仿射变换矩阵，先缩小 2 倍，然后再平移, 最后绕图像中心点旋转
    affine_matrix_3 = cv.getRotationMatrix2D((w/2.0, h/2.0), 30, 1)
    dst_img_3 = cv.warpAffine(dst_img_2, affine_matrix_3, (w, h), borderValue=125)
    cv.imshow("Scale_X2_translation_ratation", dst_img_3)

    # 利用 rotate 进行旋转，不是通过仿射变换矩阵实现的，而是通过类似矩阵转置方式的进行行列交换实现的
    rotate_img = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    cv.imshow("Rotate_img", rotate_img)

    cv.waitKey(0)
    cv.destroyAllWindows()
