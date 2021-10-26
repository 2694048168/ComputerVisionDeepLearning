#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 局部阈值分割; 自适应阈值分割;
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-09-30
"""

import sys
import numpy as np
import cv2 as cv


def adaptiveThreshold(image, winSize, ratio=0.15):
    # 1. 对图像进行均值平滑
    image_mean = cv.boxFilter(image, cv.CV_32FC1, winSize)
    # 2. 原图像和平滑图像做残差
    out_img = image - (1.0 - ratio)*image_mean
    # 3. 当差值大于或者等于 0 时， 输出值为 255；反之输出值为 0
    out_img[out_img >= 0] = 255
    out_img[out_img < 0] = 0
    out_img = out_img.astype(np.uint8)
    return out_img


# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or directory.")
            sys.exit()
        cv.imshow("OriginImage", image)

        thresholdImg = adaptiveThreshold(image, (5, 5))
        cv.imshow("ThresholdImage", thresholdImg)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print(f"Usage: python gaussBlur imageFile.")
