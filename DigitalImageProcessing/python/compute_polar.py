#!/usr/bin/env python3
# encoding: utf-8


"""
@Funciton: 笛卡尔坐标和极坐标之间转换
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-09-15
"""

import math
import numpy as np
import cv2 as cv


# --------------------------
if __name__ == "__main__":
    # 1. 直接使用数学公式进行转换
    # 笛卡尔坐标为 (11, 13) 以 (3， 5) 为中心
    r = math.sqrt(math.pow(11 - 3, 2) + math.pow(13 - 5, 2))
    theta = math.atan2(13 - 5, 11 - 3) / math.pi * 180 # 转换角度
    print(f"The r is {r}")
    print(f"The theta is {theta}")

    # 2. 利用 OpenCV 提供的函数进行计算
    x = np.array([[0, 1, 2],[0, 1, 2],[0, 1, 2]], np.float64) - 1
    y = np.array([[0, 0, 0],[1, 1, 1],[2, 2, 2]], np.float64) - 1
    r, theta = cv.cartToPolar(x, y, angleInDegrees=True)
    print(f"The r is {r}")
    print(f"The theta is {theta}")

    # 3. 将极坐标转换为笛卡尔坐标
    angle = np.array([[30, 31], [30, 31]], np.float32)
    r = np.array([[10, 10], [11, 11]], np.float32)
    # 计算出来的是以 原点 (0，0) 为变换中心的坐标，按照需要进行转换
    x, y = cv.polarToCart(r, angle, angleInDegrees=True)
    x += -12
    y += 15
    print(f"The cart is: \n {x}")
    print(f"The cart is: \n {y}")
        