#!/usr/bin/env python3
# encoding: utf-8


"""
@Funciton: 引导滤波(导向滤波); 利用几何变换加速导向滤波(快速引导滤波)
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-09-25
"""

import sys
import math
import cv2 as cv
import numpy as np

# 快速均值滤波
from integral_mean import fastMeanBlur


def guideFilter(image, p_img, winSize, eps):
    """导向滤波-何凯明-2013年
    K.He,J.Sun, and X.Tang.Guided image filtering. In ECCV, pages 1-14.2010.
    K. He,J. Sun, and X.Tang. Guided image filtering. TPAMI,35(6):1397-1409,2013.

    Args:
        image ([ndarray]): 输入图像I image，灰度值归一化[0，1]浮点数矩阵
        p_img ([ndarray]): 输入图像P 
        eps ([type]): 正则化参数

    Returns:
        [ndarray]: 灰度值[0，1]图像矩阵
    """
    # 计算 image 的均值滤波
    mean_I = fastMeanBlur(image, winSize, cv.BORDER_DEFAULT)
    # 计算 p_img 的均值滤波
    mean_p = fastMeanBlur(p_img, winSize, cv.BORDER_DEFAULT)
    # 计算 image*p_img 的均值滤波
    Ip = image * p_img
    mean_Ip = fastMeanBlur(Ip, winSize, cv.BORDER_DEFAULT)

    # 协方差
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = fastMeanBlur(image*image, winSize, cv.BORDER_DEFAULT)

    # 方差
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # 对 a 和 b 进行均值滤波
    mean_a = fastMeanBlur(a, winSize, cv.BORDER_DEFAULT)
    mean_b = fastMeanBlur(b, winSize, cv.BORDER_DEFAULT)
    q_img = mean_a * image + mean_b

    return q_img


# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        print(image.shape)
        if image is None:
            print(f"Error: no such file or dictory.")
        cv.imshow("OriginImage", image)

        # [0-255] ----> [0-1]
        image_0_1 = image / 255.0
        # 导向滤波
        result = guideFilter(image_0_1, image_0_1, (17, 17), math.pow(0.2, 2.0))
        cv.imshow("guideFilter", result)
        # 保存导向滤波结果
        result = result * 255
        result[result > 255] = 255
        result = result.astype(np.uint8)
        cv.imwrite("./../image/guideFilter.png", result)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print(f"Usage: python histogram imageFile.")
