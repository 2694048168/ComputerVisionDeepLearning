#!/usr/bin/env python3
# encoding: utf-8


"""
@Funciton: 图像的伽马变换，可以实现全局或者局部的对比度增强，亮度增大，人眼观察的视觉更多
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-09-16
"""

import sys
import cv2 as cv
import numpy as np


# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or dictory.")
        cv.imshow("OriginImage",image)
        
        # 图像的伽马变换的实质就是对图像矩阵中的每一个数值进行幂运算
        # 1. 图像归一化操作
        image = image / 255.0
        # 伽马变换
        gamma = 0.5
        output_gamma_img = np.power(image, gamma)
        cv.imshow("GammaTransform", output_gamma_img)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print(f"Usage: python histogram imageFile.")
    