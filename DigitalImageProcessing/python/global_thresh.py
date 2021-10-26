#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 全局阈值分割
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-09-28
"""

import sys
import numpy as np
import cv2 as cv


# --------------------------
if __name__ == "__main__":
    # 0. 全局阈值分割
    src = np.array([[123, 234, 68], [33, 51, 17], [48, 98, 234], [129, 89, 27], [45, 167, 134]])
    # 全局阈值操作 (在图像处理中，一般不改变原图，可以 copy 一份)
    src[src > 150] = 255
    src[src <= 150] = 0
    print(f"The result of global thresh :\n {src}")

    # 1. OpenCV 提供的函数
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or directory.")
            sys.exit()
        cv.imshow("OriginImage", image)

        # 手动设置阈值
        threshold_value = 150
        maxVal = 255
        ret, dst = cv.threshold(image, threshold_value, maxVal, cv.THRESH_BINARY)
        if ret is None:
            print("The operator of threthold is unsuccfully.")
        print(f"The result of global thresh :\n {dst}")
        cv.imshow("binary_threthold", dst)

        # 阈值处理方式
        otsuThe = 0
        utsuThe, dst_Otsu = cv.threshold(image, otsuThe, maxVal, cv.THRESH_OTSU)
        print(f"The result of global thresh :\n {utsuThe}")
        print(f"The result of global thresh :\n {dst_Otsu}")
        cv.imshow("OTSU_threshold", dst_Otsu)

        # 阈值处理方式 enum ThresholdTypes {}
        triThe = 0
        triThe, dst_tri = cv.threshold(image, triThe, maxVal, cv.THRESH_TRIANGLE + cv.THRESH_BINARY_INV)
        print(f"The result of global thresh :\n {triThe}")
        print(f"The result of global thresh :\n {dst_tri}")
        cv.imshow("TRIANGLE_threshold", dst_tri)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print(f"Usage: python gaussBlur imageFile.")
