#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 二值图的逻辑运算
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-09-30
"""

import numpy as np
import cv2 as cv


# --------------------------
if __name__ == "__main__":
    src_img1 = np.array([[255, 0, 255]])
    src_img2 = np.array([[255, 0, 0]])
    # 1. 逻辑 与 运算
    dst_and = cv.bitwise_and(src1=src_img1, src2=src_img2)
    print(f"The and logic opeator is \n{dst_and}")
    # 2. 逻辑 或 运算
    dst_or = cv.bitwise_or(src1=src_img1, src2=src_img2)
    print(f"The or logic opeator is \n{dst_or}")
