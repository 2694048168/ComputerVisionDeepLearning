#!/usr/bin/env python3
# encoding: utf-8


"""
@Funciton: 椒盐噪声; 中值滤波(平滑)-非线性滤波器
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-09-24
"""

import sys
import random
import cv2 as cv
import numpy as np


def salt(image, number):
    rows, cols = image.shape
    saltImg = np.copy(image)
    for idx in range(number):
        rand_row = random.randint(0, rows-1)
        rand_col = random.randint(0, cols-1)
        saltImg[rand_row][rand_col] = 255

    return saltImg


def medianBlur(image, winSize):
    rows, cols = image.shape
    winH, winW = winSize
    halfWinH = (winH - 1) // 2
    halfWinW = (winW - 1) // 2
    medianBlurImg = np.zeros(image.shape, image.dtype)
    for idx_row in range(rows):
        for idx_col in range(cols):
            # 边界判断
            rTop = 0 if idx_row - halfWinH < 0 else idx_row - halfWinH
            rBottom = rows - 1 if idx_row + halfWinH > rows - 1 else idx_row + halfWinH
            cLeft = 0 if idx_col - halfWinW < 0 else idx_col - halfWinW
            cRight = cols - 1 if idx_col + halfWinW > cols - 1 else idx_col + halfWinW
            # 取领域
            region = image[rTop:rBottom+1, cLeft:cRight+1]
            # 求中值
            medianBlurImg[idx_row][idx_col] = np.median(region)

    return medianBlurImg


# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or dictory.")
        cv.imshow("OriginImage", image)

        # 0. 添加椒盐噪声
        saltImg = salt(image, 142)
        cv.imshow("saltImg", saltImg)

        # 1. 中值滤波
        medianImg = medianBlur(saltImg, (5, 5))
        cv.imshow("medianImg", medianImg.astype(np.uint8))

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print(f"Usage: python histogram imageFile.")
