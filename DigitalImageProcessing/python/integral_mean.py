#!/usr/bin/env python3
# encoding: utf-8


"""
@Funciton: 图像积分; 快速均值滤波(平滑)
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-09-23
"""

import sys
import cv2 as cv
import numpy as np


def integral(image):
    """图像积分思想：首先对图像矩阵进行按行积分，然后按列积分；反之亦然；
    为了在快速均值平滑中省去判断边界问题，对积分后的图像矩阵的上边和左边进行 zero-padding

    Args:
        image ([ndarray]): 输入图像矩阵

    Returns:
        [ndarray]: 积分图像矩阵，zero-padding
    """
    rows, cols = image.shape
    # 按行积分
    inteImgCol = np.zeros((rows, cols), np.float32)
    for idx_row in range(rows):
        for idx_col in range(cols):
            if idx_col == 0:
                inteImgCol[idx_row][idx_col] = image[idx_row][idx_col]
            else:
                inteImgCol[idx_row][idx_col] = inteImgCol[idx_row][idx_col - 1] + image[idx_row][idx_col]
    # 按列积分
    inteImgColRow = np.zeros((rows, cols), np.float32)
    for idx_row in range(rows):
        for idx_col in range(cols):
            if idx_col == 0:
                inteImgColRow[idx_row][idx_col] = inteImgCol[idx_row][idx_col]
            else:
                inteImgColRow[idx_row][idx_col] = inteImgColRow[idx_row - 1][idx_col] + inteImgCol[idx_row][idx_col]
    # 上边和左边进行 zero-padding
    integral_img = np.zeros((rows + 1, cols + 1), np.float32)
    integral_img[1:rows+1, 1:cols+1] = inteImgColRow

    return integral_img


def fastMeanBlur(image, winSize, borderType=cv.BORDER_DEFAULT):
    """实现快速均值滤波，使用镜像进行边界扩充，解决随着窗口的增大，平滑后图像边界黑色很明显的问题

    Args:
        image ([ndarray]): 输入图像
        winSize ([tuple]): 滤波核大小，宽高均为奇数
        borderType ([OpenCV], optional): 使用opencv提供的边界填充类型. Defaults to cv.BORDER_DEFAULT.

    Returns:
        [ndarray]: 均值滤波后的图像
    """
    halfH = (winSize[0] - 1) // 2
    halfW = (winSize[1] - 1) // 2
    ratio = 1.0 / (winSize[0] * winSize[1])
    # 边界填充
    paddImage = cv.copyMakeBorder(image, halfH, halfH, halfW, halfW, borderType=borderType)
    # 图像积分
    paddIntegral = integral(paddImage)
    rows, cols = image.shape
    meanBlurImage = np.zeros(image.shape, np.float32)
    border_r, border_c = 0, 0
    for idx_h in range(halfH, halfH+rows, 1):
        for idx_w in range(halfW, halfW+cols, 1):
            meanBlurImage[border_r][border_c]=(paddIntegral[idx_h+halfH+1][idx_w+halfW+1]
            +paddIntegral[idx_h-halfH][idx_w-halfW]
            -paddIntegral[idx_h+halfH+1][idx_w-halfW]
            -paddIntegral[idx_h-halfH][idx_w+halfW+1])*ratio
            border_c += 1 
        border_r += 1
        border_c = 0

    return meanBlurImage


# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or dictory.")
        cv.imshow("OriginImage", image)

        # 1. 快速均值滤波
        meanImg = fastMeanBlur(image, (5, 5), borderType=cv.BORDER_DEFAULT)
        cv.imshow("MeanImage", meanImg.astype(np.uint8))

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print(f"Usage: python histogram imageFile.")
