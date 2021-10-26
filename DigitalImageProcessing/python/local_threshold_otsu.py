#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 局部阈值分割; Otsu算法;
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-09-30
"""

import sys
import math
import numpy as np
import cv2 as cv


def calcGrayHist(image):
    rows, cols = image.shape
    grayHist = np.zeros([256], np.uint64)
    for idx_row in range(rows):
        for idx_col in range(cols):
            grayHist[image[idx_row][idx_col]] += 1
    
    return grayHist


def otsuThreshold(image):
    rows, cols = image.shape
    grayHist = calcGrayHist(image)
    uniformGrayHist = grayHist / float(rows*cols)
    # 零阶累积矩阵和一阶累积矩阵
    zeroCumuMoment = np.zeros([256], np.float32)
    oneCumuMoment = np.zeros([256], np.float32)
    for k in range(256):
        if k == 0:
            zeroCumuMoment[k] = uniformGrayHist[0]
            oneCumuMoment[k] = (k) * uniformGrayHist[0]
        else:
            zeroCumuMoment[k] = zeroCumuMoment[k-1] + uniformGrayHist[k]
            oneCumuMoment[k] = oneCumuMoment[k-1] + k * uniformGrayHist[k]
        
    # 计算类间方差
    variance = np.zeros([256], np.float32)
    for k in range(255):
        if zeroCumuMoment[k] == 0 or zeroCumuMoment[k] == 1:
            variance[k] = 0
        else:
            variance[k] = math.pow(oneCumuMoment[255]*zeroCumuMoment[k] - oneCumuMoment[k], 2) / (zeroCumuMoment[k]*(1.0-zeroCumuMoment[k]))

    # 找到阈值
    threshLoc = np.where(variance[0:255] == np.max(variance[0:255]))
    thresholdVal = threshLoc[0][0]
    thresholdImg = np.copy(image)
    thresholdImg[thresholdImg > thresholdVal] = 255
    thresholdImg[thresholdImg <= thresholdVal] = 0
    return thresholdVal, thresholdImg


# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or directory.")
            sys.exit()
        cv.imshow("OriginImage", image)

        thresholdVal, thresholdImg = otsuThreshold(image)
        print(f"The threshold value is {thresholdVal}")
        cv.imshow("ThresholdImage", thresholdImg)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print(f"Usage: python gaussBlur imageFile.")
