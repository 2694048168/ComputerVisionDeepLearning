#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 局部阈值分割; 直方图技术；
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-09-28
"""

import sys
import numpy as np
import cv2 as cv


def calcGrayHist(image):
    rows, cols = image.shape
    grayHist = np.zeros([256], np.uint64)
    for idx_row in range(rows):
        for idx_col in range(cols):
            grayHist[image[idx_row][idx_col]] += 1
    
    return grayHist

def threshTwoPeaks(image):
    histogram = calcGrayHist(image)
    # 找到灰度直方图的最大峰值对应的灰度值
    maxLoc = np.where(histogram==np.max(histogram))
    firstPeak = maxLoc[0][0]
    # 寻找灰度直方图的第二个峰值对应的灰度图
    measureDists = np.zeros([256], np.float32)
    for k in range(256):
        measureDists[k] = pow(k - firstPeak,  2) * histogram[k]
    maxLoc2 = np.where(measureDists==np.max(measureDists))
    secondPeak = maxLoc2[0][0]
    # 找到两个峰值之间的最小值对应的灰度值,作为阈值
    threshold  = 0
    # 第一个峰值在第二个峰值的右侧
    if firstPeak > secondPeak:
        temp = histogram[int(secondPeak):int(firstPeak)]
        minLoc = np.where(temp==np.min(temp))
        threshold = secondPeak + minLoc[0][0] + 1
    # 第一个峰值在第二个峰值的左侧
    else:
        temp = histogram[int(firstPeak):int(secondPeak)]
        minLoc = np.where(temp==np.min(temp))
        threshold = firstPeak + minLoc[0][0] + 1

    # 找到阈值后进行阈值处理，获得二值图
    thresholdImg = image.copy()
    thresholdImg[thresholdImg > threshold] = 255
    thresholdImg[thresholdImg <= threshold] = 0
    return (threshold, thresholdImg)


# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or directory.")
            sys.exit()
        cv.imshow("OriginImage", image)

        thresholdVal, thresholdImg = threshTwoPeaks(image)
        print(f"The threshold value is {thresholdVal}")
        cv.imshow("ThresholdImage", thresholdImg)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print(f"Usage: python gaussBlur imageFile.")
