#!/usr/bin/env python3
# encoding: utf-8

"""
@Funciton: 局部阈值分割; 信息熵技术;
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-09-28
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


def thresholdEntropy(image):
    rows, cols = image.shape
    grayHist = calcGrayHist(image)
    # 归一化灰度直方图，即概率直方图
    normGrayHist = grayHist / float(rows*cols)

    # 1. 第一步，计算累加直方图，也称零阶累积矩
    zeroCumuMonment = np.zeros([256], np.float32)
    for k in range(256):
        if k == 0:
            zeroCumuMonment[k] = normGrayHist[k]
        else:
            zeroCumuMonment[k] = zeroCumuMonment[k - 1] + normGrayHist[k]

    # 2. 第二步，计算每一个灰度级的熵
    entropy = np.zeros([256], np.float32)
    for k in range(256):
        if k == 0:
            if normGrayHist[k] == 0:
                entropy[k] = 0
            else:
                entropy[k] = - normGrayHist[k] * math.log10(normGrayHist[k])
        else:
            if normGrayHist[k] == 0:
                entropy[k] = entropy[k-1]
            else:
                entropy[k] = entropy[k-1] - normGrayHist[k]*math.log10(normGrayHist[k])

    # 3. 第三步，找阈值
    findThresholds = np.zeros([256], np.float32)
    findThreshold_1, findThreshold_2 = 0.0, 0.0 
    totalEntropy = entropy[255]
    for k in range(255):
        # 找最大值
        maxFront = np.max(normGrayHist[0:k+1])
        maxBack = np.max(normGrayHist[k+1:256])
        if (maxFront == 0 or zeroCumuMonment[k]==0 or maxFront==1 or zeroCumuMonment[k]==1 or totalEntropy==0):
            findThreshold_1 = 0
        else:
            findThreshold_1 = entropy[k] / totalEntropy*math.log10(zeroCumuMonment[k]) / math.log10(maxFront)
        if (maxBack==0 or 1 - zeroCumuMonment[k]==0 or maxBack==1 or 1-zeroCumuMonment[k]==1):
            findThreshold_2 = 0
        else:
            if totalEntropy==0:
                findThreshold_2 = (math.log10(1-zeroCumuMonment[k])/math.log10(maxBack))
            else:
                findThreshold_2 = (1-entropy[k]/totalEntropy)*(math.log10(1-zeroCumuMonment[k])/math.log10(maxBack))
        findThresholds[k] = findThreshold_1 + findThreshold_2
    # 找到最大值的索引，作为获取的阈值
    thresholdLoc = np.where(findThresholds==np.max(findThresholds))
    thresholdVal = thresholdLoc[0][0]
    # 进行阈值处理
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

        thresholdVal, thresholdImg = thresholdEntropy(image)
        print(f"The threshold value is {thresholdVal}")
        cv.imshow("ThresholdImage", thresholdImg)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print(f"Usage: python gaussBlur imageFile.")
