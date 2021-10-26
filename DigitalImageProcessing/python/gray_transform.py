#!/usr/bin/env python3
# encoding: utf-8


"""
@Funciton: 线性变换进行对比度增强; 直方图正规化；
@Python Version: 3.8
@Author: Wei Li
@Date: 2021-09-16
"""

import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# --------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image = cv.imread(sys.argv[1], 0)
        if image is None:
            print(f"Error: no such file or dictory.")
        cv.imshow("OriginImage",image)
        # 2. 利用 matplotlib 计算直方图
        rows, cols = image.shape
        # 二维矩阵转换为一维数组
        pixelSequence = image.reshape([rows*cols, ])
        numberBins = 256 # 灰度等级
        histgram, bins, patch = plt.hist(pixelSequence, numberBins, facecolor="black", histtype="bar")
        plt.xlabel(u"OriginImgGrayLevel")
        plt.ylabel(u"Number of Pixels")
        plt.axis([0, 255, 0, np.max(histgram)])
        plt.show()
        plt.close()
        
        # 1. 线性变换进行对比度增强       
        scale = 1.2
        # 注意数据存储的格式，uint8, float64, 以及图像本身的能够容纳的深度(8bit)
        output_img = float(scale) * image
        # 进行像数值截断，饱和操作
        output_img[output_img > 255] = 255
        output_img = output_img.astype(np.uint8)

        cv.imshow("LinearTransorformImage", output_img)
        # 2. 利用 matplotlib 计算直方图
        rows, cols = output_img.shape
        # 二维矩阵转换为一维数组
        pixelSequence = output_img.reshape([rows*cols, ])
        numberBins = 256 # 灰度等级
        histgram, bins, patch = plt.hist(pixelSequence, numberBins, facecolor="black", histtype="bar")
        plt.xlabel(u"LinearTransorformImgGrayLevel")
        plt.ylabel(u"Number of Pixels")
        plt.axis([0, 255, 0, np.max(histgram)])
        plt.show()
        plt.close()

        # 3. 自适应选择线性变换的系数，直方图正规化方法
        # 计算图像像数值的最大值和最小值
        pixel_max = np.max(image)
        pixel_min = np.min(image)
        # 设置输出图像的最小灰度级和最大灰度级
        output_img_min = 0
        output_img_max = 255
        # 计算线性变换的系数
        a = float(output_img_max - output_img_min) / (pixel_max - pixel_min)
        b = output_img_min - a * pixel_min
        # 以矩阵形式进行线性变换
        norm_output_img = a * image + b
        norm_output_img = output_img.astype(np.uint8)
        cv.imshow("NormImage", norm_output_img)

        # 4. 利用 OpenCV 正规化函数实现直方图的正规化
        # 利用函数可以直接处理多通道矩阵(图像)
        dstNormImg = cv.normalize(image, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
        cv.imshow("NormalizeImage", dstNormImg)

        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print(f"Usage: python histogram imageFile.")
    