#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 08_laplaceOperator.py
@Python Version: 3.12.1
@Author: Wei Li (Ithaca)
@Email: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/
@Date: 2024-03-09
@copyright Copyright (c) 2024 Wei Li
@Description: 图像边缘检测之拉普拉斯算子,水墨效果的边缘图
'''

import os
import glob
import tqdm
import cv2 as cv
import numpy as np
from scipy import signal


class EdgeDetect():
    def __init__(self, original_folder, save_folder):
        self.m_image_list = []
        self.m_original_folder = original_folder
        self.m_save_folder = save_folder

    def readImageFolder(self):
        """从本地磁盘文件夹中读取图像文件到列表,
            支持的图像格式后缀: png, jpg, jpeg, bmp, tiff, gif, webp

        Args:
            original_folder (string): 文件夹路径
            image_list: 图像文件列表
        """
        for extension in ["png", "jpg", "jpeg", "bmp", "tiff"]:
            self.m_image_list += glob.glob(os.path.join(self.m_original_folder, f"*.{extension}"))
        assert len(self.m_image_list), f"there is not any image in the {self.m_original_folder}!"

    def laplaceOperator(self, image, _boundary="fill", _fillvalue=0):
        # laplace convolution kernel
        laplace_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32)
        # laplace_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32)

        img_laplace_conv = signal.convolve2d(image, laplace_kernel, mode="same", boundary=_boundary, fillvalue=_fillvalue)

        return img_laplace_conv

    def imageEdgeDetect(self):
        """针对输入的图像进行拉普拉斯算子操作"""
        for file in tqdm.tqdm(self.m_image_list):
            image = cv.imread(file, flags=0)
            img_laplce_conv = self.laplaceOperator(image, "symm")
            # case 1, 阈值化处理
            thresholdEdge = np.copy(img_laplce_conv)
            thresholdEdge[thresholdEdge>0] = 255
            thresholdEdge[thresholdEdge<0] = 0
            thresholdEdge = thresholdEdge.astype(np.uint8)

            # case 2, 抽象化处理(水墨画效果)
            asbstractionEdge = np.copy(img_laplce_conv)
            asbstractionEdge = asbstractionEdge.astype(np.float32)
            asbstractionEdge[asbstractionEdge>=0] = 1.0
            asbstractionEdge[asbstractionEdge<0] = 1.0 + np.tanh(asbstractionEdge[asbstractionEdge<0])

            cv.imshow("AsbstractionEdge", asbstractionEdge)
            cv.waitKey(0)
        print(f"--->finished downsampling processing for all image and saving into {self.m_save_folder}!")


# --------------------------
if __name__ == "__main__":
    original_folder = R"D:\Development\GitRepository\PythonTutorial/images/"
    save_folder = R"D:\Development\GitRepository\PythonTutorial/laplaceImages/"
    os.makedirs(save_folder, exist_ok=True)

    sharpen_algorithm = EdgeDetect(original_folder, save_folder)
    sharpen_algorithm.readImageFolder()
    sharpen_algorithm.imageEdgeDetect()
