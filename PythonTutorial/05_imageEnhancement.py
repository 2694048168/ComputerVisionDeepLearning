#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 05_imageEnhancement.py
@Python Version: 3.12.1
@Author: Wei Li (Ithaca)
@Email: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/
@Date: 2024-03-06
@copyright Copyright (c) 2024 Wei Li
@Description: 利用 OpenCV 和 Numpy 库进行图像对比度增强
'''

import os
import glob
import tqdm
import cv2 as cv
import numpy as np


class ImageProcessing():
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

    def imageEnhancement(self, scale = 1.2):
        """针对输入的图像进行对比度增强

        Args:
            scale (int, optional): 图像的伽马变换数值. Defaults to 1.2
        """
        for file in tqdm.tqdm(self.m_image_list):
            img = cv.imread(file, flags=cv.IMREAD_UNCHANGED)

            # 1. 线性变换进行对比度增强       
            # 注意数据存储的格式，uint8, float64, 以及图像本身的能够容纳的深度(8bit)
            output_img = float(scale) * img
            # 进行像数值截断，饱和操作
            output_img[output_img > 255] = 255
            output_img = output_img.astype(np.uint8)

            basename = os.path.basename(file)
            cv.imwrite(os.path.join(self.m_save_folder, basename), output_img)
        print(f"--->finished downsampling processing for all image and saving into {self.m_save_folder}!")


# --------------------------
if __name__ == "__main__":
    original_folder = R"D:\Development\GitRepository\PythonTutorial/images/"
    save_folder = R"D:\Development\GitRepository\PythonTutorial/enhancementImages/"
    os.makedirs(save_folder, exist_ok=True)

    sharpen_algorithm = ImageProcessing(original_folder, save_folder)
    sharpen_algorithm.readImageFolder()
    sharpen_algorithm.imageEnhancement(scale=1.6)
