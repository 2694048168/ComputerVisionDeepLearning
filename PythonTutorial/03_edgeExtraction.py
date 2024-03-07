#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 03_edgeExtraction.py
@Python Version: 3.12.1
@Author: Wei Li (Ithaca)
@Email: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/
@Date: 2024-03-04
@copyright Copyright (c) 2024 Wei Li
@Description: 利用 OpenCV 图像处理库进行图像锐利化处理
'''

import os
import glob
import tqdm
import cv2 as cv


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
        for extension in ["png", "jpg", "jpeg", "bmp", "tiff", "gif", "webp"]:
            self.m_image_list += glob.glob(os.path.join(self.m_original_folder, f"*.{extension}"))
        assert len(self.m_image_list), f"there is not any image in the {self.m_original_folder}!"

    def edgeSharpen(self):
        """从图像文件列表中读取图像, 边缘锐利化处理, 保存结果到文件夹中

        Args:
            original_folder (string): 原始图像文件夹路径
            save_folder (string): 保存结果图像的文件夹路径
        """
        for file in tqdm.tqdm(self.m_image_list):
            img_ = cv.imread(file, flags=cv.IMREAD_UNCHANGED)
            img_blur = cv.GaussianBlur(img_, (5, 5), 0)
            img = img_ - img_blur + img_
            basename = os.path.basename(file)
            cv.imwrite(os.path.join(self.m_save_folder, basename), img)
        print(f"--->finished processing for all image and saving into {self.m_save_folder}!")


# --------------------------
if __name__ == "__main__":
    original_folder = R"D:\Development\PythonTutorial/images/"
    save_folder = R"D:\Development\PythonTutorial/sharpenImages/"
    os.makedirs(save_folder, exist_ok=True)

    sharpen_algorithm = ImageProcessing(original_folder, save_folder)
    sharpen_algorithm.readImageFolder()
    sharpen_algorithm.edgeSharpen()
