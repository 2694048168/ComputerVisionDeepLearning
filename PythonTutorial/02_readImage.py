#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 02_readImage.py
@Python Version: 3.12.1
@Author: Wei Li (Ithaca)
@Email: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/
@Date: 2024-03-03
@copyright Copyright (c) 2024 Wei Li
@Description: 利用 OpenCV 图像处理库加载并读取本地图像
'''

import os
import glob
import tqdm
import cv2 as cv


def readImageFolder(original_folder):
    """从本地磁盘文件夹中读取图像文件到列表,
        支持的图像格式后缀: png, jpg, jpeg, bmp, tiff, gif, webp

    Args:
        original_folder (string): 文件夹路径

    Returns:
        list: 图像文件列表
    """
    image_list = []
    for extension in ["png", "jpg", "jpeg", "bmp", "tiff", "gif", "webp"]:
        image_list += glob.glob(os.path.join(original_folder, f"*.{extension}"))
    assert len(image_list), f"there is not any image in the {original_folder}!"

    for file in tqdm.tqdm(image_list):
        img = cv.imread(file, flags=cv.IMREAD_UNCHANGED) # BGR or BGRA
        basename = os.path.basename(file)

        cv.imshow(basename, img)
        cv.waitKey()
        print(f"--->finished processing for the image {basename}.")
    print(f"--->finished processing for all image!")


# --------------------------
if __name__ == "__main__":
    original_folder = R"D:/Development/PythonTutorial/images/"
    image_list = readImageFolder(original_folder)
