#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 09_imageSuperResolution.py
@Python Version: 3.12.1
@Author: Wei Li (Ithaca)
@Email: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/
@Date: 2024-03-10
@copyright Copyright (c) 2024 Wei Li
@Description: 数字图像处理+卷积神经网络: 图像超分辨率重建 SRCNN
@Paper: Image Super-Resolution Using Deep Convolutional Networks
@Link: https://arxiv.org/abs/1501.00092
'''

import os
import glob
import tqdm
import cv2 as cv
import numpy as np
from scipy import signal


class ImageSR():
    """利用预训练的 SRCNN 模型进行推理, 对图像进行超分辨率重建;
        计算重建后的图像 img_SR 和 img_HR 的 PSNR & SSIM;
    """
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

    

# --------------------------
if __name__ == "__main__":
    pass