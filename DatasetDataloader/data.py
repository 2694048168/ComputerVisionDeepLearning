#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: data.py
@Python Version: 3.11.2
@Author: Wei Li
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2023/04/04 19:05:31
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Description: DIV2K dataset and generate LR version
'''

import os
import glob
import tqdm
import cv2
import datetime
from multiprocessing import Pool
from functools import partial


"""
enum InterpolationFlags
{
    INTER_NEAREST      = 0,
    INTER_LINEAR       = 1,
    INTER_CUBIC        = 2,
    INTER_AREA         = 3,
    INTER_LANCZOS4     = 4,
    INTER_MAX          = 7,
    WARP_FILL_OUTLIERS = 8,
    WARP_INVERSE_MAP   = 16,
};
"""
def image_downsampling(origin_folder, save_folder, scale=4):
    image_list = []
    for extension in ["png", "jpg", "jpeg", "bmp", "tiff"]:
        image_list += glob.glob(os.path.join(origin_folder, f"*.{extension}"))
    assert len(image_list), f"there is not any image in the {origin_folder}!"

    for file in tqdm.tqdm(image_list):
        img = cv2.imread(file, flags=cv2.IMREAD_UNCHANGED) # BGR or BGRA
        height, width, channels = img.shape
        if min(height, width) >= 2160: # the 4K spatial resolution
            img = cv2.GaussianBlur(img, ksize=(13, 13), sigmaX=0)
        img = cv2.resize(img, (width//scale, height//scale), cv2.INTER_CUBIC)

        basename = os.path.basename(file)
        cv2.imwrite(os.path.join(save_folder, basename), img)

        print(f"--->finished downsampling for the image {basename}.")

    print(f"--->finished downsampling for all image in {origin_folder} \
          and saving into {save_folder}!")


# 利用线程池并行处理大数据集
def downsampling_func(file, save_folder=None, scale=4):
    img = cv2.imread(file, flags=cv2.IMREAD_UNCHANGED) # BGR or BGRA
    height, width, channels = img.shape
    if min(height, width) >= 2160: # the 4K spatial resolution
        img = cv2.GaussianBlur(img, ksize=(13, 13), sigmaX=0)
    img = cv2.resize(img, (width//scale, height//scale), cv2.INTER_CUBIC)

    basename = os.path.basename(file)
    cv2.imwrite(os.path.join(save_folder, basename), img)

    print(f"--->finished downsampling for the image {basename}.")

# ------------------------
if __name__ == "__main__":
    origin_folder = r"D:\Datasets\DIV2K_valid_HR"
    save_folder = r"D:\Datasets\DIV2K_valid_LR"
    os.makedirs(save_folder, exist_ok=True)

    # -----------------------------------------------------
    starttime = datetime.datetime.now()
    img_list = []
    for extension in ["png", "jpg", "jpeg", "bmp", "tiff"]:
        img_list += glob.glob(os.path.join(origin_folder, f"*.{extension}"))
    assert len(img_list), f"there is not any image in the {origin_folder}!"

    with Pool() as processThread:
        processThread.map(partial(downsampling_func, save_folder=save_folder),
                          img_list)
    
    print(f"--->finished downsampling for all image in {origin_folder} \
          and saving into {save_folder}!")
    
    endtime = datetime.datetime.now()
    print(f"the Time Consumption is: {(endtime - starttime).seconds} seconds")
    # -----------------------------------------------------------------

    # # 感受一下线程池处理大数据集的魅力
    # starttime = datetime.datetime.now()
    # image_downsampling(origin_folder, save_folder)
    # endtime = datetime.datetime.now()
    # print(f"the Time Consumption is: {(endtime - starttime).seconds} seconds")
