#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The utiliy of useful function for Deep Learning Models with PyTorch
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-04-11 UTC + 08:00, Chinese Standard Time(CST)

# =================================================
# Step 0. Auxiliary functions or scripts
# Step 1. Create mkdir and mkdirs for folders and files
# Step 2. Image convert functions
# Step 2. Compute metrics PSNR and SSIM for image restoration
# =============================================================
"""

"""Python Packages Imports should be grouped in the following order:
1. standard library imports
2. related third party imports
3. local application/library specific imports
You should put a blank line between each group of imports.
"""
# ----- standard library -----
import os
import datetime
import math

# ----- standard library -----
import numpy as np
from PIL import Image
import cv2 as cv

# ----- custom library -----


# ----------------------------------------
# Step 0. Auxiliary functions or scripts
def get_timestamp():
    # year month day-hour minute second e.g. 20220409-160445
    return datetime.datetime.now().strftime(r"%Y%m%d-%H%M%S")


# ------------------------------------------------
# Step 1. mkdir and mkdirs for folders and files
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)

def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + "_archived" + get_timestamp()
        print("[Warning] Path {} already exists. Rename it to {}".format(path, new_name))
        os.rename(path, new_name)
    
    os.makedirs(path)


# ---------------------------------
# Step 2. image convert functions
def quantize_image(image, rgb_range):
    pixel_range = 255. / rgb_range
    return image.mul(pixel_range).clamp(0, 255).round()


def tensor2np(tensor_list, rgb_range):
    def _tensor2numpy(tensor, rgb_range):
        array_np = np.transpose(quantize_image(tensor, rgb_range).numpy(), (1, 2, 0)).astype(np.uint8)
        return array_np

    return [_tensor2numpy(tensor, rgb_range) for tensor in tensor_list]


def rgb2ycbcy(img, only_y=True):
    """Color Space convert [RGB] ---> [YCbCr] for image super-resolution to compute PSNR and SSIM

    Args:
        img (_type_): image with uint8, [0, 255] or float, [0, 1]
        only_y (bool, optional): only return Y channel for YCbCr space. Defaults to True.

    Returns:
        _type_: only return image Y channel
    """
    input_img_type = img.dtype
    img.astype(np.float32)
    if input_img_type != np.uint8:
        img *= 255.
    
    # convert RGB to YCbCr 
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]

    if input_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.

    return rlt.astype(input_img_type)


def ycbcr2rgb(img):
    """Color Space convert [YCbCr] ---> [RGB] same as matlab ycbcr2rgb

    Args:
        img (_type_): image with uint8, [0, 255] or float, [0, 1]
        only_y (bool, optional): only return Y channel for YCbCr space. Defaults to True.

    Returns:
        _type_: only return image Y channel
    """
    input_img_type = img.dtype
    img.astype(np.float32)
    if input_img_type != np.uint8:
        img *= 255.

    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                           [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]

    if input_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.

    return rlt.astype(input_img_type)


def save_img_np(img_np, img_path, mode="RGB"):
    if img_np.ndim == 2:
        mode = "L"
    
    img_pil = Image.fromarray(img_np, mode=mode)
    img_pil.save(img_path)


# --------------------------------------------------------------
# Step 2. Compute metrics PSNR and SSIM for image restoration
def calculate_metrics(img1, img2, crop_border, test_Y=True):
    img1 = img1 / 255.
    img2 = img2 / 255.

    if test_Y and img1.shape[2] == 3: # evaluate on Y channel in YCbCr color space for RGB images
        img1_input = rgb2ycbcy(img1)
        img2_input = rgb2ycbcy(img2)
    else:
        img1_input = img1
        img2_input = img2

    height, width = img1.shape[:2]

    if img1_input.ndim == 3:
        cropped_img1 = img1_input[crop_border: height - crop_border, crop_border : width - crop_border, :]
        cropped_img2 = img2_input[crop_border: height - crop_border, crop_border : width - crop_border, :]
    elif img1_input.ndim == 2: # gray images
        cropped_img1 = img1_input[crop_border: height - crop_border, crop_border : width - crop_border]
        cropped_img2 = img2_input[crop_border: height - crop_border, crop_border : width - crop_border]
    else:
        raise ValueError("Wrong image dimension: {} Should be 2 or 3".format(img1_input.ndim))

    # custom function to compute PSNR and SSIM
    psnr = calculate_psnr(cropped_img1 * 255, cropped_img2 * 255)
    ssim = calculate_ssim(cropped_img1 * 255, cropped_img2 * 255)

    return psnr, ssim


def calculate_psnr(img1, img2):
    """img1 and img2 have range [0, 255]

    Args:
        img1 (_type_): _description_
        img2 (_type_): _description_
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)

    if mse == 0:
        return float("inf")
    
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    """calculate SSIM the same outputs as MATLAB's for img1, img2: [0, 255]

    Args:
        img1 (_type_): _description_
        img2 (_type_): _description_
    """
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")

    if img1.ndim == 2:
        return ssim(img1, img2) # custom function to compute SSIM for window
    elif img1.ndim == 3:
        # 3 channels images
        if img1.shape[2] == 3:
            ssims = []
            for _ in range(3):
                ssims.append(ssim(img1, img2))
            
            return np.array(ssims).mean()

        # 1 channel images : gray images
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")


def ssim(img1, img2):
    """https://en.wikipedia.org/wiki/Structural_similarity"""
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()