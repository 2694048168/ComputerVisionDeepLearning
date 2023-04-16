#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The common image processing methods for Deep Learning Models with PyTorch
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-04-10 UTC + 08:00, Chinese Standard Time(CST)

# ==================================================
# Step 0. Common settings for images
# Step 1. Common settings for files and IO
# Step 2. Common image processing with numpy format
# ==================================================
"""

"""Python Packages Imports should be grouped in the following order:
1. standard library imports
2. related third party imports
3. local application/library specific imports
You should put a blank line between each group of imports.
"""
# ----- standard library -----
import os
import random

# ----- standard library -----
import numpy as np
import imageio
from tqdm import tqdm
import torch
# ----- custom library -----


# -------------------------------------
# Step 0. Common settings for images 
# image extensions format
image_extensions = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP"]

# images with binary saving format
binary_extensions = [".npy"]

# image benchmark dataset, e.g. image super-resolution
benchmark_datasets = ["Set5", "Set14", "BSD100", "Urban100", "Manga109", "DIV2K", "DF2K"]


# -------------------------------------------
# Step 1. Common settings for files and IO
def is_image_file(filename):
    return any(filename.endswith(img_extension) for img_extension in image_extensions)

def is_binary_file(filename):
    return any(filename.endswith(img_extension) for img_extension in binary_extensions)

def get_paths_from_images(path):
    assert os.path.isdir(path), "[Error] {} is not a valid folder.".format(path)

    images = []
    for dir_path, _, file_names in sorted(os.walk(path)):
        for file_name in sorted(file_names):
            if is_image_file(file_name):
                img_path = os.path.join(dir_path, file_name)
                images.append(img_path)

    assert images, "[{}] has no valid image file.".format(path)

    return images

def get_paths_from_binary(path):
    assert os.path.isdir(path), "[Error] {} is not a valid folder.".format(path)

    files = []
    for dir_path, _, file_names in sorted(os.walk(path)):
        for file_name in sorted(file_names):
            if is_binary_file(file_name):
                binary_path = os.path.join(dir_path, file_name)
                files.append(binary_path)

    assert files, "[{}] has no valid image file.".format(path)

    return files


def get_image_paths(data_type, dataroot):
    paths = None
    if dataroot is not None:
        if data_type == "img":
            paths = sorted(get_paths_from_images(dataroot))

        elif data_type == "npy":
            if dataroot.find("_npy") < 0: # determine whether there is "*_npy" folder
                old_folder = dataroot
                dataroot = dataroot + "_npy"
                if not os.path.exists(dataroot):
                    print("========> Creating binary files in {}".format(dataroot))
                    os.makedirs(dataroot, exist_ok=True)
                    img_paths = sorted(get_paths_from_images(old_folder))
                    path_bar = tqdm(img_paths)
                    for v in path_bar:
                        img = imageio.imread(v, pilmode="RGB")
                        ext = os.path.splitext(os.path.basename(v))[-1]
                        name_sep = os.path.basename(v.replace(ext, ".npy"))
                        np.save(os.path.join(dataroot, name_sep), img)

                else: # have been generated "npy" the binary file
                    print("========> Binary files alread exists in {}. Skip binary files generation.".format(dataroot))

            paths = sorted(get_paths_from_binary(dataroot))

        else:
            raise NotImplementedError("[Error] Data_Type {} is not implementation.".format(data_type))

    return paths


def find_benchmark_datasets(dataroot):
    benchmark_dataset_list = [dataroot.find(benchmark) >= 0 for benchmark in benchmark_datasets]

    if not sum(benchmark_dataset_list) == 0:
        benchmark_idx = benchmark_dataset_list.index(True)
        benchmark_dataset_name = benchmark_datasets[benchmark_idx]
    else:
        benchmark_dataset_name = "CustomImage"

    return benchmark_dataset_name


def read_img(path, data_type):
    """Read image by misc or from .npy and then return numpy float32, HWC, RGB, [0, 255] 

    Args:
        path (_type_): _description_
        data_type (_type_): _description_
    """
    if data_type == "img":
        img = imageio.imread(path, pilmode="RGB")
    elif data_type.find("npy") >= 0:
        img = np.load(path)
    else:
        raise NotImplementedError

    if img.ndim == 2: # single channel image (gray image)
        img = np.expand_dims(img, axis=2)

    return img


# ---------------------------------------------------
# Step 2. Common image processing with numpy format
def np2tensor(images, rgb_range):
    """Convert the numpy format images to Tensor format.

    Args:
        images (_type_): _description_
        rgb_range (_type_): _description_
    """
    def _np2tensor(img):
        # HWC ----> CHW
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor_img = torch.from_numpy(np_transpose).float()
        tensor_img.mul_(rgb_range / 255.)

        return tensor_img

    return [_np2tensor(_img) for _img in images]


# the input low-resolution Patch size for image super-resolution with scale
def get_patch(img_input, img_target, patch_size, scale):
    input_height, input_width = img_input.shape[:2]
    output_height, output_width = img_target.shape[:2]

    # random crop patch images using random.rand in the image pixel coordinate system
    # Obtain a random integer, the integer + patch_size no more than index range of image matrix, this completes the random crop.
    input_patch = patch_size # the input size is a square shape ?

    if input_height == output_height:
        target_patch = input_patch
        input_x = random.randrange(0, input_width - input_patch + 1)
        input_y = random.randrange(0, input_height - input_patch + 1)
        target_x, target_y = input_x, input_y
    else: # for image super-resolution
        target_patch = input_patch * scale
        input_x = random.randrange(0, input_width - input_patch + 1)
        input_y = random.randrange(0, input_height - input_patch + 1)
        target_x, target_y = scale * input_x, scale * input_y

    img_input = img_input[input_y: input_y + input_patch, input_x: input_x + input_patch, :]
    img_target = img_target[target_y: target_y + target_patch, target_x: target_x + target_patch, :]

    return img_input, img_target


def add_noise(img, noise="."):
    if noise != ".":
        noise_type = noise[0]
        noise_value = int(noise[1:])

        if noise_type == "G": # Gaussian distribution noise
            noises = np.random.normal(scale=noise_value, size=img.shape)
            noises = noises.round()
        elif noise_type == "S": # Poisson distribution noise
            noises = np.random.poisson(img * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)

        # add additive noise to image
        img_noise = img.astype(np.int16) + noises.astype(np.int16)
        img_noise = img_noise.clip(0, 255).astype(np.uint8)

        return img_noise

    else:
        return img


def augment_probability(img_list, flip=True, rot=True):
    # horizontal flip and ratate with random probability,
    # eliminate the image of the displacement, flip, rotate factors such as the influence of the results, 
    # keep the convolutional neural network features such as translation invariance
    horizontal_flip = flip and (random.random() < 0.5)
    vertical_flip = flip and (random.random() < 0.5)
    rotating_90 = rot and (random.random() < 0.5)

    def _augment_probability(img):
        if horizontal_flip:
            img = img[:, ::-1, :] 
        if vertical_flip:
            img = img[::-1, :, :]
        if rotating_90:
            img = img.transpose(1, 0, 2)

        return img

    return [_augment_probability(img) for img in img_list]


def modcrop(img_input, scale):
    """HR crop ???

    Args:
        img_input (_type_): _description_
        scale (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    
    img = np.copy(img_input)

    if img.ndim == 2: # gray image
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3: # RGB color image
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError("Wrong image dimension in channels {}".format(img.ndim))
    
    return img