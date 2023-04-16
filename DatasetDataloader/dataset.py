#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: dataset.py
@Python Version: 3.11.2
@Platform: PyTorch 2.0.0+cu118
@Author: Wei Li
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2023/04/04 21:08:45
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Description: Create Super-Resolution Dataset.
'''

import os
import glob
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# 继承并重写Dataset抽象类的三个函数
class CustomDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __len__(self):
        num_sample = len()
        return num_sample
"""PyTorch Dataset Standard Pipeline
只需要根据算法模型的数据读取, 分别往 __init__()、__getitem__() 和 __len__() 
三个方法里添加数据的读取逻辑即可, PyTorch 数据读取范式以及 Dataloader, 三个方法缺一不可.

注:
__init__() 用于初始化数据读取逻辑, 如读取label和image的csv文件、定义transform组合等;
__getitem__() 用来返回数据和label, 单个样本数据(single sample), 被后续 Dataloader 调用;
__len__() 则用于返回样本数量, 便于 Dataloader 中计算 BatchSize;
"""

# 根据 Dataloader 获取满足模型的输入数据
# 可以查看 Dataloader 的 signature(函数签名)
# Python 中 class and function 习惯
def get_dataset(dataset, mode, batch_size):
    if mode == "train":
        train_dataset = DataLoader(dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=0,
                                   pin_memory=True, 
                                   drop_last=False)
        return train_dataset
    elif mode == "validaton" or "test":
        val_dataset = DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=False,
                                 drop_last=False)
        return val_dataset
    else:
        raise NotImplementedError(
            f"----[Error] Data mode {mode} is not implementation.")


# Image Transformation Processing
class ImageDataset(Dataset):
    def __init__(self, path2input, path2gt, transform=None, scale=4):
        super().__init__()

        assert os.path.isdir(path2input), \
            f"[Error] {path2input} is not a valid folder."
        
        assert os.path.isdir(path2gt), \
            f"[Error] {path2gt} is not a valid folder."
        
        # list to load image data form folder on dist
        # Host RAM memory? and pick into binary format?
        # 实际的数据读取到内存, 在构造函数完成, 还是在 __getitem__ 完成？
        self.input_image = list()
        self.ground_true = list()
        img_extension = ["png", "jpg", "jpeg", "bmp", "tiff"]

        for extension in img_extension:
            self.input_image += glob.glob(os.path.join(path2input, f"*.{extension}"))
        assert len(self.input_image), f"there is not any image in the {path2input}!"
        
        for extension in img_extension:
            self.ground_true += glob.glob(os.path.join(path2gt, f"*.{extension}"))
        assert len(self.ground_true), f"there is not any image in the {path2gt}!"

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        input_path = self.input_image[index]
        gt_path = self.ground_true[index]

        # each sample dimension is different?
        # RuntimeError: stack expects each tensor to be equal size
        # 'collate_fn' in Dataloader
        img_input, img_gt = get_patch(Image.open(input_path),
                                      Image.open(gt_path),
                                      patch_size=57,
                                      scale=4)
        sample_input = self.transform(img_input)
        sample_gt = self.transform(img_gt)

        return sample_input, sample_gt
        # return {"LR": sample_input, "HR": sample_gt}

    def __len__(self):
        return len(self.ground_true)

""" each sample dimension is different?
RuntimeError: stack expects each tensor to be equal size
the input low-resolution Patch size for image super-resolution with scale.
We can process this process via 'collate_fn' in Dataloader
---------------------------------------------------------- """
def get_patch(img_input, img_target, patch_size, scale):
    img_input, img_target = np.array(img_input), np.array(img_target)
    input_height, input_width = img_input.shape[:2]
    output_height, output_width = img_target.shape[:2]

    """ random crop patch images using random.rand in the image pixel 
    coordinate system; Obtain a random integer, the integer + patch_size 
    no more than index range of image matrix, this completes the random crop.
    ---------------------------------------------------------------------- """
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


# -------------------------
if __name__ == "__main__":
    path2lr = r"D:\Datasets\DIV2K_valid_LR"
    path2hr = r"D:\Datasets\DIV2K_valid_HR"
    sr_dataset = ImageDataset(path2lr, path2hr)

    train_dataset = get_dataset(sr_dataset, mode="train", batch_size=16)

    for train_samples in train_dataset:
        img_input, img_gt = train_samples
        print(f"the input shape of Model: {img_input.shape}\n")
        print(f"the label shape of Model: {img_gt.shape}\n")
        break
