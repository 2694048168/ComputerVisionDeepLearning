#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: inference.py
@Python Version: 3.11.2
@Platform: PyTorch 2.0.0+cu118
@Author: Wei Li
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2023/04/04 22:12:17
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Description: Inference phase the SRCNN model.
'''

import os
import glob
import torch
from PIL import Image
import tqdm
from torchvision.transforms import ToTensor, ToPILImage

from model import SRCNN


def inference(checkpoint, file, save_folder):
    model = SRCNN()
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    lr_img = Image.open(file)
    lr_img = ToTensor()(lr_img).view(1, -1, lr_img.size[1], lr_img.size[0])

    sr_img = model(lr_img).cpu()
    sr_img = sr_img[0].detach() # [1, C, H, W] ---> [C, H, W]
    img = ToPILImage()(sr_img) # [C, H, W] ---> [H, W, C]

    basename = "SR_" + os.path.basename(file)
    img.save(os.path.join(save_folder, basename))
    print(f"====> Saving the SR image into {save_folder}.")


# ------------------------
if __name__ == "__main__":
    input_folder = r"./dataset/super_resolution/test/LR"
    save_folder = r"./results/"
    os.makedirs(save_folder, exist_ok=True)

    epoch = 30
    checkpoint_path = f"./checkpoints/SRCNN_epoch_{epoch}.pth"

    image_list = []
    for extension in ["png", "jpg", "jpeg", "bmp", "tiff"]:
        image_list += glob.glob(os.path.join(input_folder, f"*.{extension}"))
    assert len(image_list), f"there is not any image in the {input_folder}!"

    for file in tqdm.tqdm(image_list):
        inference(checkpoint_path, file, save_folder)
