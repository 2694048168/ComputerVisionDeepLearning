#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: model.py
@Python Version: 3.11.2
@Platform: PyTorch 2.0.0+cu118
@Author: Wei Li
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2023/04/04 21:10:01
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Paper: https://arxiv.org/abs/1501.00092
@Description: SRCNN model for image Super-Resolution
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9//2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5//2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5//2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_bicubic = F.interpolate(x, scale_factor=4,
                          mode='bicubic', align_corners=False)
        
        x = self.relu(self.conv1(x_bicubic))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        sr_img = x_bicubic + x
        return sr_img


# -------------------------
if __name__ == "__main__":
    model = SRCNN()
    # num_param = sum(param.numel() for param in model.parameters())
    num_param = sum(param.numel() for param in model.parameters()
                    if param.requires_grad)
    print(f"The trainable parameters for model is: {num_param}")
    print(f"The trainable parameters for model is: {num_param/1000.0:.6f}K")
    print(f"The trainable parameters for model is: {num_param/1000000.0:.6f}M")

    lr_img = torch.randn((1, 3, 128, 128))
    sr_img = model(lr_img)
    print(f"the input image from SRCNN: {lr_img.shape}")
    print(f"the output type of SRCNN: {type(sr_img)}")
    print(f"the SR image from SRCNN: {sr_img.shape}")

    print("\n======== The Summary of SRCNN Model ========")
    print(model)
