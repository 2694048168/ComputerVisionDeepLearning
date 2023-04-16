#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: ESPCN: Image and Video Super-Resolution with PyTorch
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-04-13 UTC + 08:00, Chinese Standard Time(CST)

# ==================================================================================================================
@Paper: Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
@Year: 2016
@Author: Wenzhe Shi, Jose Caballero, Ferenc Huszár, Johannes Totz, 
        Andrew P. Aitken, Rob Bishop, Daniel Rueckert, Zehan Wang
@Publisher: IEEE of Computer Vision and Pattern Recognition (CVPR)
# ==================================================================================================================
@arXiv: https://arxiv.org/abs/1609.05158
# ==================================================================================================================
"""

"""Python Packages Imports should be grouped in the following order:
1. standard library imports
2. related third party imports
3. local application/library specific imports
You should put a blank line between each group of imports.
"""
# ----- standard library -----
# ----- standard library -----
import torch
# pip install torchkeras
import torchkeras

# ----- custom library -----


class ESPCN(torch.nn.Module):
    """Model Framework, e.g. Fig.1. in the Paper.
    Figure 1. The proposed efficient sub-pixel convolutional neural network (ESPCN), 
    with two convolution layers for feature maps extraction,
    and a sub-pixel convolution layer that aggregates the feature maps 
    from LR space and builds the SR image in a single step.

    Args:
        torch (_type_): _description_
    """
    def __init__(self, upscale_factor):
        super(ESPCN, self).__init__()
        self.conv_1 = torch.nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv_2 = torch.nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv_3 = torch.nn.Conv2d(32, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = torch.nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = torch.tanh(self.conv_1(x))
        x = torch.tanh(self.conv_2(x))
        y = torch.sigmoid(self.pixel_shuffle(self.conv_3(x)))
        
        return y


# ------------------------------------------------
# 检查一下网络结构是否正确, e.g. Figure 1 in Paper
# ------------------------------------------------
if __name__ == "__main__":
    model = ESPCN(upscale_factor=4)
    print("======== The Summary of EDSR Model ========")
    print(model)
    print("======== The Summary of EDSR Model ========")
    torchkeras.summary(model, input_shape=(1, 64, 64))
# ---------------------------------------------------------------
# python networks/sisr_espcn.py | tee networks/ESPCN_model.txt
# ---------------------------------------------------------------