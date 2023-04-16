#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: SRCNN: Image Super-Resolution with PyTorch
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-04-13 UTC + 08:00, Chinese Standard Time(CST)

# =================================================================================
@Paper: Image super-resolution using deep convolutional networks
@Year: 2015
@Author: Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang
@Publisher: IEEE transactions on pattern analysis and machine intelligence (TPAMI)
# =================================================================================
@Paper: Learning a deep convolutional network for image super-resolution
@Year: 2014
@Author: Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang
@Publisher: Springer, Cham. European conference on computer vision (ECCV)
# =================================================================================
@arXiv: https://arxiv.org/abs/1501.00092
# =================================================================================
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


class SRCNN(torch.nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = torch.nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = torch.nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# ------------------------------------------------
# 检查一下网络结构是否正确, e.g. Figure 1 in Paper
# ------------------------------------------------
if __name__ == "__main__":
    model = SRCNN()
    print("======== The Summary of EDSR Model ========")
    print(model)
    print("======== The Summary of EDSR Model ========")
    torchkeras.summary(model, input_shape=(1, 64, 64))
# ---------------------------------------------------------------
# python networks/sisr_srcnn.py | tee networks/SRCNN_model.txt
# ---------------------------------------------------------------