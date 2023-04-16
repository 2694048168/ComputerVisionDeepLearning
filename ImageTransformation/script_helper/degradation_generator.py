#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The Degradation Process to Estimate the Real-World for Image Restoration
@Paper on arXiv: https://arxiv.org/abs/2103.14006
@Paper on ICCV: https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Designing_a_Practical_Degradation_Model_for_Deep_Blind_Image_Super-Resolution_ICCV_2021_paper.pdf
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-04-30 UTC + 08:00, Chinese Standard Time(CST)
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
from glob import glob
from multiprocessing.dummy import Pool as ThreadPool

# ----- standard library -----
import numpy as np
import imageio
from tqdm import tqdm

# ----- custom library -----
from utils import utils_blindsr as blindsr


def degradation_model(image, scale_factor, shuffle_probability, use_sharp, LR_patchsize, isp_model):

    return img_LR, img_HR = blindsr.degradation_bsrgan_plus(img=image, sf=scale_factor, shuffle_prob=shuffle_probability,
        use_sharp=use_sharp, lq_patchsize=LR_patchsize, isp_model=isp_model)

# ---------------------------
if __name__ == "__main__":
