#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The Degradation Process to Estimate the Real-World for Image Restoration
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-04-30 UTC + 08:00, Chinese Standard Time(CST)

# ==================================================
# Step 0. Import the Degaradation Model from Paper
# ==================================================
"""

"""Python Packages Imports should be grouped in the following order:
1. standard library imports
2. related third party imports
3. local application/library specific imports
You should put a blank line between each group of imports.
"""
# ----- standard library -----
import datetime
import argparse
import random

# ----- standard library -----
import torch
import numpy as np
from tqdm import tqdm

# ----- custom library -----
import options.options_configure as option
from data_improcessing import create_dataset
from data_improcessing import create_dataloader
from solver import create_solver
from utils import utiliy

from utils import utils_blindsr as blindsr
# ----------------------------------------------
from glob import glob
from flags import *
from scipy import misc
import imageio
from multiprocessing.dummy import Pool as ThreadPool
# -------------------------------------------------------

starttime = datetime.datetime.now()

save_HR_path = os.path.join(save_dir, 'HR_x4')
save_LR_path = os.path.join(save_dir, 'LR_x4')
os.mkdir(save_HR_path)
os.mkdir(save_LR_path)
file_list = sorted(glob(os.path.join(train_HR_dir, '*.png')))
HR_size = [100, 0.8, 0.7, 0.6, 0.5]


def save_HR_LR(img, size, path, idx):
	HR_img = misc.imresize(img, size, interp='bicubic')
	HR_img = modcrop(HR_img, 4)
	rot180_img = misc.imrotate(HR_img, 180)
	x4_img = misc.imresize(HR_img, 1 / 4, interp='bicubic')
	x4_rot180_img = misc.imresize(rot180_img, 1 / 4, interp='bicubic')

	img_path = path.split('/')[-1].split('.')[0] + '_rot0_' + 'ds' + str(idx) + '.png'
	rot180img_path = path.split('/')[-1].split('.')[0] + '_rot180_' + 'ds' + str(idx) + '.png'
	x4_img_path = path.split('/')[-1].split('.')[0] + '_rot0_' + 'ds' + str(idx) + '.png'
	x4_rot180img_path = path.split('/')[-1].split('.')[0] + '_rot180_' + 'ds' + str(idx) + '.png'

	misc.imsave(save_HR_path + '/' + img_path, HR_img)
	misc.imsave(save_HR_path + '/' + rot180img_path, rot180_img)
	misc.imsave(save_LR_path + '/' + x4_img_path, x4_img)
	misc.imsave(save_LR_path + '/' + x4_rot180img_path, x4_rot180_img)


def modcrop(image, scale=4):
	if len(image.shape) == 3:
		h, w, _ = image.shape
		h = h - np.mod(h, scale)
		w = w - np.mod(w, scale)
		image = image[0:h, 0:w, :]
	else:
		h, w = image.shape
		h = h - np.mod(h, scale)
		w = w - np.mod(w, scale)
		image = image[0:h, 0:w]
	return image


def main(path):
	print('Processing-----{}/0800'.format(path.split('/')[-1].split('.')[0]))
	img = imageio.imread(path)
	idx = 0
	for size in HR_size:
		save_HR_LR(img, size, path, idx)
		idx += 1

items = file_list
pool = ThreadPool()
pool.map(main, items)
pool.close()
pool.join()
endtime = datetime.datetime.now()
print((endtime - starttime).seconds)