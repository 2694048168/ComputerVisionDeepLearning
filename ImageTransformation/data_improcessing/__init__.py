#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The Dataset and DataLoader with PyTorch
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-04-13 UTC + 08:00, Chinese Standard Time(CST)
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

# ----- custom library -----


def create_dataset(opt_dataset):
    mode = opt_dataset["mode"].upper()
    if mode == "UNPAIRED": # only for test or inference phase
        from data_improcessing.unpaired_dataset import UnpairedDataset as D
    elif mode == "PAIRED":
        from data_improcessing.paired_dataset import PairedDataset as D
    else:
        raise NotImplementedError("Dataset {} is not implementation.".format(mode))

    dataset = D(opt_dataset)
    print("========> {} Dataset is created.".format(mode))

    return dataset


def create_dataloader(dataset, opt_dataset):
    phase = opt_dataset["phase"]
    if phase == "train":
        batch_size = opt_dataset["batch_size"]
        shuffle = True
        num_workers = opt_dataset["num_workers"]

    else:
        batch_size = 1
        shuffle = False
        num_workers = 0

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=num_workers, pin_memory=True)