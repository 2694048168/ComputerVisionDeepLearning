#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The Paired dataset for Deep Learning Models with PyTorch
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-04-10 UTC + 08:00, Chinese Standard Time(CST)
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
# python import packages the function of the import to write their own package and __init__.py
# directly to the root directory of the project is added to the system search path
# import sys
# sys.path.append("./")
# from function_packages_dir import function_file_name
# function_file_name.FunctionName ----> this method of use
from data_improcessing import common


class PairedDataset(torch.utils.data.Dataset):
    """Read Paired images for train phase and eval phase.

    Args:
        torch (_type_): _description_
    """
    def name(self):
        return common.find_benchmark_datasets(self.opt["GroundTrues"])

    def __init__(self, opt_dataset):
        super(PairedDataset, self).__init__()
        self.opt = opt_dataset
        self.train = (self.opt["phase"] == "train")
        self.split = "train" if self.train else "test" # test for paired images dataset
        self.scale = self.opt["scale"] # scale for image super-resolution
        self.paths_inputs, self.paths_groundtrue = None, None

        # change the integer of train dataset,
        # influence the number of iterations in each epoch.
        self.repeat_batch = 2

        # read image list from image/binary files.
        self.paths_groundtrue = common.get_image_paths(self.opt["data_type"], self.opt["GroundTrues"])
        self.paths_inputs = common.get_image_paths(self.opt["data_type"], self.opt["InputImages"])

        assert self.paths_groundtrue, "[Error] GroundTrue paths are empty."
        assert self.paths_inputs, "[Error] Input paths are empty."

        if self.paths_inputs and self.paths_groundtrue:
            assert len(self.paths_inputs) == len(self.paths_groundtrue), "[Error] GroundTrue: {} and Input {} have different number of images.".format(len(self.paths_groundtrue), len(self.paths_inputs))

    def __getitem__(self, idx):
        """Overwritting __getitem__ method of the base class torch.utils.data.Dataset. 

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        idx = self._get_index(idx)
        input_path = self.paths_inputs[idx]
        groundtrue_path = self.paths_groundtrue[idx]
        input_img = common.read_img(input_path, self.opt["data_type"])
        groundtrue = common.read_img(groundtrue_path, self.opt["data_type"])

        if self.train: # patch crop as the inputs of Model
            input_img_size = self.opt["input_img_size"]
            # random crop and augment
            input_img, groundtrue = common.get_patch(input_img, groundtrue, input_img_size, self.scale)
            input_img, groundtrue = common.augment_probability([input_img, groundtrue])
            input_img = common.add_noise(input_img, self.opt["noise"])

        input_tensor, groundtrue_tensor = common.np2tensor([input_img, groundtrue], self.opt["rgb_range"])

        return {"input_img": input_tensor, "groundtrue": groundtrue_tensor, "input_path": input_path, "groundtrue_path": groundtrue_path}

    def __len__(self):
        """Overwritting __len__ method of the base class torch.utils.data.Dataset. 
        """
        if self.train:
            return len(self.paths_groundtrue) * self.repeat_batch
        else: # for valid phase
            return len(self.paths_inputs)

    def _get_index(self, idx):
        # due to the training set the reuse a batch of data, e.g. self.repeat_batch, 
        # possible access to cross-border problems, solve the index using modulus calculation
        if self.train: 
            return idx % len(self.paths_groundtrue)
        else:
            return idx