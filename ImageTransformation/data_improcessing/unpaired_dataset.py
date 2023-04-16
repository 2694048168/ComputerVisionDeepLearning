#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The Unpaired dataset for Deep Learning Models with PyTorch
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
from data_improcessing import common


class UnpairedDataset(torch.utils.data.Dataset):
    """Read Unpaired images only for inference or test phase.

    Args:
        torch (_type_): _description_
    """
    def name(self):
        return common.find_benchmark_datasets(self.opt["InputImages"])

    def __init__(self, opt_dataset):
        super(UnpairedDataset, self).__init__()
        self.opt = opt_dataset
        self.scale = self.opt["scale"] # scale for image super-resolution
        self.paths_unpaired = None

        # read image list from image/binary(npy format) files
        self.paths_unpaired = common.get_image_paths(self.opt["data_type"], self.opt["InputImages"])
        assert self.paths_unpaired, "[Error] Unpaired paths are empty."

    def __getitem__(self, idx):
        """Overwritting __getitem__ method of the base class torch.utils.data.Dataset. 

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        unpaired_path = self.paths_unpaired[idx]
        unpaired_img = common.read_img(unpaired_path, self.opt["data_type"])
        unpaired_tenor = common.np2tensor([unpaired_img], self.opt["rgb_range"])[0]
        return {"input_img": unpaired_tenor, "input_path": unpaired_path}

    def __len__(self):
        """Overwritting __len__ method of the base class torch.utils.data.Dataset. 
        """
        return len(self.paths_unpaired)