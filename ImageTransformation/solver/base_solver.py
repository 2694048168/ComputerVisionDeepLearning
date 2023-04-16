#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The base class for solver with PyTorch
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-04-11 UTC + 08:00, Chinese Standard Time(CST)
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


class BaseSolver(object):
    def __init__(self, opt):
        self.opt = opt
        self.scale = opt['scale'] # image super-resolution
        self.is_train = opt['is_train']
        self.use_chop = opt['use_chop'] # whether enable memory-efficient test
        self.self_ensemble = opt['self_ensemble'] # image super-resolution in inference phase

        # GPU verify
        self.use_gpu = torch.cuda.is_available()
        self.Tensor = torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor

        # for better training (Stablizeation and less GPU memory usage)
        self.last_epoch_loss = 1e8
        self.skip_threshold = opt['solver']['skip_threshold']
        # save GPU memory during training
        self.split_batch = opt['solver']['split_batch']

        # experimental dirs
        self.experiment_root = opt['path']['experiment_root']
        self.checkpoint_dir = opt['path']['epochs']
        self.records_dir = opt['path']['records']
        self.visual_dir = opt['path']['visual']

        # log and vis scheme
        self.save_ckp_step = opt['solver']['save_ckp_step']
        self.save_vis_step = opt['solver']['save_vis_step']

        self.best_epoch = 0
        self.current_epoch = 1
        self.best_pred = 0.0

    def feed_data(self, batch):
        pass

    def train_step(self):
        pass

    def test(self):
        pass

    def _forward_geometric_8(self, x, forward_function):
        pass

    def _overlap_crop_forward(self, upscale):
        pass

    def get_current_log(self):
        pass

    def get_current_visual(self):
        pass

    def get_current_learning_rate(self):
        pass

    def set_current_log(self, log):
        pass

    def update_learning_rate(self, epoch):
        pass

    def save_checkpoint(self, epoch, is_best):
        pass

    def load(self):
        pass

    def save_current_visual(self, epoch, iter):
        pass

    def save_current_log(self):
        pass

    def print_network(self):
        pass

    def get_network_description(self, network):
        '''Get the string name and total parameters of the Network or Model'''
        # # 多个 GPU 训练模型, 将模型设置为数据并行风格模型情况下
        if isinstance(network, torch.nn.DataParallel): 
            network = network.module

        str_name_model = str(network)
        # torch.numel(input); Returns the total number of elements in the input tensor.
        num_parameter = sum(map(lambda x: x.numel(), network.parameters()))

        return str_name_model, num_parameter
