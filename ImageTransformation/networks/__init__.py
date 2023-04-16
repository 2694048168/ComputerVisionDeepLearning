#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The initialization of Model Methods to image transformation processing with PyTorch
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-04-12 UTC + 08:00, Chinese Standard Time(CST)

# ==============================================================
# Step 0. initialization for weights ans bias of Module
# Step 1. define the Model according to image processing tasks
# ==============================================================
"""

"""Python Packages Imports should be grouped in the following order:
1. standard library imports
2. related third party imports
3. local application/library specific imports
You should put a blank line between each group of imports.
"""
# ----- standard library -----
import functools

# ----- third-party library -----
import torch


# --------------------------------------------------------
# Step 0. initialization for weights ans bias of Module
def weights_init_normal(module, mean_normal=0.0, std_normal=0.02):
    """以 正态分布 normal distribution 对 模块的 权重和偏置 进行初始化

    Args:
        module (_type_): _description_
        mean_normal (float, optional): _description_. Defaults to 0.0.
        std_normal (float, optional): _description_. Defaults to 0.02.
    """
    class_name = module.__class__.__name__

    # if the module is the Convolution or Transpose Convolution for two-dimensional Images
    if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        if class_name != "MeanShift":
            print("Initializing {} Module with Normal Distribution.".format(class_name))
            # Initializing the weights using normal distribution for Module
            torch.nn.init.normal_(module.weight.data, mean=mean_normal, std=std_normal)
            if module.bias is not None: # bias initialize to zeros
                module.bias.data.zero_()

    # if the module is the linear layer in PyTorch
    elif isinstance(module, (torch.nn.Linear)):
        torch.nn.init.normal_(module.weight.data, mean=mean_normal, std=std_normal)
        if module.bias is not None:
            module.bias.data.zero_()

    # if the module is the BatchNorm layer for two-dimensional Images in PyTorch
    elif isinstance(module, (torch.nn.BatchNorm2d)):
        torch.nn.init.normal_(module.weight.data, mean=1.0, std=std_normal)
        torch.nn.init.constant_(module.bias.data, 0.0)


def weights_init_kaiming(module, scale=1):
    """以 何凯明的方式 kaiming 对模块的 权重和偏置 进行初始化

    Args:
        module (_type_): _description_
        scale (int, optional): _description_. Defaults to 1.
    """
    class_name = module.__class__.__name__

    if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        if class_name != "MeanShift":
            print("Initializing {} Module with Kaiming way.".format(class_name))
            torch.nn.init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
            module.weight.data *= scale
            if module.bias is not None:
                module.bias.data.zero_()

    elif isinstance(module, (torch.nn.Linear)):
        torch.nn.init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
        module.weight.data *= scale
        if module.bias is not None:
            module.bias.data.zero_()

    elif isinstance(module, (torch.nn.BatchNorm2d)):
        torch.nn.init.constant_(module.weight.data, 1.0)
        module.weight.data *= scale
        torch.nn.init.constant_(module.bias, 0.0)


def weights_init_orthogonal(module, gain_orthogonal=1):
    """Using orthogonal matrix, as described in 
    Exact solutions to the nonlinear dynamics of learning in deep linear neural networks (2013).

    Args:
        module (_type_): _description_
    """
    class_name = module.__class__.__name__

    if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        if class_name != "MeanShift":
            print("Initializing {} Module with Orthogonal Matrix.".format(class_name))
            torch.nn.init.orthogonal_(module.weight.data, gain=gain_orthogonal)
            if module.bias is not None:
                module.bias.data.zero_()

    elif isinstance(module, (torch.nn.Linear)):
        torch.nn.init.orthogonal_(module.weight.data, gain=gain_orthogonal)
        if module.bias is not None:
            module.bias.data.zero_()

    elif isinstance(module, (torch.nn.BatchNorm2d)):
        torch.nn.init.normal_(module.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(module.bias, 0.0)


def init_weights(net, init_type="kaiming", scale=1, mean_norm=0.0, std_norm=0.02, gain_orthogonal=1):
    """Chose init way in "kaiming", "normal" or "orthogonal".

    Args:
        net (_type_): _description_
        init_type (str, optional): _description_. Defaults to "kaiming".
        scale (int, optional): _description_. Defaults to 1.
        mean_norm (float, optional): _description_. Defaults to 0.0.
        std_norm (float, optional): _description_. Defaults to 0.02.
        gain_orthogonal (int, optional): _description_. Defaults to 1.
    """
    print("Using Initialization method {}.".format(init_type))

    if init_type == "normal":
        weights_init_normal_ = functools.partial(weights_init_normal, mean_normal=mean_norm, std_normal=std_norm)
        net.apply(weights_init_normal_)
    elif init_type == "kaiming":
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == "orthogonal":
        weights_init_orthogonal_ = functools.partial(weights_init_orthogonal, gain_orthogonal=gain_orthogonal)
        net.apply(weights_init_orthogonal_)
    else:
        raise NotImplementedError("Initialization method {} is not implemented.".format(init_type))


# ----------------------------------------------------------------
# Step 1. define the Model according to image processing tasks
def create_model(opt):
    """Creating Model accorading to the configure files.

    Args:
        opt (_type_): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    if opt["mode"] == "SISR":
        # define_networks : custom function to choose SISR methods or models.
        model = define_networks(opt["networks"])
        return model
    else:
        raise NotImplementedError("The mode {} of networks is not Implemented.".format(opt['mode']))


def define_networks(opt):
    """Choose one Model for SISR accorading to configure file.

    Args:
        opt (_type_): _description_
    """
    which_model = opt["which_model"].upper()
    print("========> Building Model Methods {}".format(which_model))

    if which_model == "ESPCN":
        from .sisr_espcn import ESPCN
        # list of parameters are same as the __init__ the constructor for ESPCN class.
        net = ESPCN()

    elif which_model.find("EDSR") >= 0:
        from .sisr_edsr import EDSR
        # list of parameters are same as the __init__ the constructor for EDSR class.
        net = EDSR(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                        num_features=opt['num_features'], num_blocks = opt['num_blocks'], res_scale=opt['res_scale'], upscale_factor=opt['scale'])

    elif which_model.find("SRCNN") >= 0:
        from .sisr_srcnn import SRCNN
        # list of parameters are same as the __init__ the constructor for SRCNN class.
        net = SRCNN()

    else:
        raise NotImplementedError("Model method {} is not Implemented.".format(which_model))

    if torch.cuda.is_available():
        # 如果有可用的 GPU, 则使用 GPU 进行训练, 并支持 Multi-GPU 进行数据并行训练
        net = torch.nn.DataParallel(net).cuda()

    return net