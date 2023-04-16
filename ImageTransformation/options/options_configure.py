#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The options configure file for Deep Learning Models with PyTorch
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-04-10 UTC + 08:00, Chinese Standard Time(CST)

# =========================================
# Step 0. Auxiliary functions or scripts
# Step 1. Parse JSON configure file
# =========================================
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
import json
from collections import OrderedDict

# ----- standard library -----
import torch

# ----- custom library -----
from utils import utiliy


# Step 0. Auxiliary functions or scripts
def get_timestamp():
    # year month day-hour minute second e.g. 20220409-160445
    return datetime.datetime.now().strftime(r"%Y%m%d-%H%M%S")


# Step 1. Parse JSON configure file
def parse_config(path2config):
    # remove comments starting with "//"
    json_str = ""
    with open(path2config, "r") as config_file:
        for line in config_file:
            line = line.split("//")[0] + "\n"
            json_str += line
    
    # the dictionary form of data structure to store configuration information (train/inference, networks, dataset...)
    opt_config = json.loads(json_str, object_pairs_hook=OrderedDict)
    # add time stamp for log information, to record file and change the file name
    opt_config["timestamp"] = get_timestamp()

    scale = opt_config["scale"] # for image super-resolution
    rgb_range = opt_config["rgb_range"]

    # -----------------------------------------------------------------
    # CUDA Pro Tip: Control GPU Visibility with CUDA_VISIBLE_DEVICES
    # https://developer.nvidia.com/zh-cn/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/
    # export CUDA_VISIBLE_DEVICES 
    # ------------------------------
    if torch.cuda.is_available():
        gpu_list = ",".join(str(gpu_id) for gpu_id in opt_config["gpu_ids"])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        print("========> Export CUDA_VISIBLE_DEVICES = [{}]".format(gpu_list))
    else:
        print("========> CPU mode is setting (NOTE: GPU is recommended)")

    # dataset confirgureation
    for phase, dataset in opt_config["datasets"].items():
        phase = phase.split("_")[0]
        dataset["phase"] = phase
        dataset["scale"] = scale # for image super-resolution
        dataset["rgb_range"] = rgb_range

    # for network initialize
    opt_config["networks"]["scale"] = opt_config["scale"] # for image super-resolution networks
    network_opt = opt_config["networks"]

    # ----------------------------------------------------------------------
    # file record and save the file name format e.g. Model_name_in64f64_x4
    config_str = "{}_in{}f{}_x{}".format(network_opt["which_model"].upper(), network_opt["in_channels"], network_opt["num_features"], opt_config["scale"])

    experiment_path = os.path.join(os.getcwd(), "experiments", config_str)

    if opt_config["is_train"] and opt_config["solver"]["pretrain"]: # 断点续训
        if "pretrained_path" not in list(opt_config["solver"].keys()):
            raise ValueError("[Error] The pretrained_path does not declarate in *.json")

        experiment_path = os.path.dirname(os.path.dirname(opt_config["solver"]["pretrained_path"]))

        if opt_config["solver"]["pretrain"] == "finetune": # 微调网络模型
            experiment_path += "_finetune"

    experiment_path = os.path.relpath(experiment_path)
    # --------------------------------------------------

    path_opt_config = OrderedDict()
    path_opt_config["experiment_root"] = experiment_path
    path_opt_config["epochs"] = os.path.join(experiment_path, "epochs") # saveing the epochs weight .pth
    path_opt_config["visual"] = os.path.join(experiment_path, "visual") # saveing the epochs visual results
    path_opt_config["records"] = os.path.join(experiment_path, "records") # record the information of model
    opt_config["path"] = path_opt_config

    if opt_config["is_train"]:
        # create folders
        if opt_config["solver"]["pretrain"] == "resume": # 断点续训
            opt_config = dict_to_nonedict(opt_config) # custom function
        else:
            utiliy.mkdir_and_rename(opt_config["path"]["experiment_root"]) # rename old experiments with timestamp format if exists
            utiliy.mkdirs(path for key, path in opt_config["path"].items() if not key == "experiment_root")

            save_configure_file(opt_config) # custom function to save configure file with JSON format

            opt_config = dict_to_nonedict(opt_config)
        
        print("========> Experimental Folder: {}".format(experiment_path))

    return opt_config


def dict_to_nonedict(opt):
    """convert to NoneDict, which return None for missing key.

    Args:
        opt (_type_): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(opt, dict):
        new_opt = dict()
        # In Python Dictionary, items() method is used to return the list with all dictionary keys with values. 
        for key, sub_opt in opt.items():
            # recursive use, complete the nested dictionary form
            new_opt[key] = dict_to_nonedict(sub_opt)

        #  **new_opt 表示把 new_opt 这个 dict 的所有 key-value 用关键字参数传入到函数的 **kw 参数
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def save_configure_file(opt):
    dump_dir = opt["path"]["experiment_root"]
    dump_path = os.path.join(dump_dir, "options_configure.json")
    with open(dump_path, "w") as dump_file:
        json.dump(opt, dump_file, indent=2)


class NoneDict(dict):
    def __missing__(self, key):
        return None