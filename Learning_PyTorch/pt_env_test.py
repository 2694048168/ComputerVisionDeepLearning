#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 测试 PyTorch 安装环境以及 PyTorch 版本
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-03-29
"""

import torch
import torchvision
import torchtext
import torchkeras

if __name__ == "__main__":
    print(f"PyTorch Version : {torch.__version__}")
    print(f"TorchVision Version : {torchvision.__version__}")
    print(f"Torchtext Version : {torchtext.__version__}")
    print(f"Torchkeras Version : {torchkeras.__version__}")