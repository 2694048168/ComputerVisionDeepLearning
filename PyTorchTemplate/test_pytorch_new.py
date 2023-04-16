#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: test the lastest version of PyTorch 1.12.1
@Python Version: 3.10.4
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-08-13
"""

import sys
from platform import python_version, python_implementation, python_compiler

import torch
import torchvision
import numpy as np


if __name__ == "__main__":
    print(f'the version of python: {sys.version}')
    print(f'the version of python: {python_version()}')
    print(f'the compiler of python: {python_compiler()}')
    print(f'the implementation of python: {python_implementation()}')

    print('------------------------------------------')
    print(f'the version of torch: {torch.__version__}')
    print(f'the version of torchvision: {torchvision.__version__}')
    print(f'test the cuda is: {torch.cuda.is_available()}')

    print('------------------------------------------')
    print(f'the version of numpy: {np.__version__}')
