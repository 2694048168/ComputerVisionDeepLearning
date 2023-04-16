#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The solver to image transformation processing with Deep Learning Models with PyTorch
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
# ----- custom library -----
from .SRSolver import SRSolver


def create_solver(opt):
    if opt['mode'] == 'SISR':
        solver = SRSolver(opt)
    else:
        raise NotImplementedError

    return solver