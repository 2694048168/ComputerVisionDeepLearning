#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: 01_pipPackage.py
@Python Version: 3.12.1
@Author: Wei Li (Ithaca)
@Email: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/
@Date: 2024-03-02
@copyright Copyright (c) 2024 Wei Li
@Description: 利用 pip 安装 Python 的第三方库
@Doc: https://2694048168.github.io/blog/#/PaperMD/python_env_ai
'''

import cv2 as cv


# --------------------------
if __name__ == "__main__":
    # 使用命令行: pip install package
    # pip install opencv-python
    print("the version of OpenCV for python: ", cv.__version__)
