#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 00_os_pylib.py
@Python Version: 3.12.1
@Platform: PyTorch 2.0.0 + cu117
@Author: Wei Li (Ithaca)
@Date: 2024-08-13.
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V2.1
@License: Apache License Version 2.0, January 2004
    Copyright 2024. All rights reserved.

@Description: 
"""

# Python 标准库 OS
# 1. 系统相关变量和操作
# 2. 文件和目录相关操作
# 3. 执行命令和管理进程
import os


if __name__ == "__main__":
    # 1. 系统相关变量和操作
    # https://blog.csdn.net/qq_30159015/article/details/82658345
    # 分别是posix , nt , java， 对应linux/windows/java虚拟机
    print(f"the OS of current HOST: {os.name}")

    print(f"the env-var of current HOST: {os.environ}")

    # Windows ---> /   Linux ----> \
    print(f"the file Separator of current OS System: {os.sep}")
    # Windows ---> ;   Linux ----> :
    print(f"the env-var Separator of current OS System: {os.pathsep}")
    # Windows ---> \r\n   Linux ----> \n
    print(f"the line Separator of current OS System: {os.linesep}")

    # 2. 文件和目录相关操作
    folder_path = "./subfolder"
    # os.mkdir(folder_path)
    os.makedirs(folder_path, exist_ok=True)
    # rename remove ...
    print(f"the current folder path: {os.getcwd()}")

    # os and os.path
    file = os.getcwd() + "/00_os_pylib.py"
    folder, filename = os.path.split(file)
    print(f"the current folder: {folder} and file: {filename}")
    print(f"the current filepath is abs-path: {os.path.isabs(file)}")
    print(f"the current filepath is abs-path: {os.path.isabs(filename)}")

    # isdir isfile exists
    print(f"the filename is exists: {os.path.exists(file)}")
    # getatime
    print(f"the A-time of filename: {os.path.getatime(file)}")
    print(f"the C-time of filename: {os.path.getctime(file)}")
    print(f"the size of filename: {os.path.getsize(file)} Bytes")

    # 3. 执行命令和管理进程
    # system popen
    os.system("ipconfig")
    os.system("python hello_world.py")
