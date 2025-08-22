#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: constructor.py
@Python Version: 3.12.1
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2025-08-22.
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V1.0
@License: Apache License Version 2.0, January 2004
    Copyright 2025. All rights reserved.

@Description:
"""

# Python class constructor 构造函数和初始化函数


class Shape:
    def __init__(self, name: str, dim: int):
        self.name = name
        self.dim = dim
        print(f"Inside __init__ Initializing object: {self.__init__.__name__}")
        print(f"Inside Initializing name: {self.name}")
        print(f"Inside Initializing dim: {self.dim}")

    def __new__(cls, name: str, dim: int) -> "Shape":
        """__new__ needed when modifying object creation itself"""
        print(f"Inside __new__ Creating object: {cls.__new__.__name__}")
        return super().__new__(cls)


if __name__ == "__main__":
    shape = Shape("circle", 2)
