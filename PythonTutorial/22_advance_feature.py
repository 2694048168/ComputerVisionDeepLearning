#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 22_advance_feature.py
@Python Version: 3.12.13
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2026-10-11
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V0.1
@License: Apache License Version 2.0, January 2004
    Copyright 2026. All rights reserved.

@Description: Python 高级特性: 切片、迭代、列表生成式、生成器、迭代器
"""


def Slice_test():
    """取一个list或tuple的部分元素是非常常见的操作
    对这种经常取指定索引范围的操作，用循环十分繁琐，
    因此，Python提供了切片（Slice）操作符
    L[0:3]表示，从索引0开始取，直到索引3为止，但不包括索引3
    Python支持L[-1]取倒数第一个元素，那么它同样支持倒数切片
    甚至什么都不写，只写[:]就可以原样复制一个list

    tuple也是一种list，唯一区别是tuple不可变。
    因此，tuple也可以用切片操作，只是操作的结果仍是tuple：

    在很多编程语言中，针对字符串提供了很多各种截取函数（例如，substring），
    其实目的就是对字符串切片。
    Python没有针对字符串的截取函数，只需要切片一个操作就可以完成
    """
    L = ["Michael", "Sarah", "Tracy", "Bob", "Jack"]

    print("L[0:3] =", L[0:3])
    print("L[:3] =", L[:3])
    print("L[1:3] =", L[1:3])
    print("L[-2:] =", L[-2:])

    R = list(range(100))
    print("R[:10] =", R[:10])
    print("R[-10:] =", R[-10:])
    print("R[10:20] =", R[10:20])
    print("R[:10:2] =", R[:10:2])
    print("R[::5] =", R[::5])

    S = "hello, world"
    print("S[:5] =", S[:5])
    print("S[7:] =", S[7:])


def trim(s):
    """利用切片操作，实现一个trim()函数，去除字符串首尾的空格，
    注意不要调用str的strip()方法

    利用切片操作，通过找到字符串中第一个非空格字符和最后一个非空格字符的位置，
    然后对原字符串进行切片，从而去除首尾空格
    """
    start = 0
    end = len(s)
    # 从左向右找到第一个非空格字符的位置
    while start < end and s[start] == " ":
        start += 1
    # 从右向左找到最后一个非空格字符的位置
    while end > start and s[end - 1] == " ":
        end -= 1
    # 利用切片返回去除首尾空格后的子串
    return s[start:end]


def Iteration_test():
    """
    通过for ... in来完成的迭代（Iteration）
    可以作用在其他可迭代对象上

    默认情况下，dict迭代的是key。
    如果要迭代value，可以用for value in d.values()，
    如果要同时迭代key和value，可以用for k, v in d.items()

    Python内置的enumerate函数可以把一个list变成索引-元素对，
    这样就可以在for循环中同时迭代索引和元素本身：
    """
    data_list = [1, 2, 3]
    for idx, data in enumerate(data_list):
        print(f"the index = {idx} data value = {data}")

    data_dict = {"a": 1, "b": 2, "c": 3}
    for k, v in data_dict.items():
        print(f"the key = {k}, and value = {v}")


def findMinAndMax(lst):
    if not lst:
        return (None, None)

    min_val = max_val = lst[0]  # 用第一个元素初始化

    for item in lst:  # 迭代整个列表
        if item < min_val:
            min_val = item
        if item > max_val:
            max_val = item

    return (min_val, max_val)


# --------------------------
if __name__ == "__main__":
    Slice_test()

    # 测试:
    print("---------------------------")
    if trim("hello  ") != "hello":
        print("测试失败!")
    elif trim("  hello") != "hello":
        print("测试失败!")
    elif trim("  hello  ") != "hello":
        print("测试失败!")
    elif trim("  hello  world  ") != "hello  world":
        print("测试失败!")
    elif trim("") != "":
        print("测试失败!")
    elif trim("    ") != "":
        print("测试失败!")
    else:
        print("测试成功!")

    print("---------------------------")
    Iteration_test()

    # 测试
    print("---------------------------")
    if findMinAndMax([]) != (None, None):
        print("测试失败!")
    elif findMinAndMax([7]) != (7, 7):
        print("测试失败!")
    elif findMinAndMax([7, 1]) != (1, 7):
        print("测试失败!")
    elif findMinAndMax([7, 1, 3, 9, 5]) != (1, 9):
        print("测试失败!")
    else:
        print("测试成功!")
