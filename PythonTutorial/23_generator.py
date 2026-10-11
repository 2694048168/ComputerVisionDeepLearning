#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 23_generator.py
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

import os
from collections.abc import Iterator


def ListComprehensions():
    """
    列表生成式即List Comprehensions，
    是Python内置的非常简单却强大的可以用来创建list的生成式
    循环太繁琐，而列表生成式则可以用一行语句代替循
    """
    data = list(range(1, 11))
    for value in data:
        print(f"the data value: {value}")

    # 写列表生成式时，把要生成的元素x * x放到前面，后面跟for循环，就可以把list创建
    [x * x for x in range(1, 11)]
    print(f"the value: {[x * x for x in range(1, 11)]}")

    # 运用列表生成式，可以写出非常简洁的代码。例如，列出当前目录下的所有文件和目录名
    [d for d in os.listdir(".")]
    print(f"the value: {[d for d in os.listdir('.')]}")

    # 把一个list中所有的字符串变成小写
    print("------------------------")
    L = ["Hello", "World", "IBM", "Apple"]
    [s.lower() for s in L]
    print(f"the value: {L}")
    print(f"the value: {[s.lower() for s in L]}")


def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n = n + 1
    return "done"


def generator_test():
    """
    通过列表生成式，可以直接创建一个列表。
    但是，受到内存限制，列表容量肯定是有限的。
    而且，创建一个包含100万个元素的列表，不仅占用很大的存储空间，

    列表元素可以按照某种算法推算出来，
    可以在循环的过程中不断推算出后续的元素
    这样就不必创建完整的list，从而节省大量的空间。
    在Python中，这种一边循环一边计算的机制，称为生成器：generator

    generator的工作原理，它是在for循环的过程中不断计算出下一个元素，
    并在适当的条件结束for循环。对于函数改成的generator来说，
    遇到return语句或者执行到函数体最后一行语句，
    就是结束generator的指令，for循环随之结束
    """
    # 把一个列表生成式的[]改成()，就创建了一个generator
    g = (x * x for x in range(10))
    # generator保存的是算法，
    # 每次调用next(g)，就计算出g的下一个元素的值，直到计算到最后一个元素
    for n in g:
        print(f"the value = {n}")

    # 定义generator的另一种方法。如果一个函数定义中包含yield关键字，
    # 那么这个函数就不再是一个普通函数，而是一个generator函数
    f = fib(6)
    for n in f:
        print(f"the fib value = {n}")


def triangles():
    row = [1]
    while True:
        yield row
        row = [1] + [a + b for a, b in zip(row, row[1:])] + [1]


def iterator_test():
    """
    可以直接作用于for循环的数据类型有以下几种：
    一类是集合数据类型，如list、tuple、dict、set、str等；
    一类是generator，包括生成器和带yield的generator function。
    这些可以直接作用于for循环的对象统称为可迭代对象：Iterable。
    可以使用isinstance()判断一个对象是否是 Iterable 对象：
    """
    print(f"the dict is Iterable: {isinstance({}, Iterator)}")
    print(f"the list is Iterable: {isinstance([], Iterator)}")
    print(f"the list is Iterable: {isinstance(iter([]), Iterator)}")


# --------------------------
if __name__ == "__main__":
    ListComprehensions()

    print("====================")
    generator_test()

    print("====================")
    n = 0
    for t in triangles():
        print(t)
        n += 1
        if n == 10:
            break

    print("===========================")
    iterator_test()
