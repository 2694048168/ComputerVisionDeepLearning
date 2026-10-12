#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 24_high_order_function.py
@Python Version: 3.12.13
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2026-10-12
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V0.1
@License: Apache License Version 2.0, January 2004
    Copyright 2026. All rights reserved.

@Description: 高阶函数 Higher-order function
"""

from functools import reduce


def add(x, y, f):
    return f(x) + f(y)


def map_reduce_test():
    """
    Python内建了map()和reduce()函数
    “MapReduce: Simplified Data Processing on Large Clusters” paper from Google

    map() 函数接收两个参数，一个是函数，一个是Iterable，
    map将传入的函数依次作用到序列的每个元素，并把结果作为新的Iterator返回。

    reduce 把一个函数作用在一个序列[x1, x2, x3, ...]上，
    这个函数必须接收两个参数，
    reduce 把结果继续和序列的下一个元素做累积计算
    """

    def f(x):
        return x * x

    data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(f"the original data: {data}")
    data_ = map(f, data)
    print(f"the process result: {data_}")
    print(f"the process data: {list(data_)}")

    def fn(x, y):
        return x * 10 + y

    res = reduce(fn, [1, 3, 5, 7, 9])
    print(f"the process result: {res}")


def demo(s):
    # 字符串str也是一个序列，配合map()，就可以写出把str转换为int的函数
    DIGITS = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
    }

    def fn(x, y):
        return x * 10 + y

    def char2num(s):
        return DIGITS[s]

    return reduce(fn, map(char2num, s))


def normalize(name: str) -> str:
    """将名字规范化为首字母大写，其余小写"""
    return name.strip().capitalize()


def normalize_name(name_list: list) -> list:
    """将名字规范化为首字母大写，其余小写"""
    return list(map(normalize, name_list))


def filter_test():
    """
    Python内建的filter()函数用于过滤序列。

    和map()类似，filter()也接收一个函数和一个序列。
    filter()把传入的函数依次作用于每个元素，
    然后根据返回值是True还是False决定保留还是丢弃该元素。

    注意到filter()函数返回的是一个Iterator，也就是一个惰性序列，
    所以要强迫filter()完成计算结果，需要用list()函数获得所有结果并返回list
    """

    def is_odd(n):
        return n % 2 == 1

    res = list(filter(is_odd, [1, 2, 4, 5, 6, 9, 10, 15]))
    print(f"the result is {res}")

    def not_empty(s):
        return s and s.strip()

    ret = list(filter(not_empty, ["A", "", "B", None, "C", "  "]))
    print(f"the result is {ret}")


def sorted_test():
    """
    排序也是在程序中经常用到的算法。无论使用冒泡排序还是快速排序，
    排序的核心是比较两个元素的大小。如果是数字，可以直接比较，
    但如果是字符串或者两个dict呢？
    直接比较数学上的大小是没有意义的，因此，比较的过程必须通过函数抽象出来

    Python内置的sorted()函数就可以对list进行排序
    sorted()函数也是一个高阶函数，它还可以接收一个key函数来实现自定义的排序
    默认情况下，对字符串排序，是按照ASCII的大小比较的，由于'Z' < 'a'
    要进行反向排序，不必改动key函数，可以传入第三个参数reverse=True
    """
    list_val = [36, 5, -12, 9, -21]
    sorted_val = sorted(list_val)
    sorted_data = sorted(list_val, key=abs)
    print(f"the data before sorted: {list_val}")
    print(f"the data after sorted: {sorted_val}")
    print(f"the data after sorted(abs): {sorted_data}")


# =-------------------------
if __name__ == "__main__":
    # 函数名也是变量
    print(f"the value == {add(-12, 6, abs)}")

    print("=====================")
    map_reduce_test()

    number = demo("8208208821")
    print(f"the number is {number}")

    name_input = ["adam", "LISA", "barT"]
    name_ouput = normalize_name(name_input)
    print(f"the original name: {name_input}")
    print(f"the process name: {name_ouput}")

    print("=====================")
    filter_test()

    print("=====================")
    sorted_test()
