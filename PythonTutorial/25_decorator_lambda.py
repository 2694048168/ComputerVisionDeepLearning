#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 25_decorator_lambda.py
@Python Version: 3.12.13
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2026-10-12
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V0.1
@License: Apache License Version 2.0, January 2004
    Copyright 2026. All rights reserved.

@Description: Python 函数式编程: Lambda和Decorator
"""

import functools
import time
from functools import wraps


def lambda_test():
    """
    当在传入函数时，有些时候，不需要显式地定义函数，直接传入匿名函数更方便。

    匿名函数lambda x: x * x
    关键字lambda表示匿名函数，冒号前面的x表示函数参数
    匿名函数有个限制，就是只能有一个表达式，不用写return，返回值就是该表达式的结果
    匿名函数也是一个函数对象，也可以把匿名函数赋值给一个变量

    """
    val = list(map(lambda x: x * x, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
    print(f"the list value {val}")

    res = list(filter(lambda n: n % 2 == 1, range(1, 20)))
    print(f"the result value {res}")


def decorator_test():
    """
    在代码运行期间动态增加功能的方式，称之为“装饰器”（Decorator）
    在函数调用前后自动打印日志，但又不希望修改now()函数的定义
    本质上，decorator就是一个返回函数的高阶函数
    """

    def log(func):
        def wrapper(*args, **kw):
            print("call %s():" % func.__name__)
            return func(*args, **kw)

        return wrapper

    # 借助Python的@语法，把decorator置于函数的定义处
    @log
    def now():
        print("2026-10-12")

    now()

    def log(text):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kw):
                print("%s %s():" % (text, func.__name__))
                return func(*args, **kw)

            return wrapper

        return decorator

    @log("execute")
    def now():
        print("2026-10-12")

    now()


# 设计一个decorator，它可作用于任何函数上，并打印该函数的执行时间
def metric(repeat=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            total = 0.0
            result = None
            for _ in range(repeat):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                total += time.perf_counter() - start
            avg = total / repeat
            print(f"[{func.__name__}] 平均耗时({repeat}次): {avg * 1000:.4f} ms")
            return result

        return wrapper

    return decorator


# 测试
@metric(10)
def fast(x, y):
    time.sleep(0.0012)
    return x + y


@metric(10)
def slow(x, y, z):
    time.sleep(0.1234)
    return x * y * z


# ---------------------------
if __name__ == "__main__":
    lambda_test()

    print("-----------------")
    decorator_test()

    f = fast(11, 22)
    s = slow(11, 22, 33)
