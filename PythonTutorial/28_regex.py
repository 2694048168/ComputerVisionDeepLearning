#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 27_regex.py
@Python Version: 3.12.13
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2026-10-14
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V0.1
@License: Apache License Version 2.0, January 2004
    Copyright 2026. All rights reserved.

@Description: 正则表达式
"""

import re


def check_email(email: str) -> bool:
    email_re = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$")
    return bool(email_re.match(email))


def extract_email_name(s):
    name_email_re = re.compile(
        r"^\s*(?:<([^>]+)>\s*)?([A-Za-z0-9._%+-]+)@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$"
    )
    m = name_email_re.match(s.strip())
    if not m:
        return None
    # group(1) 是尖括号里的名字，如 Tom Paris
    # group(2) 是普通邮箱的 @ 前面部分，如 bob
    return m.group(1) or m.group(2)


# -------------------------
if __name__ == "__main__":
    # 字符串是编程时涉及到的最多的一种数据结构，对字符串进行操作的需求几乎无处不在
    """
    正则表达式是一种用来匹配字符串的强有力的武器。
    它的设计思想是用一种描述性的语言来给字符串定义一个规则，
    凡是符合规则的字符串，就认为它“匹配”了，
    否则，该字符串就是不合法的。

    所以我们判断一个字符串是否是合法的Email的方法是：
        创建一个匹配Email的正则表达式；
        用该正则表达式去匹配用户的输入来判断是否合法。

    正则表达式也是用字符串表示的,首先了解如何用字符来描述字符
    在正则表达式中，如果直接给出字符，就是精确匹配。
    用\d可以匹配一个数字，\w可以匹配一个字母或数字，所以：
        '00\d'可以匹配'007'，但无法匹配'00A'；
        '\d\d\d'可以匹配'010'；
        '\w\w\d'可以匹配'py3'；
        .可以匹配任意字符，所以：
            'py.'可以匹配'pyc'、'pyo'、'py!'等等。
    要匹配变长的字符，在正则表达式中，用*表示任意个字符（包括0个），
    用+表示至少一个字符，用?表示0个或1个字符，
    用{n}表示n个字符，用{n,m}表示n-m个字符：
    要做更精确地匹配，可以用[]表示范围

    Python提供re模块，包含所有正则表达式的功能。
    Python的字符串本身也用 \ 表达转义, 除了原生字符串 r前缀表示的

    正则表达式要重复使用几千次，出于效率的考虑，
    我们可以预编译该正则表达式，接下来重复使用
    """
    regex_express = r"^\d{3}\-\d{3,8}$"
    regex_match = re.match(regex_express, "010-12345")
    print(f"the result match: {regex_match}")
    regex_match = re.match(regex_express, "010 12345")
    print(f"the result match: {regex_match}")

    # 编译:
    re_telephone = re.compile(r"^(\d{3})-(\d{3,8})$")
    # 使用：编译后生成Regular Expression对象，由于该对象自己包含了正则表达式
    regex_match = re_telephone.match("010-12345").groups()
    print(f"the result match: {regex_match}")
    regex_match = re_telephone.match("010-8086").groups()
    print(f"the result match: {regex_match}")

    print("-------------------------------")
    print(check_email("someone@gmail.com"))  # True
    print(check_email("bill.gates@microsoft.com"))  # True
    print(check_email("bad-email"))  # False

    print("-------------------------------")
    print(extract_email_name("<Tom Paris> tom@voyager.org"))  # Tom Paris
    print(extract_email_name("bob@example.com"))  # bob
