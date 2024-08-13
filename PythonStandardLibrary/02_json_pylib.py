#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 02_json_pylib.py
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

# Python 标准库 Json
# 1. Python针对 Json 文本格式读写
import json


if __name__ == "__main__":
    # python 字典类型
    person = {
        "name": "Sniper",
        "age": 26,
        "tel": ["19923229871", "19212196773"],
        "isOnly": True,
    }
    print(f"the dict type: {type(person)} of Python:\n{person}")

    # python 转换为 json 字符串
    # person_jsonStr = json.dumps(person)
    # person_jsonStr = json.dumps(person, indent=4)
    person_jsonStr = json.dumps(person, indent=4, sort_keys=True)
    print(f"the json-str: {type(person_jsonStr)} type of Python:\n{person_jsonStr}")

    # 将字典数据类型直接写入json格式文件中
    # json.dump(person, open("./person.json", "w"), indent=4)
    json.dump(person, open("./person.json", "w"), indent=4, sort_keys=True)

    print("===========================")
    pythonObj = json.loads(person_jsonStr)
    print(f"the python obj type: {type(pythonObj)}\n{pythonObj}")

    pythonObj = json.load(open("emp.json", "r"))
    print(f"\nthe python obj type: {type(pythonObj)}\n{pythonObj}")
