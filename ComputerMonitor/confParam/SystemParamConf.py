#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: SystemParamConf.py
@Python Version: 3.12.1
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2025-08-15.
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V1.0
@License: Apache License Version 2.0, January 2004
    Copyright 2025. All rights reserved.

@Description: 系统参数序列化与反序列化 json
"""

import os
from dataclasses import is_dataclass, asdict

from confParam.DataStruct import SystemParam
from confParam.Serializer import SerializationType, SerializerFactory


class SystemParamConf:
    def __init__(self):
        self.params = SystemParam()
    
    def __str__(self) -> str:
        """返回参数的字符串表示"""
        if is_dataclass(self.params):
            return str(asdict(self.params))
        return str(self.params)

    def SerializeToFile(self, file_path: str, serialization_type: SerializationType = SerializationType.JSON):
        """序列化保存到文件"""
        serializer = SerializerFactory.create_serializer(serialization_type)
        serialized = serializer.serialize(self.params)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            file.write(serialized)
    
    def DeserializeFromFile(self, file_path: str, serialization_type: SerializationType = SerializationType.JSON):
        """从文件加载并反序列化"""
        serializer = SerializerFactory.create_serializer(serialization_type)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'rb') as file:
            serialized = file.read()
        
        # 数据类类型, 传递给反序列化器
        if is_dataclass(self.params):
            self.params = serializer.deserialize(serialized, SystemParam)
        else:
            self.params = serializer.deserialize(serialized)
