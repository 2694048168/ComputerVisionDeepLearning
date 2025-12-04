#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: Serializer.py
@Python Version: 3.12.1
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2025-08-15.
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V1.0
@License: Apache License Version 2.0, January 2004
    Copyright 2025. All rights reserved.

@Description: 序列化基类
"""

from enum import Enum
from abc import ABC, abstractmethod
from typing import Any, Type, TypeVar, Dict, Union, Optional, List
from dataclasses import dataclass, asdict, is_dataclass, fields
import inspect
import json
import pickle
import msgpack
import yaml


class SerializationType(Enum):
    """序列化类型枚举"""
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    YAML = "yaml"


class Serializer(ABC):
    """序列化抽象基类"""
    @abstractmethod
    def serialize(self, data: Any) -> bytes:
        pass
    
    @abstractmethod
    def deserialize(self, serialized_data: bytes, target_type: Optional[Type] = None) -> Any:
        pass


class SerializerJson(Serializer):
    def serialize(self, data: Any) -> bytes:
        # 如果是数据类，转换为字典
        if is_dataclass(data):
            data = asdict(data)
        return json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8')
    
    def deserialize(self, serialized_data: bytes, target_type: Optional[Type] = None) -> Any:
        data = json.loads(serialized_data.decode('utf-8'))
        
        # 如果指定了目标类型并且是数据类，转换为数据类实例
        if target_type and is_dataclass(target_type):
            return self._dict_to_dataclass(data, target_type)
        return data
    
    def _dict_to_dataclass(self, data: Dict, target_type: Type) -> Any:
        """将字典转换为数据类实例"""
        # 处理嵌套数据类
        field_types = {f.name: f.type for f in fields(target_type)}
        kwargs = {}
        for key, value in data.items():
            if key in field_types:
                field_type = field_types[key]
                # 如果字段是数据类并且值是字典，递归转换
                if is_dataclass(field_type) and isinstance(value, dict):
                    value = self._dict_to_dataclass(value, field_type)
                # 如果字段是数据类列表
                elif (inspect.isclass(field_type) and 
                      hasattr(field_type, '__origin__') and 
                      field_type.__origin__ is list and
                      len(field_type.__args__) > 0 and
                      is_dataclass(field_type.__args__[0]) and
                      isinstance(value, list)):
                    item_type = field_type.__args__[0]
                    value = [self._dict_to_dataclass(item, item_type) if isinstance(item, dict) else item for item in value]
            kwargs[key] = value
        return target_type(**kwargs)


class SerializerPickle(Serializer):
    def serialize(self, data: Any) -> bytes:
        return pickle.dumps(data)
    
    def deserialize(self, serialized_data: bytes, target_type: Optional[Type] = None) -> Any:
        return pickle.loads(serialized_data)


class SerializerMsgpack(Serializer):
    def serialize(self, data: Any) -> bytes:
        # 如果是数据类，转换为字典
        if is_dataclass(data):
            data = asdict(data)
        return msgpack.packb(data, use_bin_type=True)
    
    def deserialize(self, serialized_data: bytes, target_type: Optional[Type] = None) -> Any:
        data = msgpack.unpackb(serialized_data, raw=False)
        
        # 如果指定了目标类型并且是数据类，转换为数据类实例
        if target_type and is_dataclass(target_type):
            return SerializerJson()._dict_to_dataclass(data, target_type)
        return data


class SerializerYaml(Serializer):
    def serialize(self, data: Any) -> bytes:
        # 如果是数据类，转换为字典
        if is_dataclass(data):
            data = asdict(data)
        return yaml.safe_dump(data).encode('utf-8')
    
    def deserialize(self, serialized_data: bytes, target_type: Optional[Type] = None) -> Any:
        data = yaml.safe_load(serialized_data.decode('utf-8'))
        
        # 如果指定了目标类型并且是数据类，转换为数据类实例
        if target_type and is_dataclass(target_type):
            return SerializerJson()._dict_to_dataclass(data, target_type)
        return data


class SerializerFactory:
    """序列化工厂"""
    @staticmethod
    def create_serializer(serialization_type: SerializationType) -> Serializer:
        if serialization_type == SerializationType.JSON:
            return SerializerJson()
        elif serialization_type == SerializationType.PICKLE:
            return SerializerPickle()
        elif serialization_type == SerializationType.MSGPACK:
            return SerializerMsgpack()
        elif serialization_type == SerializationType.YAML:
            return SerializerYaml()
        else:
            raise ValueError(f"Unsupported serialization type: {serialization_type}")
        
