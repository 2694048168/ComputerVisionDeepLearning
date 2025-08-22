#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 01_package.py
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

# Regular Package and Namespace Package
# 1. Regular Package is Folder with spec. file '__init__.py';
# 2. Namespace Package is just Folder;
# 3. Python import OP. (搜索 sys.modules)
# 3.1 检查缓存 cache, 如 __pycache__ 路径
# 3.2 检查搜索目录 dirs, 包括built-in包路径, Python可执行路径, 当前路径
# 3.3 加载或编译文件 loading & compiling files
# 3.4 执行代码 executing code
# 3.5 管理依赖项 dependencies
# 包的类型: built-in module 则 import sys.modules
# 包的类型为文件python file: 则编译 Bytecode ---> __pycache__ , 缓存下次更快加载
# -------------------------------------------
# '__init__.py' 最佳实践要点总结
# 1. 提供包文档：在文件顶部添加文档字符串，描述包的功能和使用方法
# 2. 定义元数据：包括版本号、作者等信息
# 3. 控制导入：使用 __all__ 明确公开的API
# 4. 初始化逻辑：执行包级别的初始化操作
# 5. 错误处理：妥善处理可选依赖和导入错误
# 6. 资源管理：使用 importlib.resources 安全访问包内文件
# 7. 日志配置：设置适当的日志记录
# 8. 版本兼容性：检查Python版本要求
# 9. 清理命名空间：删除不需要导出的临时变量
# -------------------------------------------
# """
# package_name - 一个示例Python包

# 提供主要功能描述和使用示例。
# """

# # 元数据
# __version__ = "1.0.0"
# __author__ = "Your Name"
# __license__ = "MIT"

# import logging
# from importlib import resources

# # 设置日志
# _logger = logging.getLogger(__name__)
# _logger.addHandler(logging.NullHandler())

# # 包初始化
# def _initialize():
#     """包初始化函数"""
#     _logger.info("初始化 %s 包", __name__)
#     # 执行必要的初始化操作

# # 导入主要API
# try:
#     from .core import (
#         main_function,
#         PrimaryClass,
#         helper_function,
#     )

#     from .utils import (
#         validate_input,
#         process_data,
#     )

#     # 可选功能
#     try:
#         from .optional import advanced_operation
#     except ImportError:
#         advanced_operation = None
#         _logger.debug("可选功能未启用")

# except ImportError as e:
#     _logger.error("导入包模块时出错: %s", e)
#     raise

# # 定义公共API
# __all__ = [
#     'main_function',
#     'PrimaryClass',
#     'helper_function',
#     'validate_input',
#     'process_data',
#     'advanced_operation',
#     'CONSTANT_VALUE',
#     'get_config'
# ]

# # 包常量
# CONSTANT_VALUE = 42

# # 包级别函数
# def get_config():
#     """获取包配置"""
#     try:
#         # 使用importlib.resources安全地访问包内文件
#         with resources.open_text('package_name', 'config.ini') as f:
#             return f.read()
#     except FileNotFoundError:
#         return "默认配置"

# # 执行初始化
# _initialize()

# # 清理不需要的名称
# del _initialize, resources, logging
# -------------------------------------------

import os
import sys

if __name__ == "__main__":
    print(f"The 'os' Package path: {os.__package__}")
    print("The 'os' Package path: package_name.__path__")

    print(f"The system modules: {sys.modules}")
