#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@Python Version: 3.12.8
@Author: Wei Li (Ithaca)
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2025/07/27
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Paper:
@Description: 
'''

import os
import sys
import logging

class Logger:
    def __init__(self, 
                 name: str = "default",
                 log_file: str = None,
                 log_folder: str = "./log/",
                 console_level: int = logging.NOTSET,
                 file_level: int = logging.NOTSET,):
        """自定义日志记录器
        :param name: 日志记录器名称 (默认使用模块名)
        :param log_file: 日志文件路径 (不提供则不输出到文件)
        :param console_level: 控制台日志级别 (默认INFO)
        :param file_level: 文件日志级别 (默认DEBUG)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # 设置总级别为最低
        
        # 清除现有处理器避免重复
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 默认格式设置
        default_fmt = '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
        date_fmt = '%Y-%m-%d %H:%M:%S'
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter(default_fmt, datefmt=date_fmt)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # 创建文件处理器（如果提供了文件路径）
        os.makedirs(log_folder, exist_ok=True)
        if log_file:
            # 添加时间戳到文件名（可选）
            # log_file = log_file.replace('.log', f'_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            file_handler = logging.FileHandler(os.path.join(log_folder, log_file), encoding='utf-8')
            file_handler.setLevel(file_level)
            file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s]-[%(name)s][%(levelname)s]-[%(process)d:%(processName)s][%(thread)d:%(threadName)s]-[%(funcName)s:%(lineno)d]-[%(message)s]-%(pathname)s"
            )
        )
            self.logger.addHandler(file_handler)
    
    def GetLogger(self) -> logging.Logger:
        return self.logger

    def DebugMessage(self, message):
        self.logger.debug(message)
        
    def InfoMessage(self, message):
        self.logger.info(message)
        
    def ErrorMessage(self, message):
        self.logger.error(message)

    def CriticalMessage(self, message):
        self.logger.critical(message)


# ---------------------------
if __name__ == "__main__":
    pass
