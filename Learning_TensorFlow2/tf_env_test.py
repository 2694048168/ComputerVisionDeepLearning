#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 测试 TensorFlow 安装环境以及 TensorFlow 版本
@Python Version: 3.8.12
@Author: Wei Li
@Date: 2022-03-12
"""

import tensorflow as tf


def tensorflow_environment():
    """Test host computer whether supported TensorFlow version and GPU ?
    """
    print(f"TensorFlow Version : {tf.__version__}")
    print(f"Host Computer supported Devices : {tf.config.list_physical_devices()}")


if __name__ == "__main__":
    tensorflow_environment()