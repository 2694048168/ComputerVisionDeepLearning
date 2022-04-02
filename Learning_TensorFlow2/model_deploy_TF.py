#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The high-level API of TensorFlow2
@Python Version: 3.8.12
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-03-29
"""

""" 
TensorFlow 的高阶 API
high-level API of TensorFlow 主要是 tensorflow.keras.models

1. 模型的构建: Sequential、functional API、Model子类化
2. 模型的训练: 内置 fit 方法, 内置 train_on_batch 方法, 自定义训练循环, 单 GPU 训练模型, 多 GPU 训练模型, TPU训练模型
3. 模型的部署: tensorflow serving 部署模型, 使用 spark(scala) 调用 tensorflow 模型

使用 tensorflow-serving 部署模型
TensorFlow 训练好的模型以 tensorflow 原生方式保存成 protobuf 文件后可以用许多方式部署运行
1. 通过 tensorflow-js 可以用 javascrip 脚本加载模型并在浏览器中运行模型
2. 通过 tensorflow-lite 可以在移动和嵌入式设备上加载并运行 TensorFlow 模型
3. 通过 tensorflow-serving 可以加载模型后提供网络接口 API 服务, 通过任意编程语言发送网络请求都可以获取模型预测结果
4. 通过 tensorFlow for Java 接口, 可以在 Java 或者 spark(scala) 中调用 tensorflow 模型进行预测

介绍 tensorflow serving 部署模型

https://jackiexiao.github.io/eat_tensorflow2_in_30_days/chinese/6.%E9%AB%98%E9%98%B6API/6-6%2C%E4%BD%BF%E7%94%A8tensorflow-serving%E9%83%A8%E7%BD%B2%E6%A8%A1%E5%9E%8B/#%E3%80%87tensorflow-serving%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2%E6%A6%82%E8%BF%B0

tensorflow serving 模型部署概述
使用 tensorflow serving 部署模型要完成以下步骤
(1) 准备 protobuf 模型文件
(2) 安装 tensorflow serving
(3) 启动 tensorflow serving 服务
(4) 向API服务发送请求, 获取预测结果

使用 spark(scala) 调用 tensorflow 模型的方法

https://jackiexiao.github.io/eat_tensorflow2_in_30_days/chinese/6.%E9%AB%98%E9%98%B6API/6-7%2C%E4%BD%BF%E7%94%A8spark-scala%E8%B0%83%E7%94%A8tensorflow%E6%A8%A1%E5%9E%8B/

spark-scala 调用 tensorflow 模型概述
在 spark(scala) 中调用 tensorflow 模型进行预测需要完成以下几个步骤
1. 准备 protobuf 模型文件
2. 创建 spark(scala) 项目,在项目中添加java版本的 tensorflow 对应的 jar包依赖 (Maven 管理项目)
3. 在 spark(scala) 项目中 driver 端加载 tensorflow 模型调试成功
4. 在 spark(scala) 项目中通过 RDD 在 executor 上加载 tensorflow 模型调试成功
5. 在 spark(scala) 项目中通过 DataFrame 在 executor 上加载 tensorflow 模型调试成功
"""