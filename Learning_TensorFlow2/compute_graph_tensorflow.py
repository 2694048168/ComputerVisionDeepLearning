#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The Tensor Data Structure of TensorFlow2
@Brief: https://tensorflow.google.cn/api_docs/python/tf/Graph
@Python Version: 3.8.12
@Author: Wei Li
@Date: 2022-03-21
"""

"""
Tensorflow 底层最核心的概念是张量(tensor)，计算图(Compute Graph)以及自动微分(autograd)
有三种计算图的构建方式: 静态计算图(static compute graph), 动态计算图(dynamic compute graph), 以及 Autograph

在 TensorFlow 1.0, 采用的是静态计算图, 需要先使用 TensorFlow 的各种算子创建计算图, 然后再开启一个会话 Session, 显式执行计算图. 而在 TensorFlow 2.0, 采用的是动态计算图, 即每使用一个算子后, 该算子会被动态加入到隐含的默认计算图中立即执行得到结果, 而无需开启 Session. 使用动态计算图即 Eager Excution 的好处是方便调试程序, 它会让 TensorFlow 代码的表现和 Python原生代码的表现一样, 写起来就像写 numpy 一样, 各种日志打印, 控制流全部都是可以使用的.

使用动态计算图的缺点是运行效率相对会低一些, 因为使用动态图会有许多次 Python进程和 TensorFlow的C++进程之间的通信. 而静态计算图构建完成之后几乎全部在 TensorFlow core 内核上使用C++代码执行, 效率更高. 此外静态图会对计算步骤进行一定的优化, 剪去和结果无关的计算步骤.

如果需要在 TensorFlow 2.0中使用静态图, 可以使用 @tf.function 装饰器将普通 Python函数转换成对应的 TensorFlow 计算图构建代码.运行该函数就相当于在 TensorFlow 1.0 中用 Session 执行代码. 使用 tf.function 构建静态图的方式叫做 Autograph.

计算图由节点(nodes)和线(edges)组成, 节点表示操作符Operator, 或者称之为算子, 线表示计算间的依赖.
实线表示有数据传递依赖, 传递的数据即张量; 虚线通常可以表示控制依赖, 即执行先后顺序
"""

import os, pathlib
import datetime
import tensorflow as tf

# ===========================================
# Static Compute Graph in TensorFlow 1.0
# https://tensorflow.google.cn/api_docs/python/tf/compat/v1
# ===========================================
# Step 1 to Define the Compute Graph
graph = tf.compat.v1.Graph()
with graph.as_default():
    # placeholder 占位符, 执行会话 session 时会指定填充对象
    x = tf.compat.v1.placeholder(name="x", shape=[], dtype=tf.string)
    y = tf.compat.v1.placeholder(name="y", shape=[], dtype=tf.string)
    z = tf.compat.v1.string_join([x, y], name="join", separator=" ")

# Step 2 to Execute the Compute Graph 
with tf.compat.v1.Session(graph=graph) as sess:
    print("==== \033[1;33;40m The Static Compute Graph in TensorFlow 1.0 \033[0m ====")
    print(sess.run(fetches=z, feed_dict={x: "Hello", y: "World"}))


# ===========================================
# Dynamic Compute Graph in TensorFlow 2.0
# ===========================================
# Eager Execution 动态计算图在每个算子处都进行构建, 构建后立即执行
print("==== \033[1;33;40m The Dynamic Compute Graph in TensorFlow 2.0 with Eager Execution \033[0m ====")
x = tf.constant("hello")
y = tf.constant("world")
z = tf.strings.join([x,y],separator=" ")
tf.print(z)


# ===========================================
# Dynamic Compute Graph in TensorFlow 2.0
# ===========================================
# Autograph with @tf.function warpper
# TensorFlow 1.0 使用计算图分两步, 定义计算图, 会话中执行计算图
# TensorFlow 2.0 采用 Autograph 的方式使用计算图, 定义计算图变成定义函数, 执行计算图变成调用函数, 不需要使用会话, 像原始的python 语法一样自然
# 实践中, 一般会先用动态计算图调试代码, 然后在需要提高性能的的地方利用 @tf.function 切换成 Autograph 获得更高的效率
# @tf.function 的使用需要遵循一定的规范

print("==== \033[1;33;40m The Dynamic Compute Graph in TensorFlow 2.0 with AutoGraph \033[0m ====")
@tf.function
def string_join(x, y):
    z = tf.strings.join([x, y], separator=" ")
    tf.print(z)

    return z

result = string_join(tf.constant("Hello"), tf.constant("World"))
print(result)


# AutoGraph for TensorBoard
stamp_time = datetime.datetime.now().strftime(r"%Y%m%d-%H%M%S")
log_folder = r"./tensorboard/graph"
os.makedirs(log_folder, exist_ok=True)
logs = str(pathlib.Path(os.path.join(log_folder, stamp_time)))

writer = tf.summary.create_file_writer(logs)
# opening autograph tracking
tf.summary.trace_on(graph=True)

# execute autograph
result = string_join("Hello", "Wei Li")
# 将计算图信息写入日志
with writer.as_default():
    tf.summary.trace_export(
        name="autograph",
        step=0,
        profiler_outdir=logs
    )

# tensorboard --logdir ./tensorboard/graph/