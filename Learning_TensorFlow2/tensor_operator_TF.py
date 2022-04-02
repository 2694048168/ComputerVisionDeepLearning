#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The low-level API of TensorFlow2
@Python Version: 3.8.12
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-03-27
"""

""" 
TensorFlow low-level API 主要包括张量操作, 计算图和自动微分
low-level API, 可以把 TensorFlow 当做一个增强版的 numpy 来使用
TensorFlow 提供的方法比 numpy 更全面, 运算速度更快, 还可以使用 GPU 进行加速

张量的操作主要包括张量的结构操作和张量的数学运算
张量结构操作: 张量创建, 索引切片, 维度变换, 合并分割
张量数学运算: 标量运算, 向量运算, 矩阵运算, 张量运算的广播机制

Autograph 计算图, 使用 Autograph 的规范建议, Autograph 的机制原理, Autograph 和 tf.Module.
"""

import numpy as np
import tensorflow as tf

# ------------------------------
# 标量运算 scalar operation
# ------------------------------
# 张量的数学运算符可以分为标量运算符, 向量运算符, 以及矩阵运算符
# 加减乘除乘方, 以及三角函数, 指数, 对数等常见函数, 逻辑比较运算符等都是标量运算符
# 标量运算符的特点是对张量实施逐元素运算
# 有些标量运算符对常用的数学运算符进行了重载, 并且支持类似 numpy 的广播特性
# 许多标量运算符都在 tf.math 模块下
# ------------------------------
tensor_1 = tf.constant([[1.0, 2], [-3, 4.0]])
tensor_2 = tf.constant([[5.0, 6], [7.0, 8.0]])
tensor_add = tensor_1 + tensor_2 # 运算符重载
tf.print(tensor_add)
tf.print(tensor_1 - tensor_2)
tf.print(tensor_1 * tensor_2)
tf.print(tensor_1 / tensor_2)
tf.print(tensor_1 ** 2)
tf.print(tensor_1 ** 0.5)
tf.print(tensor_1 % 3) # mod 的运算符重载, 等价于 m = tf.math.mod(tensor_1, 3)
tf.print(tf.math.mod(tensor_1, 3))
tf.print(tensor_1 // 3)
tf.print(tensor_1 >= 2)
tf.print((tensor_1 >= 2) & (tensor_1 <= 3))
tf.print((tensor_1 >= 2) | (tensor_1 <= 3))
tf.print(tensor_1 == 5) # tf.equal(tensor_1, 5)
tf.print(tf.equal(tensor_1, 5))
tf.print(tf.sqrt(tensor_1))

# This op does not [broadcast]
tensor_3 = tf.constant([1.0, 8.0])
tensor_4 = tf.constant([5.0, 6.0])
tensor_5 = tf.constant([6.0, 7.0])
tf.print(tf.add_n([tensor_3, tensor_4, tensor_5]))
tf.print(tf.maximum(tensor_3, tensor_4))
tf.print(tf.minimum(tensor_3, tensor_4))

tensor_6 = tf.constant([2.6, -2.7])
tf.print(tf.math.round(tensor_6)) # 保留整数部分, 四舍五入
tf.print(tf.math.floor(tensor_6)) # 保留整数部分, 向下归整
tf.print(tf.math.ceil(tensor_6))  # 保留整数部分, 向上归整

# 幅值裁剪
tensor_7 = tf.constant([0.9, -0.8, 100.0, -20.0, 0.7])
tensor_8 = tf.clip_by_value(tensor_7, clip_value_min=-1, clip_value_max=1)
tensor_9 = tf.clip_by_norm(tensor_7, clip_norm=3)
tf.print(tensor_8)
tf.print(tensor_9)


# ------------------------------
# 向量运算 vector operations
# ------------------------------
# 向量运算符只在一个特定轴上运算, 将一个向量映射到一个标量或者另外一个向量
# 许多向量运算符都以 reduce 开头
# ------------------------------
tensor_10 = tf.range(1, 10)
tf.print(tf.reduce_sum(tensor_10))
tf.print(tf.reduce_mean(tensor_10))
tf.print(tf.reduce_max(tensor_10))
tf.print(tf.reduce_min(tensor_10))
tf.print(tf.reduce_prod(tensor_10))

# 指定 tensor 维度进行 reduce
tensor_11 = tf.reshape(tensor_10, (3, 3))
tf.print(tensor_11)
tf.print(tf.reduce_sum(tensor_11, axis=1, keepdims=True))
tf.print(tf.reduce_sum(tensor_11, axis=0, keepdims=True))

# bool 类型的 reduce
tensor_12 = tf.constant([True, False, False])
tensor_13 = tf.constant([False,False,True])
tf.print(tf.reduce_all(tensor_13))  # and logic operator
tf.print(tf.reduce_any(tensor_13))  # or logic operator
 
# 利用 tf.foldr 实现 tf.reduce_sum
tensor_14 = tf.foldr(lambda a, b: a + b, tf.range(10))
tf.print(tensor_14)

# cum 扫描累积
tensor_15 = tf.range(1, 10)
tf.print(tf.math.cumsum(tensor_15))  # the accumulative addition
tf.print(tf.math.cumprod(tensor_15)) # the cumulative

# arg 最大最小值索引
tensor_16 = tf.range(1, 10)
tf.print(tf.argmax(tensor_16))
tf.print(tf.argmin(tensor_16))

# tf.math.top_k 可以用于对张量排序
# 利用 tf.math.top_k 可以在 TensorFlow 中实现 KNN 算法
tensor_17 = tf.constant([1, 3, 7, 5, 4, 8])
values, indices = tf.math.top_k(tensor_17, 3, sorted=True)
tf.print(values)
tf.print(indices)


# ------------------------------
# 矩阵运算 matrix operations
# ------------------------------
# 矩阵必须是二维的, 类似 tf.constant([1,2,3]) 这样的不是矩阵
# 矩阵运算包括: 矩阵乘法, 矩阵转置, 矩阵逆, 矩阵求迹, 矩阵范数, 矩阵行列式, 矩阵求特征值, 矩阵分解等运算
# 除了一些常用的运算外, 大部分和矩阵有关的运算都在 tf.linalg 子包中
# ------------------------------
tensor_18 = tf.constant([[1, 2], [3, 4]])
tensor_19 = tf.constant([[2, 0], [0, 2]])
tf.print(tensor_18 @ tensor_19) # 等价于 tf.matmul(a, b)
tf.print(tf.matmul(tensor_18, tensor_19))
tf.print(tf.transpose(tensor_18)) # 矩阵转置

# 矩阵的逆, 必须为 tf.float32 或者 tf.double 数据类型
tensor_20 = tf.constant([[1.0, 2], [3, 4.0]], dtype=tf.float32)
tf.print(tf.linalg.inv(tensor_20))
# 矩阵的迹 trace
tf.print(tf.linalg.trace(tensor_20))
# 矩阵求范数
tf.print(tf.linalg.norm(tensor_20))
# 矩阵的行列式计算
tf.print(tf.linalg.det(tensor_20))
# 矩阵的特征值
print(tf.linalg.eigvals(tensor_20))

# 矩阵 QR 分解, 将一个方阵分解为一个正交矩阵 Q 和上三角矩阵 R
# QR 分解实际上是对矩阵 A 实施 Schmidt 正交化得到 Q
matrix_1 = tf.constant([[1.0, 2.0], [3.0, 4.0], [5, 6]], dtype=tf.float32)
q, r = tf.linalg.qr(matrix_1)
tf.print(q)
tf.print(r)
tf.print(q@r)

# 矩阵 SVD 奇异值分解, svd分解可以将任意一个矩阵分解为一个正交矩阵 u,一个对角阵 s 和一个正交矩阵 v.t() 的乘积
# svd 常用于矩阵压缩和降维
# 利用 svd 分解可以在 TensorFlow 中实现主成分分析降维
matrix_2  = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=tf.float32)
s, u, v = tf.linalg.svd(matrix_2)
tf.print(u, "\n")
tf.print(s, "\n")
tf.print(v, "\n")
tf.print(u@tf.linalg.diag(s)@tf.transpose(v))


# ------------------------------
# 广播机制 broadcast mechanism
# ------------------------------
# TensorFlow 的广播规则和 numpy 是一样的:
# 1. 如果张量的维度不同, 将维度较小的张量进行扩展, 直到两个张量的维度都一样
# 2. 如果两个张量在某个维度上的长度是相同的, 或者其中一个张量在该维度上的长度为 1, 那么就说这两个张量在该维度上是相容的
# 3. 如果两个张量在所有维度上都是相容的, 它们就能使用广播
# 4. 广播之后, 每个维度的长度将取两个张量在该维度长度的较大值
# 5. 在任何一个维度上, 如果一个张量的长度为 1, 另一个张量长度大于 1, 那么在该维度上, 就好像是对第一个张量进行了复制
# tf.broadcast_to 以显式的方式按照广播机制扩展张量的维度
# ------------------------------
tensor_21 = tf.constant([1, 2, 3])
tensor_22 = tf.constant([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
tf.print(tensor_22 + tensor_21) # 等价于 tensor_22 + tf.broadcast_to(tensor_21, tensor_22.shape)
tf.print(tf.broadcast_to(tensor_21, tensor_22.shape))

# 计算广播后计算结果的形状shape, 静态形状, TensorShape 类型参数
print(tf.broadcast_static_shape(tensor_21.shape, tensor_22.shape))

# 计算广播后计算结果的形状shape, 动态形状, Tensor 类型参数
tensor_23 = tf.constant([1, 2, 3])
tensor_24 = tf.constant([[1], [2], [3]])
print(tf.broadcast_dynamic_shape(tf.shape(tensor_23), tf.shape(tensor_24)))

# 广播效果
print(tensor_23 + tensor_24) # 等价于 tf.broadcast_to(tensor_23, [3, 3]) + tf.broadcast_to(tensor_24, [3, 3])