#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The low-level API of TensorFlow2
@Python Version: 3.8.12
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-03-26
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

# ----------------------
# create a tensor
# ----------------------
tensor_1 = tf.constant([1, 2, 3], dtype=tf.float32)
tf.print(tensor_1)

tensor_2 = tf.range(1, 10, delta=2)
tf.print(tensor_2)

tensor_3 = tf.linspace(0.0, 2*3.14, 100)
tf.print(tensor_3)

tensor_4 = tf.zeros([3, 3])
tf.print(tensor_4)

tensor_5 = tf.ones([3, 3])
tf.print(tensor_5)

tensor_6 = tf.zeros_like(tensor_5, dtype=tf.float32)
tf.print(tensor_6)

tensor_7 = tf.fill([3, 2], 5)
tf.print(tensor_7)

# 均匀分布随机
tf.random.set_seed(42)
tensor_8 = tf.random.uniform([5], minval=0, maxval=10)
tf.print(tensor_8)

# 正态分布随机
tensor_9 = tf.random.normal([3, 3], mean=0.0, stddev=1.0)
tf.print(tensor_9)

# 正态分布随机, 剔除 2 倍方差以外数据重新生成
tensor_10 = tf.random.truncated_normal((5, 5), mean=0.0, stddev=1.0, dtype=tf.float32)
tf.print(tensor_10)

# 单位矩阵
tensor_11 = tf.eye(3, 3)
tf.print(tensor_11)

# 对角矩阵
tensor_12 = tf.linalg.diag([1, 2, 3])
tf.print(tensor_12)


# --------------------------------
# 索引切片 index slice of Tensor
# --------------------------------
# 张量的索引切片方式和 numpy 几乎是一样的, 切片时支持缺省参数和省略号
# 对于 tf.Variable, 可以通过索引和切片对部分元素进行修改
# 对于提取张量的连续子区域, 也可以使用 tf.slice
# 此外对于不规则的切片提取, 可以使用 tf.gather, tf.gather_nd, tf.boolean_mask
# tf.boolean_mask 功能最为强大, 它可以实现 tf.gather, tf.gather_nd 的功能, 并且 tf.boolean_mask 还可以实现布尔索引
# 如果要通过修改张量的某些元素得到新的张量, 可以使用 tf.where, tf.scatter_nd
# --------------------------------
tf.random.set_seed(42)
tensor_13 = tf.random.uniform([5, 5], minval=0, maxval=10, dtype=tf.int32)
tf.print(tensor_13)

tf.print(tensor_13[0])      # 第 0 行
tf.print(tensor_13[-1])     # 最后一行
tf.print(tensor_13[1, 3])   # 第 1 行 第 3 列
tf.print(tensor_13[1][3])   # 第 1 行 第 3 列
tf.print(tensor_13[1:4, :]) # 第 1 行至 3 行
tf.print(tf.slice(tensor_13, [1, 0], [3, 5])) # tf.slice(input, begin_vector, size_vector)
tf.print(tensor_13[1:4, :4:2]) # 第 1 行至最后一行，第 0 列到最后一列每隔两列取一列

# 对于变量来说, 可以使用索引和切片修改部分元素
tensor_14 = tf.Variable([[1, 2], [3, 4]], dtype=tf.float32)
tensor_14[1, :].assign(tf.constant([0.0, 0.0]))

tensor_15 = tf.random.uniform([3, 3, 3], minval=0, maxval=10, dtype=tf.int32)
tf.print(tensor_15)
tf.print(tensor_15[..., 1]) # 省略号 ellipsis(...) 可以表示多个冒号 colon(:)


# 上切片方式相对规则, 对于不规则的切片提取, 可以使用 tf.gather, tf.gather_nd, tf.boolean_mask
# 考虑班级成绩册,有 4 个班级, 每个班级 10 个学生, 每个学生 7 门科目成绩, 用 4×10×7 的张量来表示
tensor_scores = tf.random.uniform((4, 10, 7), minval=0, maxval=100, dtype=tf.int32)
tf.print(tensor_scores)

# rank of tensor, shape of tensor, axes of tensor
# 抽取每个班级第 0 个学生, 第 5 个学生, 第 9 个学生的全部成绩
tensor_16 = tf.gather(tensor_scores, [0,5,9], axis=1)
tf.print(tensor_16)

# 抽取每个班级第 0 个学生, 第 5 个学生, 第 9 个学生的第 1 门课程, 第 3 门课程, 第 6 门课程成绩
tensor_17 = tf.gather(tf.gather(tensor_scores, [0,5,9], axis=1), [1,3,6], axis=2)
tf.print(tensor_17)

# 抽取第 0 个班级第 0 个学生, 第 2 个班级的第 4 个学生, 第 3 个班级的第 6 个学生的全部成绩
# indices 的长度为采样样本的个数, 每个元素为采样位置的坐标
tensor_18 = tf.gather_nd(tensor_scores, indices=[(0, 0), (2, 4), (3, 6)])
tf.print(tensor_18)

# tf.gather 和 tf.gather_nd 的功能也可以用 tf.boolean_mask 来实现
# 抽取每个班级第 0 个学生, 第 5 个学生, 第 9 个学生的全部成绩
tensor_19 = tf.boolean_mask(tensor_scores, [True, False, False, False, False, True, False, False, False, True], axis=1)
tf.print(tensor_19)

# 抽取第 0 个班级第 0 个学生, 第 2 个班级的第 4 个学生, 第 3 个班级的第 6 个学生的全部成绩
tensor_20 = tf.boolean_mask(tensor_scores,
    [[True, False, False, False, False, False, False, False, False, False],
     [False, False, False, False, False, False, False, False, False, False],
     [False, False, False, False, True, False, False, False, False, False],
     [False, False, False, False, False, False, True, False, False, False]])

tf.print(tensor_20)

# 利用 tf.boolean_mask 可以实现布尔索引, 找到矩阵中小于 0 的元素
tensor_21 = tf.constant([[-1, 1, -1], [2, 2, -2], [3, -3, 3]], dtype=tf.float32)
tf.print(tensor_21, "\n")
tf.print(tf.boolean_mask(tensor_21, tensor_21 < 0), "\n") 
tf.print(tensor_21[tensor_21 < 0]) # 布尔索引, 为 boolean_mask 的语法糖形式


""" 以上这些方法仅能提取张量的部分元素值, 但不能更改张量的部分元素值得到新的张量
如果要通过修改张量的部分元素值得到新的张量, 可以使用 tf.where 和 tf.scatter_nd
tf.where 可以理解为 if 的张量版本, 此外它还可以用于找到满足条件的所有元素的位置坐标
tf.scatter_nd 的作用和 tf.gather_nd 有些相反, tf.gather_nd 用于收集张量的给定位置的元素
tf.scatter_nd 可以将某些值插入到一个给定 shape 的全 0 的张量的指定位置处
"""
# 找到张量中小于 0 的元素, 将其换成 np.nan 得到新的张量
# tf.where 和 np.where 作用类似, 可以理解为 if 的张量版本
tensor_22 = tf.constant([[-1, 1, -1], [2, 2, -2], [3, -3, 3]], dtype=tf.float32)
tensor_23 = tf.where(tensor_22 < 0, tf.fill(tensor_22.shape, np.nan), tensor_22)
tf.print(tensor_23)

#如果 where 只有一个参数, 将返回所有满足条件的位置坐标
indexes = tf.where(tensor_22 < 0)
print(indexes)

# 将张量的第 [0, 0] 和 [2, 1] 两个位置元素替换为 0 得到新的张量
tensor_24 = tensor_22 - tf.scatter_nd([[0,0], [2,1]], [tensor_22[0, 0], tensor_22[2, 1]], tensor_22.shape)
tf.print(tensor_24)

# scatter_nd 的作用和 gather_nd 有些相反, 可以将某些值插入到一个给定 shape 的全 0 的张量的指定位置处
indices = tf.where(tensor_22 < 0)
tensor_25 = tf.scatter_nd(indices, tf.gather_nd(tensor_22, indices), tensor_22.shape)
tf.print(tensor_25)

# -----------------------------
# 维度变换
# -----------------------------
# 维度变换相关函数主要有 tf.reshape, tf.squeeze, tf.expand_dims, tf.transpose
# tf.reshape 可以改变张量的形状
# tf.squeeze 可以减少维度
# tf.expand_dims 可以增加维度
# tf.transpose 可以交换维度
# tf.reshape可以改变张量的形状, 但是其本质上不会改变张量元素的存储顺序, 该操作实际上非常迅速,并且是可逆的
# -----------------------------
tensor_26 = tf.random.uniform(shape=[1, 3, 3, 2], minval=0, maxval=255, dtype=tf.int32)
tf.print(tensor_26.shape)
tf.print(tensor_26)

# convert tensor_26 shape to [3, 6]
tensor_27 = tf.reshape(tensor_26, [3, 6])
tf.print(tensor_27.shape)
tf.print(tensor_27)

tensor_28 = tf.reshape(tensor_27, [1, 3, 3, 2])
tf.print(tensor_28.shape)
tf.print(tensor_28)


# 如果张量在某个维度上只有一个元素, 利用 tf.squeeze 可以消除这个维度
# tf.reshape 相似, 它本质上不会改变张量元素的存储顺序
# 张量的各个元素在内存中是线性存储的, 其一般规律是, 同一层级中的相邻元素的物理地址也相邻
tensor_28 = tf.squeeze(tensor_26)
tf.print(tensor_28.shape)
tf.print(tensor_28)

tensor_29 = tf.expand_dims(tensor_28, axis=0) # 在第 0 维度插入长度为 1 的一个维度
tf.print(tensor_29.shape)
tf.print(tensor_29)

# tf.transpose 可以交换张量的维度，与 tf.reshape 不同, 它会改变张量元素的存储顺序
# tf.transpose 常用于图片存储格式的变换上
# Batch, Height, Width, Channel [B, H, W, C] ----> [C, H, W, B]
tensor_imgs = tf.random.uniform(shape=[100, 600, 600, 3], minval=0, maxval=255, dtype=tf.int32)
tf.print(tensor_imgs.shape)

tensor_img = tf.transpose(tensor_imgs, perm=[3, 1, 2, 0])
tf.print(tensor_img.shape)


# -------------------------------
# 合并分割 merge segmentation
# numpy 类似, 可以用 tf.concat 和 tf.stack 方法对多个张量进行合并, 可以用 tf.split 方法把一个张量分割成多个张量
# tf.concat 和 tf.stack 有略微的区别, tf.concat是连接, 不会增加维度, 而 tf.stack 是堆叠, 会增加维度
# -------------------------------
tensor_30 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
tensor_31 = tf.constant([[5.0, 6.0], [7.0, 8.0]])
tensor_32 = tf.constant([[9.0, 10.0], [11.0, 12.0]])

tensor_cat_1 = tf.concat([tensor_30, tensor_31, tensor_32], axis=0)
tf.print(tensor_cat_1)
tensor_cat_2 = tf.concat([tensor_30, tensor_31, tensor_32], axis=1)
tf.print(tensor_cat_2)

tensor_stack_1 = tf.stack([tensor_30, tensor_31, tensor_32])
tf.print(tensor_stack_1)
tensor_stack_2 = tf.stack([tensor_30, tensor_31, tensor_32], axis=1)
tf.print(tensor_stack_2)

# tf.split 是 tf.concat 的逆运算, 可以指定分割份数平均分割, 也可以通过指定每份的记录数量进行分割
# function signatures: tf.split(value, num_or_size_splits, axis)
tensor_split_1 = tf.split(tensor_cat_1, 3, axis=0)  # 指定分割份数, 平均分割
tf.print(tensor_split_1)
tf.print(type(tensor_split_1))

tensor_split_2 = tf.split(tensor_cat_1, [2, 2, 2], axis=0) # 指定每份的记录数量
tf.print(tensor_split_2)
tf.print(type(tensor_split_2))