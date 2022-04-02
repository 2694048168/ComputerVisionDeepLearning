#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: PyTorch Tensor 张量数据结构
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-03-31
"""

""" Pytorch 的核心概念
Pytorch 是一个基于 Python 的机器学习库, 广泛应用于计算机视觉 CV, 自然语言处理 NLP
目前和 TensorFlow 分庭抗礼的深度学习框架, 在学术圈颇受欢迎

它主要提供了以下两种核心功能:
1. 支持 GPU 加速的张量计算
2. 方便优化模型的自动微分机制

Pytorch的主要优点:
1. 简洁易懂: Pytorch 的 API 设计的相当简洁一致, 基本上就是 tensor, autograd, nn 三级封装, 学习起来非常容易

有一个这样的段子, 说 TensorFlow 的设计哲学是 Make it complicated, 
Keras 的设计哲学是 Make it complicated and hide it, 
Pytorch 的设计哲学是 Keep it simple and stupid.

2. 便于调试: Pytorch 采用动态图, 可以像普通 Python 代码一样进行调试,

不同于 TensorFlow, Pytorch 的报错说明通常很容易看懂,
有一个这样的段子, 说你永远不可能从 TensorFlow 的报错说明中找到它出错的原因

3. 强大高效(指开发效率，并非执行效率): Pytorch 提供了非常丰富的模型组件, 可以快速实现想法, 并且运行速度很快

目前大部分深度学习相关的 Paper 都是用 PyTorch 实现的
有些研究人员表示, 从使用 TensorFlow 转换为使用 Pytorch 之后, 他们的睡眠好多了, 头发比以前浓密了, 皮肤也比以前光滑了

PyTorch 底层最核心的概念是张量 tensor, 动态计算图 dynamic compute graph, 自动微分 automatic differential
"""


import numpy as np
import torch

# ----------------------------
# Step 1. 张量数据结构
# PyTorch 的基本数据结构是张量 Tensor, 张量即多维数组, PyTorch 的张量和 numpy 中的 nd-array 类似
# 张量的数据类型和 numpy.array 基本对应, 但是不支持 str 类型
# 1. torch.float64 (torch.double)
# 2. torch.float32 (torch.float)
# 3. torch.float16
# 4. torch.int64 (torch.long)
# 5. torch.int32 (torch.int)
# 6. torch.int16
# 7. torch.int8
# 8. torch.uint8
# 9. torch.bool
# 一般神经网络建模使用的都是 torch.float32 类型

# 自动推断类型
tensor_1 = torch.tensor(1)
print("The value and data type of tensor_1 are {} and {} ,correspondingly".format(tensor_1, tensor_1.dtype))

tensor_2 = torch.tensor(2.0)
print("The value and data type of tensor_2 are {} and {} ,correspondingly".format(tensor_2, tensor_2.dtype))

tensor_3 = torch.tensor(True)
print("The value and data type of tensor_3 are {} and {} ,correspondingly".format(tensor_3, tensor_3.dtype))

# 指定数据类型
tensor_4 = torch.tensor(1, dtype=torch.int32)
print("The value and data type of tensor_4 are {} and {} ,correspondingly".format(tensor_4, tensor_4.dtype))

tensor_5 = torch.tensor(2.0, dtype=torch.double)
print("The value and data type of tensor_5 are {} and {} ,correspondingly".format(tensor_5, tensor_5.dtype))

# 使用特定类型构造函数
tensor_6 = torch.IntTensor(1)
print("The value and data type of tensor_6 are {} and {} ,correspondingly".format(tensor_6, tensor_6.dtype))

tensor_7 = torch.Tensor(np.array(2.0))
print("The value and data type of tensor_7 are {} and {} ,correspondingly".format(tensor_7, tensor_7.dtype))

tensor_8 = torch.BoolTensor(np.array([1, 0, 2, 0]))
print("The value and data type of tensor_8 are {} and {} ,correspondingly".format(tensor_8, tensor_8.dtype))

# 不同类型进行转换
tensor_9 = torch.tensor(1)
print("The value and data type of tensor_9 are {} and {} ,correspondingly".format(tensor_9, tensor_9.dtype))

# 调用 .floate method 转换为浮点类型
tensor_10 = tensor_9.float()
print("The value and data type of tensor_10 are {} and {} ,correspondingly".format(tensor_10, tensor_10.dtype))

# 使用 type function 转换为浮点类型
tensor_11 = tensor_9.type(torch.float)
print("The value and data type of tensor_11 are {} and {} ,correspondingly".format(tensor_11, tensor_11.dtype))

# 使用 .type_as method 转换为某个 tensor 相同类型
tensor_12 = tensor_9.type_as(tensor_10)
print("The value and data type of tensor_12 are {} and {} ,correspondingly".format(tensor_12, tensor_12.dtype))


# ----------------------------
# Step 2. 张量的维度
# 不同类型的数据可以用不同维度 (dimension) 的张量来表示
# 标量为 0 维张量, 向量为 1 维张量, 矩阵为 2 维张量
# 彩色图像有 rgb 三个通道, 可以表示为 3 维张量
# 视频还有时间维, 可以表示为 4 维张量
# 可以简单地总结为: 有几层中括号, 就是多少维的张量
tensor_scalar = torch.tensor(True)
print(tensor_scalar)
print("The dimension of tensor_scalar is {}".format(tensor_scalar.dim()))

tensor_vector = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(tensor_vector)
print("The dimension of tensor_vector is {}".format(tensor_vector.dim()))

tensor_matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
print(tensor_matrix)
print("The dimension of tensor_matrix is {}".format(tensor_matrix.dim()))

tensor_image = torch.tensor([[[1.0, 2.0, 3.0], 
                              [3.0, 4.0, 5.0]], 
                             [[5.0, 6.0, 7.0],
                              [7.0, 8.0, 9.0]]])
print(tensor_image)
print("The dimension of tensor_image is {}".format(tensor_image.dim()))


tensor_4dims = torch.tensor([[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]],
                        [[[5.0, 5.0], [6.0, 6.0]], [[7.0, 7.0], [8.0, 8.0]]]])  # 4维张量
print(tensor_4dims)
print("The dimension of tensor_4dims is {}".format(tensor_4dims.dim()))


# ----------------------------
# Step 3. 张量的尺寸
# 可以使用 tensor.shape 属性或者 tensor.size() 方法查看张量在每一维的长度
# 可以使用 tensor.view 方法改变张量的尺寸
# 如果 tensor.view 方法改变尺寸失败, 可以使用 tensor.reshape 方法
print(tensor_scalar.size())
print(tensor_scalar.shape)

print(tensor_vector.size())
print(tensor_vector.shape)

print(tensor_matrix.size())
print(tensor_matrix.shape)

# using tensor.view() method to change the size (shape)
tensor_13 = torch.arange(0, 12)
print(tensor_13)
print(tensor_13.shape)

tensor_14 = tensor_13.view(3, 4)
print(tensor_14)
print(tensor_14.shape)

tensor_15 = tensor_13.view(4, -1) # -1 表示该位置的长度(根据元素进行)自动推断
print(tensor_15)
print(tensor_15.shape)

# 有些操作会让张量存储结构扭曲, 直接使用 tensor.view 会失败, 可以用 tensor.reshape 方法
tensor_16 = torch.arange(0, 12).view(2, 6)
print(tensor_16)
print(tensor_16.shape)

# 转置操作让张量存储结构扭曲
tensor_17 = tensor_16.t()
print(tensor_17.is_contiguous())

# 直接使用 tensor.view 方法会失败, 可以使用 tensor.reshape 方法
# tensor_18 = tensor_17.view(3,4) # [Error] RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.

tensor_18 = tensor_17.reshape(3, 4) # 等价于 matrix34 = matrix62.contiguous().view(3, 4)
tensor_19 = tensor_17.contiguous().view(3, 4) # 等价于 matrix34 = matrix62.contiguous().view(3, 4)
print(tensor_18)
print(tensor_18.shape)
print(tensor_19.shape)


# ----------------------------
# Step 4. 张量和 Numpy 数组
# 可以用 tensor.numpy() 方法从 Tensor 得到 numpy 数组
# 可以用 torch.from_numpy() 从 numpy 数组得到 Tensor
# 这两种方法关联的 Tensor 和 numpy 数组是共享数据内存的, 如果改变其中一个, 另外一个的值也会发生改变
# 如果有需要, 可以用张量的 tensor.clone 方法拷贝张量, 中断这种关联
# 此外，还可以使用 tensor.item() 方法从 标量张量 得到对应的 Python数值
# 使用 tensor.tolist 方法从 张量 得到对应的 Python数值列表
# -----------------------------------------------------------------------

# torch.from_numpy 函数从 numpy 数组得到 Tensor
array_np = np.zeros(3)
tensor_np = torch.from_numpy(array_np)
print("before add 1:")
print(array_np)
print(tensor_np)

print("\nafter add 1:")
np.add(array_np, 1, out=array_np) # 给 array_np 增加 1, tensor_np 也随之改变
print(array_np)
print(tensor_np)


# numpy 方法从 Tensor 得到 numpy 数组
tensor_pt = torch.zeros(3)
arr_np = tensor_pt.numpy()
print("before add 1:")
print(tensor_pt)
print(arr_np)

print("\nafter add 1:")
# 使用带下划线的方法表示计算结果会返回给调用 张量
tensor_pt.add_(1) # 给 tensor_pt 增加 1, arr_np 也随之改变 
# torch.add(tensor_pt, 1, out=tensor_pt)
print(tensor_pt)
print(arr_np)


# 可以用 clone() 方法拷贝张量, 中断这种关联
tensor_20 = torch.zeros(3)
#使用 clone 方法拷贝张量, 拷贝后的张量和原始张量内存独立
arr = tensor_20.clone().numpy() # 也可以使用 tensor.data.numpy()
print("before add 1:")
print(tensor_20)
print(arr)

print("\nafter add 1:")
tensor_20.add_(1)
print(tensor_20)
print(arr)


# item 方法和 tolist 方法可以将张量转换成 Python 数值和数值列表
scalar = torch.tensor(1.0)
s = scalar.item()
print(s)
print(type(s))

tensor = torch.rand(2,2)
t = tensor.tolist()
print(t)
print(type(t))