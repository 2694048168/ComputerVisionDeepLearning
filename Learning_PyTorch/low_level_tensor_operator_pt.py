#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: Pytorch的低阶 API
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-03-31 (中国标准时间 CST) = 协调世界时(Coordinated Universal Time, UTC) + (时区)08:00
"""

""" PyTorch 的低阶 API 主要包括张量操作, 动态计算图和自动微分
在低阶 API 层次上, 可以把 PyTorch 当做一个增强版的 numpy 来使用
Pytorch 提供的方法比 numpy 更全面, 运算速度更快, 如果需要的话, 还可以使用 GPU 进行加速

1. 张量的操作主要包括张量的结构操作和张量的数学运算
2. 张量结构操作诸如: 张量创建, 索引切片, 维度变换, 合并分割
3. 张量数学运算主要有: 标量运算, 向量运算, 矩阵运算, 张量运算的广播机制
4. 动态计算图特性, 计算图中的 Function, 计算图与反向传播
"""

# ---------------------
# 张量的结构操作
# 张量的操作主要包括张量的结构操作和张量的数学运算
# 张量结构操作诸如: 张量创建, 索引切片, 维度变换, 合并分割
# 张量数学运算主要有: 标量运算, 向量运算, 矩阵运算, 张量运算的广播机制
# ---------------------

import torch

# ================================================
# Step 1. scalar operation of tensor in PyTorch
# ================================================
# 张量的数学运算符可以分为标量运算符, 向量运算符, 以及矩阵运算符
# 加减乘除乘方, 以及三角函数, 指数, 对数等常见函数, 逻辑比较运算符等都是标量运算符
# 标量运算符的特点是对张量实施逐元素运算
# 有些标量运算符对常用的数学运算符进行了重载, 并且支持类似 numpy 的广播特性

tensor_1 = torch.tensor([[1.0, 2], [-3, 4.0]])
tensor_2 = torch.tensor([[5.0, 6], [7, 8.0]])
print(tensor_1 + tensor_2) # "+" 运算符重载
print(tensor_1 - tensor_2)
print(tensor_1 * tensor_2)
print(tensor_1 / tensor_2)
print(tensor_1 ** 2)
print(tensor_1 ** (0.5))
print(tensor_1 % 3)
print(tensor_1 // 3)

print(tensor_1 >= 2) # torch.ge(tensor_1, 2)
print(torch.ge(tensor_1, 2)) # ge: greater_equal
print((tensor_1 >= 2) & (tensor_1 <= 3))
print((tensor_1 >= 2) | (tensor_1 <= 3))
print(tensor_1 == 5)
print(torch.eq(tensor_1, 5))
print(torch.sqrt(tensor_1))

tensor_3 = torch.tensor([1.0, 8.0])
tensor_4 = torch.tensor([5.0, 6.0])
tensor_5 = torch.tensor([6.0, 7.0])
tensor_6 = tensor_3 + tensor_4 + tensor_5
print(tensor_6)

print(torch.max(tensor_3, tensor_4))
print(torch.min(tensor_3, tensor_4))

tensor_7 = torch.tensor([2.6, -2.7])
print(torch.round(tensor_7)) # 保留整数部分, 四舍五入
print(torch.floor(tensor_7)) # 保留整数部分, 向下取整
print(torch.ceil(tensor_7)) # 保留整数部分, 向上取整
print(torch.trunc(tensor_7)) # 保留整数部分, 向 0 取整

print(torch.fmod(tensor_7, 2)) # 作除法取余数 
print(torch.remainder(tensor_7, 2)) # 作除法取剩余的部分,结果恒正

# 幅值裁剪
x = torch.tensor([0.9, -0.8, 100.0, -20.0, 0.7])
y = torch.clamp(x, min=-1, max=1)
z = torch.clamp(x, max=1)
print(y)
print(z)


# ================================================
# Step 2. vector operations of tensor in PyTorch
# ================================================
# 向量运算符只在一个特定轴上运算,将一个向量映射到一个标量或者另外一个向量
# ================================================
# 统计值
tensor_8 = torch.range(1, 9).float()
print("The sum of vector tensor is {}".format(torch.sum(tensor_8)))
print("The mean of vector tensor is {}".format(torch.mean(tensor_8)))
print("The max of vector tensor is {}".format(torch.max(tensor_8)))
print("The min of vector tensor is {}".format(torch.min(tensor_8)))
print("The multiplicative of vector tensor is {}".format(torch.prod(tensor_8)))
print("The standard deviation of vector tensor is {}".format(torch.std(tensor_8)))
print("The variance of vector tensor is {}".format(torch.var(tensor_8)))
print("The median of vector tensor is {}".format(torch.median(tensor_8)))

# 指定维度计算统计值
tensor_9 = tensor_8.view(3, 3)
print(tensor_9)
print(torch.max(tensor_9, dim=0))
print(torch.max(tensor_9, dim=1))

# cum 扫描
tensor_10 = torch.arange(1, 10)
print(torch.cumsum(tensor_10, 0))
print(torch.cumprod(tensor_10, 0))
print(torch.cummax(tensor_10, 0).values)
print(torch.cummax(tensor_10, 0).indices)
print(torch.cummin(tensor_10, 0))

# torch.sort 和 torch.topk 可以对张量排序
# 利用 torch.topk 可以在 PyTorch 中实现 KNN 算法
tensor_11 = torch.tensor([[9, 7, 8], [1, 3, 2], [5, 6, 4]]).float()
print(torch.topk(tensor_11, 2, dim=0), "\n")
print(torch.topk(tensor_11, 2, dim=1), "\n")
print(torch.sort(tensor_11, dim=1), "\n")


# ================================================
# Step 3. matrix operations of tensor in PyTorch
# ================================================
# 矩阵必须是二维的, 类似 torch.tensor([1, 2, 3]) 这样的不是矩阵
# 矩阵运算包括: 矩阵乘法, 矩阵转置, 矩阵逆, 矩阵求迹, 矩阵范数, 矩阵行列式, 矩阵求特征值, 矩阵分解等运算
# ================================================
# 矩阵乘法
tensor_12 = torch.tensor([[1, 2], [3, 4]])
tensor_13 = torch.tensor([[2, 0], [0, 2]])
print(tensor_13 @ tensor_12)  # 等价于 torch.matmul(a, b) 或 torch.mm(a, b)
print((tensor_13 @ tensor_12).equal(torch.matmul(tensor_13, tensor_12)))
print((tensor_13 @ tensor_12).equal(torch.mm(tensor_13, tensor_12)))

# 矩阵转置
tensor_14 = torch.tensor([[1.0, 2], [3, 4]])
print(tensor_14)
print(tensor_14.t())

# 矩阵逆, 必须为浮点类型
print(torch.inverse(tensor_14))

# 矩阵求 trace
print(torch.trace(tensor_14))

# 矩阵求范数
print(torch.norm(tensor_14))

# 矩阵行列式
print(torch.det(tensor_14))

# 矩阵特征值和特征向量
tensor_15 = torch.tensor([[1.0, 2], [-5, 4]], dtype=torch.float)
print(torch.eig(tensor_15, eigenvectors=True))

# 矩阵 QR 分解, 将一个方阵分解为一个正交矩阵 q 和上三角矩阵 r
# QR 分解实际上是对矩阵 A 实施 Schmidt 正交化得到 q
A  = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
q,r = torch.qr(A)
print(q, "\n")
print(r, "\n")
print((q @ r).equal(A))

# 矩阵 SVD 分解
# svd 分解可以将任意一个矩阵分解为一个正交矩阵 u,一个对角阵 s 和一个正交矩阵 v.t() 的乘积
# svd 常用于矩阵压缩和降维 利用 svd 分解可以在 Pytorch 中实现主成分分析降维
matrix_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
u, s, v = torch.svd(matrix_tensor)
print(u, "\n")
print(s, "\n")
print(v, "\n")
print((u @ torch.diag(s) @ v.t()).equal(matrix_tensor))


# ================================================
# Step 4. broadcast mechanism of tensor in PyTorch
# ================================================
# PyTorch 的广播规则和 numpy 是一样的:
# 1. 如果张量的维度不同, 将维度较小的张量进行扩展, 直到两个张量的维度都一样
# 2. 如果两个张量在某个维度上的长度是相同的, 或者其中一个张量在该维度上的长度为 1, 那么就说这两个张量在该维度上是相容的
# 3. 如果两个张量在所有维度上都是相容的, 它们就能使用广播
# 4. 广播之后, 每个维度的长度将取两个张量在该维度长度的较大值
# 5. 在任何一个维度上, 如果一个张量的长度为 1, 另一个张量长度大于 1, 那么在该维度上, 就好像是对第一个张量进行了复制
# torch.broadcast_tensors 可以将多个张量根据广播规则转换成相同的维度

tensor_16 = torch.tensor([1, 2, 3])
tensor_17 = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
print(tensor_16 + tensor_17)

tensor_16_broad, tensor_17_broad = torch.broadcast_tensors(tensor_16, tensor_17)
print(tensor_16_broad, "\n")
print(tensor_17_broad, "\n")
print(tensor_16_broad + tensor_17_broad) 
print((tensor_16_broad + tensor_17_broad).equal(tensor_17 + tensor_16)) 