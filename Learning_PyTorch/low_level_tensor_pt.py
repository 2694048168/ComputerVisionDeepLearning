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

# ====================================
# Step 1. create tensor in PyTorch
# ====================================
# 张量创建的许多方法和 numpy 中创建 array 的方法很像
tensor_1 = torch.tensor([1, 2, 3], dtype=torch.float)
print("The value of the tensor_1 is {}".format(tensor_1))

tensor_2 = torch.arange(1, 10, step=2)
print("The value of the tensor_2 is {}".format(tensor_2))

tensor_3= torch.linspace(0.0, 2 * 3.14, 10)
print("The value of the tensor_3 is\n {}".format(tensor_3))

tensor_4 = torch.zeros((3,3))
print("The value of the tensor_4 is\n {}".format(tensor_4))

tensor_5 = torch.ones((3,3), dtype=torch.int)
tensor_6 = torch.zeros_like(tensor_5, dtype=torch.float)
print("The value of the tensor_5 is\n {}".format(tensor_5))
print("The value of the tensor_6 is\n {}".format(tensor_6))

tensor_7 = torch.fill_(tensor_6, 5)
print("The value of the tensor_7 is\n {}".format(tensor_7))


# 均匀随机分布 uniform random distribution
torch.manual_seed(42)
minval, maxval = 0, 10
tensor_uniform_dist = minval + (maxval - minval) * torch.rand([5])
print("The value of the random uniform distribution is\n {}".format(tensor_uniform_dist))

# 正态分布随机 normal distribution random
tensor_normal_dist = torch.normal(mean=torch.zeros(3, 3), std=torch.ones(3, 3))
print("The value of the random normal distribution is\n {}".format(tensor_normal_dist))

# 正态分布随机
mean, std = 2, 5
tensor_normal = std * torch.randn((3, 3)) + mean
print("The value of the random normal distribution is\n {}".format(tensor_normal))

# 整数随机排列 integer random
tensor_integer_random = torch.randperm(20)
print("The value of the integer random is\n {}".format(tensor_integer_random))

# 特殊矩阵
tensor_I = torch.eye(3, 3) # 单位矩阵 unit matrix
print("The value of the unit matrix is\n {}".format(tensor_I))

tensor_diag = torch.diag(torch.tensor([1, 2, 3])) # 对角矩阵 diagonal matrix
print("The value of the diagonal matrix is\n {}".format(tensor_diag))


# ==============================================
# Step 2. slice indices of tensor in PyTorch
# ==============================================
# 张量的索引切片方式和 numpy 几乎是一样的, 切片时支持缺省参数和省略号
# 可以通过索引和切片对部分元素进行修改
# 对于不规则的切片提取,可以使用 torch.index_select, torch.masked_select, torch.take
# 如果要通过修改张量的某些元素得到新的张量, 可以使用 torch.where, torch.masked_fill, torch.index_fill
# ==============================================
torch.manual_seed(0)
minval, maxval = 0, 10
tensor_8 = torch.floor(minval + (maxval - minval) * torch.rand([5, 5])).int()
print(tensor_8)

# 第 0 行
print(tensor_8[0])

# 倒数第一行
print(tensor_8[-1])

# 第 1 行 第 3 列
print(tensor_8[1, 3])
print(tensor_8[1][3])

# 第 1 行到第 3 行
print(tensor_8[1:4, :])

# 第 1 行至最后一行, 第 0 列到最后一列, 每隔两列取一列
print(tensor_8[1:, : :2])

# ---------------------------------
# 可以使用索引和切片修改部分元素
tensor_9 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
tensor_9.data[1, :] = torch.tensor([0.0, 0.0])
print(tensor_9)

tensor_10 = torch.arange(27).view(3, 3, 3)
print(tensor_10)

# 省略号可以表示多个冒号
print(tensor_10[..., 1])

# ------------------------------------------
# 以上切片方式相对规则, 对于不规则的切片提取,
# 可以使用 torch.index_select, torch.take, torch.gather, torch.masked_select
# 考虑班级成绩册的例子, 有 4 个班级, 每个班级 10 个学生, 每个学生 7 门科目成绩, 用一个 4×10×7 的张量来表示
minval = 0
maxval = 100
student_scores = torch.floor(minval + (maxval - minval) * torch.rand([4, 10, 7])).int()
print(student_scores)
print(student_scores.shape)

# 抽取每个班级第 0 个学生, 第 5 个学生, 第 9 个学生的全部成绩
print(torch.index_select(student_scores, dim=1, index=torch.tensor([0, 5, 9])))

# 抽取每个班级第 0 个学生, 第 5 个学生, 第 9 个学生的第 1 门课程, 第 3 门课程, 第 6 门课程成绩
print(torch.index_select(torch.index_select(student_scores, dim=1, index=torch.tensor([0, 5, 9])), 
                        dim=2, index=torch.tensor([1, 3, 6])))

# 抽取第 0 个班级第 0 个学生的第 0 门课程,
# 第 2 个班级的第 4 个学生的第 1 门课程,
# 第 3 个班级的第 9 个学生第 6 门课程成绩,
# torch.take 将输入看成一维数组, 输出和 index 同形状
print(torch.take(student_scores, torch.tensor([0*10*7+0, 2*10*7+4*7+1, 3*10*7+9*7+6])))

# 抽取分数大于等于 80 分的分数(布尔索引)
# 结果是 1 维张量
print(torch.masked_select(student_scores, student_scores >= 80))

# --------------------------------------------------------------------------
# 以上这些方法仅能提取张量的部分元素值, 但不能更改张量的部分元素值得到新的张量
# 如果要通过修改张量的部分元素值得到新的张量,
# 可以使用 torch.where, torch.index_fill 和 torch.masked_fill
# 1. torch.where 可以理解为 if 的张量版本
# 2. torch.index_fill 的选取元素逻辑和 torch.index_select 相同
# 3. torch.masked_fill 的选取元素逻辑和 torch.masked_select 相同
# --------------------------------------------------------------------------

# 如果分数大于 60 分, 赋值成 1, 否则赋值成 0
if_pass = torch.where(student_scores > 60, torch.tensor(1), torch.tensor(0))
print(if_pass)

# 将每个班级第 0 个学生, 第 5 个学生, 第 9 个学生的全部成绩赋值成满分
new_score_1 = torch.index_fill(student_scores, dim=1, index=torch.tensor([0, 5, 9]), value=100)

# 等价于 student_scores.index_fill(dim=1, index=torch.tensor([0, 5, 9]), value=100)
new_score_2 = student_scores.index_fill(dim=1, index=torch.tensor([0, 5, 9]), value=100)
print()
print("\033[1;33;40m ====================================== \033[0m")
print(new_score_1 == new_score_2)
print(new_score_1)

# 将分数小于 60 分的分数赋值成 60 分
new_score_3 = torch.masked_fill(student_scores, student_scores < 60, 60)
# 等价于 student_scores.masked_fill(scores<60,60)
new_score_4 = student_scores.masked_fill(student_scores < 60, 60)
print()
print("\033[1;33;40m ====================================== \033[0m")
print(new_score_1 == new_score_2)
print(new_score_1)


# ========================================================
# Step 3. dimension transformation of tensor in PyTorch
# ========================================================
# 维度变换相关函数主要有 torch.reshape (或者调用张量的 view 方法)
# torch.squeeze, torch.unsqueeze, torch.transpose
# 1. torch.reshape 可以改变张量的形状
# 2. torch.squeeze 可以减少维度
# 3. torch.unsqueeze 可以增加维度
# 4. torch.transpose 可以交换维度
# ========================================================
# 张量的 view 方法有时候会调用失败, 可以使用 reshape 方法
torch.manual_seed(42)
minval, maxval = 0, 255
tensor_11 = (minval + (maxval - minval) * torch.rand([1, 3, 3, 2])).int()
print(tensor_11.shape)
print(tensor_11)

# 改成 [3, 6] 形状的张量
tensor_12 = tensor_11.view([3, 6]) # torch.reshape(tensor_11, [3,6])
tensor_13 = torch.reshape(tensor_11, [3, 6])
print(tensor_12.shape == tensor_13.shape)
print(tensor_12)

# 改回成 [1, 3, 3, 2] 形状的张量
tensor_14 = torch.reshape(tensor_12, [1, 3, 3, 2]) # tensor_12.view([1, 3, 3, 2]) 
tensor_15 = tensor_12.view([1, 3, 3, 2])
print(tensor_14.shape == tensor_15.shape)
print(tensor_15)

# 如果张量在某个维度上只有一个元素, 利用 torch.squeeze 可以消除这个维度
# torch.unsqueeze 的作用和 torch.squeeze 的作用相反
tensor_16 = torch.tensor([[1.0, 2.0]])
tensor_17 = torch.squeeze(tensor_16)
print(tensor_16)
print(tensor_17)
print(tensor_16.shape)
print(tensor_17.shape)

# 在第 0 维插入长度为 1 的一个维度
tensor_18 = torch.unsqueeze(tensor_17, axis=0)  
print(tensor_17)
print(tensor_18)
print(tensor_17.shape)
print(tensor_18.shape)

# torch.transpose 可以交换张量的维度, torch.transpose 常用于图片存储格式的变换上, 通道维度的顺序
# 如果是二维的矩阵,通常会调用矩阵的转置方法 matrix.t(), 等价于 torch.transpose(matrix, 0, 1)
minval = 0
maxval = 255
# Batch, Height, Width, Channel
data_image = torch.floor(minval + (maxval - minval) * torch.rand([100, 256, 256, 3])).int()
print(data_image.shape)

# 转换成 Pytorch 默认的图片格式 Batch, Channel, Height, Width 
# 需要交换两次
data_img = torch.transpose(torch.transpose(data_image, 1, 2), 1, 3)
print(data_img.shape)

matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(matrix)
print(matrix.t()) # 等价于 torch.transpose(matrix, 0, 1)


# ========================================================
# Step 4. 合并和分割 of tensor in PyTorch
# ========================================================
# ========================================================
# 可以用 torch.cat 方法和 torch.stack 方法将多个张量合并,
# 可以用 torch.split 方法把一个张量分割成多个张量
# torch.cat 和 torch.stack 有略微的区别, torch.cat 是连接, 不会增加维度, 而 torch.stack 是堆叠, 会增加维度
tensor_19 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
tensor_20 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
tensor_21 = torch.tensor([[9.0, 10.0], [11.0, 12.0]])

tensor_cat = torch.cat([tensor_19, tensor_20, tensor_21], dim=0)
print(tensor_cat.shape)
print(tensor_cat)

tensor_cat_1 = torch.cat([tensor_19, tensor_20, tensor_21], axis=1)
print(tensor_cat_1.shape)
print(tensor_cat_1)

tensor_stack = torch.stack([tensor_19, tensor_20, tensor_21], axis=0) # PyTorch 中 dim 和 axis 可以混用
print(tensor_stack.shape)
print(tensor_stack)

tensor_stack_1 = torch.stack([tensor_19, tensor_20, tensor_21], dim=1)
print(tensor_stack_1.shape)
print(tensor_stack_1)

# ----------------------------------------
# torch.split 是 torch.cat 的逆运算
# 可以指定分割份数平均分割, 也可以通过指定每份的记录数量进行分割
print(tensor_cat, tensor_cat.shape)
tensor_20, tensor_21, tensor_22 = torch.split(tensor_cat, split_size_or_sections=2, dim=0) # 每份 2 个进行分割
print(tensor_20, tensor_20.shape)
print(tensor_21, tensor_21.shape)
print(tensor_22, tensor_22.shape)

tensor_23, tensor_24, tensor_25 = torch.split(tensor_cat, split_size_or_sections=[4, 1, 1], dim=0) # 每份分别为 [4, 1, 1]
print(tensor_23, tensor_23.shape)
print(tensor_24, tensor_24.shape)
print(tensor_25, tensor_25.shape)