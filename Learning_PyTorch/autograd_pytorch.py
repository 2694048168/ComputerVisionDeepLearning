#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: PyTorch autograd 自动微分机制
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

import torch

""" PyTorch 自动微分机制 
神经网络通常依赖反向传播求梯度来更新网络参数, 求梯度过程通常是一件非常复杂而容易出错的事情
而深度学习框架可以帮助我们自动地完成这种求梯度运算

PyTorch 一般通过反向传播 backward 方法 实现这种求梯度计算,
该方法求得的梯度将存在对应自变量张量的 grad 属性下
除此之外, 也能够调用 torch.autograd.grad 函数来实现求梯度计算
这就是Pytorch的自动微分机制
"""

# ========================================
# Step 1. 利用 backward method 计算导数
# ========================================
# backward 方法通常在一个标量张量上调用, 该方法求得的梯度将存在对应自变量张量的 grad 属性下
# 如果调用的张量非标量, 则要传入一个和它同形状的 gradient 参数张量
# 相当于用该 gradient 参数张量与调用张量作向量点乘, 得到的标量结果再反向传播
# ========================================

# 1. 标量的反向传播
# f(x) = a * x^2 + b * x + c 的导数
x_scalar = torch.tensor(0.0, requires_grad=True) # x 需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
y_scalar = a * torch.pow(x_scalar, 2) + b * x_scalar + c 

y_scalar.backward()
dy_dx = x_scalar.grad
print(dy_dx)

# 2. 非标量的反向传播
# f(x) = a * x^2 + b * x + c
x_tensor = torch.tensor([[0.0, 0.0], [1.0, 2.0]], requires_grad=True) # x 需要被求导
y_tensor = a * torch.pow(x_tensor, 2) + b * x_tensor + c 
gradient = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

print("x_tensor:\n", x_tensor)
print("y_tensor:\n", y_tensor)
y_tensor.backward(gradient=gradient)
x_grad = x_tensor.grad
print("x_grad:\n", x_grad)

# 3. 非标量的反向传播可以用标量的反向传播实现
# f(x) = a * x^2 + b * x + c
x_scalar_tensor = torch.tensor([[0.0, 0.0], [1.0, 2.0]], requires_grad=True) # x需要被求导
y_scalar_tensor = a * torch.pow(x_scalar_tensor, 2) + b * x_scalar_tensor + c 

gradient_scalar_tensor = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
z = torch.sum(y_scalar_tensor * gradient_scalar_tensor)

print("x:", x_scalar_tensor)
print("y:", y_scalar_tensor)
z.backward()
x_grad_scalar_tensor = x_scalar_tensor.grad
print("x_grad:\n", x_grad_scalar_tensor)


# ============================================
# Step 2. 利用 autograd.grad method 计算导数
# ============================================
# f(x) = a * x^2 + b * x + c 的导数
x_1 = torch.tensor(0.0, requires_grad=True) # x需要被求导
y_1 = a * torch.pow(x_1, 2) + b * x_1 + c

# create_graph 设置为 True 将允许创建更高阶的导数 
dy_dx_1 = torch.autograd.grad(y_1, x_1, create_graph=True)[0]
print(dy_dx_1.data)

# 求二阶导数
dy2_dx2_1 = torch.autograd.grad(dy_dx_1, x_1)[0] 
print(dy2_dx2_1.data)


x1 = torch.tensor(1.0, requires_grad=True) # x需要被求导
x2 = torch.tensor(2.0, requires_grad=True)
y1 = x1 * x2
y2 = x1 + x2

# 允许同时对多个自变量求导数
(dy1_dx1, dy1_dx2) = torch.autograd.grad(outputs=y1, inputs=[x1, x2], retain_graph=True)
print(dy1_dx1,dy1_dx2)

# 如果有多个因变量，相当于把多个因变量的梯度结果求和
(dy12_dx1, dy12_dx2) = torch.autograd.grad(outputs=[y1, y2], inputs=[x1, x2])
print(dy12_dx1, dy12_dx2)


# ============================================
# Step 3. 利用自动微分和优化器求最小值
# ============================================
# f(x) = a * x^2 + b * x + c 的最小值
x_2 = torch.tensor(0.0, requires_grad=True) # x需要被求导
a_2 = torch.tensor(1.0)
b_2 = torch.tensor(-2.0)
c_2 = torch.tensor(1.0)

optimizer = torch.optim.SGD(params=[x_2], lr=0.01)

def func(x):
    result = a * torch.pow(x, 2) + b * x + c 
    return result

for i in range(500):
    optimizer.zero_grad()
    y_2 = func(x_2)
    y_2.backward()
    optimizer.step()

print("y = ", func(x_2).data, ";", "x = ", x_2.data)