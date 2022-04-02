#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: PyTorch 动态计算图 dynamic compute graph
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

import os, pathlib
import torch
# 只有这样引用 import 才是正确的 tensorboard
from torch.utils.tensorboard import SummaryWriter


"""  PyTorch 的动态计算图
1. 动态计算图简介
2. 计算图中的 Function
3. 计算图和反向传播
4. 叶子节点和非叶子节点
5. 计算图在 TensorBoard 中的可视化
"""


# =========================================== 
# Step 1. 动态计算图 dynamic compute graph
# =========================================== 
# PyTorch 的计算图由节点和边组成, 节点表示张量或者Function, 边表示张量和 Function 之间的依赖关系
# PyTorch 中的计算图是动态图, 动态主要有两重含义:
# 1. 计算图的正向传播是立即执行的, 无需等待完整的计算图创建完毕, 
#   每条语句都会在计算图中动态添加节点和边, 并立即执行正向传播得到计算结果

# 2. 计算图在反向传播后立即销毁, 下次调用需要重新构建计算图, 
#   如果在程序中使用了 backward 方法执行了反向传播, 或者利用 torch.autograd.grad 方法计算了梯度,
#  那么创建的计算图会被立即销毁, 释放存储空间, 下次调用需要重新创建
# =========================================== 

# 1. 计算图的正向传播是立即执行的
w = torch.tensor([[3.0, 1.0]], requires_grad=True)
b = torch.tensor([[3.0]], requires_grad=True)
X = torch.randn(10, 2)
Y = torch.randn(10, 1)
Y_hat = X @ w.t() + b  # Y_hat 定义后其正向传播被立即执行, 与其后面的 loss 创建语句无关
loss = torch.mean(torch.pow(Y_hat - Y, 2))

print(loss.data)
print(Y_hat.data)

# 2. 计算图在反向传播后立即销毁
# 计算图在反向传播后立即销毁, 如果需要保留计算图, 需要设置 retain_graph=True
loss.backward()  # loss.backward(retain_graph=True) 

# loss.backward() # 如果再次执行反向传播将报错 RuntimeError: Trying to backward through the graph a second time, but the saved intermediate results have already been freed. Specify retain_graph=True  when calling .backward() or autograd.grad() the first time.


# =========================================== 
# Step 2. 计算图中的 Function
# =========================================== 
# 计算图中的 张量 已经比较熟悉了, 
# 计算图中的另外一种节点是 Function, 实际上就是 PyTorch 中各种对张量操作的函数
# 这些 Function 和 Python 中的函数有一个较大的区别, 那就是它同时包括正向计算逻辑和反向传播的逻辑
# 可以通过继承 torch.autograd.Function 来创建这种支持反向传播的 Function
# =========================================== 
class MyReLU(torch.autograd.Function):
    # 正向传播逻辑, 可以用 ctx 存储一些值, 供反向传播使用
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    #反向传播逻辑
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


w1 = torch.tensor([[3.0, 1.0]], requires_grad=True)
b1 = torch.tensor([[3.0]], requires_grad=True)
X1 = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
Y1 = torch.tensor([[2.0, 3.0]])

relu = MyReLU.apply # relu 现在也可以具有正向传播和反向传播功能
Y_hat1 = relu(X1 @ w1.t() + b1)
loss1 = torch.mean(torch.pow(Y_hat1 - Y1, 2))

loss1.backward()
print(w.grad)
print(b.grad)

# Y_hat1 的梯度函数即是自定义的 MyReLU.backward
print(type(Y_hat.grad_fn))
print(Y_hat.grad_fn)


# =========================================== 
# Step 3. 计算图和反向传播
# =========================================== 
# 了解 Function 的功能, 可以简单地理解一下反向传播的原理和过程
# 理解该部分原理需要一些高等数学中求导链式法则的基础知识
# =========================================== 
x2 = torch.tensor(3.0, requires_grad=True)
y2 = x2 + 1
y3 = 2 * x2
loss2 = (y2 - y3)**2

loss2.backward()
# -----------------------------------------------
# loss.backward() 语句调用后,依次发生以下计算过程
# 1. loss 自己的 grad 梯度赋值为 1, 即对自身的梯度为 1
# 2. loss 根据其自身梯度以及关联的 backward 方法, 计算出其对应的自变量即 y1 和 y2 的梯度, 将该值赋值到 y1.grad 和 y2.grad
# 3. y2 和 y1 根据其自身梯度以及关联的 backward 方法, 分别计算出其对应的自变量 x 的梯度, x.grad 将其收到的多个梯度值累加
# 注意, 1,2,3 步骤的求梯度顺序和对多个梯度值的累加规则恰好是求导链式法则的程序表述
# 正因为求导链式法则衍生的梯度累加规则, 张量的 grad 梯度不会自动清零, 在需要的时候需要手动置零
# -----------------------------------------------


# =========================================== 
# Step 4. 叶子节点和非叶子节点
# ===========================================
# 执行下面代码, 发现 loss.grad 并不是期望的 1, 而是 None
# 类似地 y1.grad 以及 y2.grad 也是 None
# 这是为什么呢? 这是由于它们不是叶子节点张量.

# 在反向传播过程中, 只有 is_leaf=True 的叶子节点, 需要求导的张量的导数结果才会被最后保留下来
# 那么什么是叶子节点张量呢? 叶子节点张量需要满足两个条件:
# 1. 叶子节点张量是由用户直接创建的张量, 而非由某个 Function 通过计算得到的张量(中间变量)
# 2. 叶子节点张量的 requires_grad 属性必须为 True

# PyTorch 设计这样的规则主要是为了节约内存或者显存空间
# 因为几乎所有的时候, 用户只会关心他自己直接创建的张量的梯度

# 所有依赖于叶子节点张量的张量, 其 requires_grad 属性必定是 True 的
# 但其梯度值只在计算过程中被用到, 不会最终存储到 grad 属性中
# 如果需要保留中间计算结果的梯度到 grad 属性中, 可以使用 retain_grad 方法
# 如果仅仅是为了调试代码查看梯度值, 可以利用 register_hook 打印日志
# =========================================== 
x3 = torch.tensor(3.0, requires_grad=True)
y4 = x3 + 1
y5 = 2 * x3
loss3 = (y4 - y5)**2

loss3.backward(retain_graph=True)
print("loss.grad:", loss3.grad)
print("y1.grad:", y4.grad)
print("y2.grad:", y5.grad)
print(x3.grad)

print(x3.is_leaf)
print(y4.is_leaf)
print(y5.is_leaf)
print(loss3.is_leaf)


# 利用 retain_grad 可以保留非叶子节点的梯度值
# 利用 register_hook 可以查看非叶子节点的梯度值
# 非叶子节点梯度显示控制
y4.register_hook(lambda grad: print('y4 grad: ', grad))
y5.register_hook(lambda grad: print('y5 grad: ', grad))
loss3.retain_grad()

# 反向传播
loss3.backward()
print("loss.grad:", loss3.grad)
print("x.grad:", x3.grad)


# =========================================== 
# Step 5. 计算图在 TensorBoard 中的可视化
# ===========================================
# 可以利用 torch.utils.tensorboard 将计算图导出到 TensorBoard 进行可视化
# =========================================== 
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.w = torch.nn.Parameter(torch.randn(2, 1))
        self.b = torch.nn.Parameter(torch.zeros(1, 1))

    def forward(self, x):
        y = x @ self.w + self.b

        return y

net = Net()

log_tensorboard_folder = r"./tensorboard"
os.makedirs(log_tensorboard_folder, exist_ok=True)

writer = SummaryWriter(pathlib.Path(log_tensorboard_folder))
# writer = torch.utils.tensorboard.SummaryWriter(pathlib.Path(log_tensorboard_folder))
writer.add_graph(net, input_to_model=torch.rand(10, 2))
writer.close()


# ------------------------------------------------------
# TensorBoard
# https://tensorflow.google.cn/tensorboard/get_started
# ------------------------------------------------------
# %tensorboard --logdir ./tensorboard # in notebook
# tensorboard --logdir ./tensorboard # in command
# ------------------------------------------------------