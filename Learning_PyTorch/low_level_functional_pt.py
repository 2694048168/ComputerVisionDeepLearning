#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: Pytorch的低阶 API
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-04-01 (中国标准时间 CST) = 协调世界时(Coordinated Universal Time, UTC) + (时区)08:00
"""

""" PyTorch 的低阶 API 主要包括张量操作, 动态计算图和自动微分
在低阶 API 层次上, 可以把 PyTorch 当做一个增强版的 numpy 来使用
Pytorch 提供的方法比 numpy 更全面, 运算速度更快, 如果需要的话, 还可以使用 GPU 进行加速

1. 张量的操作主要包括张量的结构操作和张量的数学运算
2. 张量结构操作诸如: 张量创建, 索引切片, 维度变换, 合并分割
3. 张量数学运算主要有: 标量运算, 向量运算, 矩阵运算, 张量运算的广播机制
4. 动态计算图特性, 计算图中的 Function, 计算图与反向传播

torch.nn.functional and torch.nn.Module

"""

import torch
import torch.nn.functional as F
import torchkeras


# =============================================================
# Step 1. torch.nn.functional and torch.nn.Module in PyTorch
# =============================================================
# PyTorch 的张量的结构操作和数学运算中的一些常用 API,
# 利用这些张量的 API 可以构建出神经网络相关的组件 (如激活函数, 模型层, 损失函数)
# PyTorch 和神经网络 (neural network ,nn) 相关的功能组件大多都封装在 torch.nn 模块下
# 这些功能组件的绝大部分既有函数形式实现, 也有类形式实现
# 其中 torch.nn.functional (一般引入后改名为 F) 有各种功能组件的函数实现:
# import torch.nn.functional as F
# ------------
# 1.激活函数
# ------------
# F.relu
# F.sigmoid
# F.tanh
# F.softmax

# ------------
# 2. 模型层
# ------------
# F.linear
# F.conv2d
# F.max_pool2d
# F.dropout2d
# F.embedding

# ------------
# 3. 损失函数
# ------------
# F.binary_cross_entropy
# F.mse_loss
# F.cross_entropy

# 为了便于对参数进行管理, 一般通过继承 torch.nn.Module 转换成为类的实现形式, 并直接封装在 torch.nn 模块下:
# ------------
# 1. 激活函数
# ------------
# torch.nn.ReLU
# torch.nn.Sigmoid
# torch.nn.Tanh
# torch.nn.Softmax

# ------------
# 2. 模型层
# ------------
# torch.nn.Linear
# torch.nn.Conv2d
# torch.nn.MaxPool2d
# torch.nn.Dropout2d
# torch.nn.Embedding

# ------------
# 3. 损失函数
# torch.nn.BCELoss
# torch.nn.MSELoss
# torch.nn.CrossEntropyLoss

# 实际上 torch.nn.Module 除了可以管理其引用的各种参数, 还可以管理其引用的子模块, 功能十分强大
# ------------------------------------------------------------------------------------

# ==============================================
# Step 2. torch.nn.Module 管理参数 in PyTorch
# ==============================================
# PyTorch,模型的参数是需要被优化器训练的,通常要设置参数为 requires_grad=True 的张量
# 同时, 在一个模型中, 往往有许多的参数, 要手动管理这些参数并不是一件容易的事情
# PyTorch 一般将参数用 torch.nn.Parameter 来表示, 并且用 torch.nn.Module 来管理其结构下的所有参数

# nn.Parameter 具有 requires_grad=True 属性
w = torch.nn.Parameter(torch.randn(2, 2))
print(w)
print(w.requires_grad)

# torch.nn.ParameterList 可以将多个 torch.nn.Parameter 组成一个列表
params_list = torch.nn.ParameterList([torch.nn.Parameter(torch.rand(8, i)) for i in range(1, 3)])
print(params_list)
print(params_list[0].requires_grad)

# torch.nn.ParameterDict 可以将多个 torch.nn.Parameter 组成一个字典
params_dict = torch.nn.ParameterDict({"a": torch.nn.Parameter(torch.rand(2, 2)),
                                      "b": torch.nn.Parameter(torch.zeros(2))})
print(params_dict)
print(params_dict["a"].requires_grad)
print(params_dict["b"].requires_grad)

# 可以用 torch.nn.Module 将它们管理起来
# module.parameters() 返回一个生成器, 包括其结构下的所有 parameters
module = torch.nn.Module()
module.w = w
module.params_list = params_list
module.params_dict = params_dict

num_param = 0
for param in module.parameters():
    print(param, "\n")
    num_param = num_param + 1

print("number of Parameters = ",num_param)


# 实践当中, 一般通过继承 torhc.nn.Module 来构建模块类, 并将所有含有需要学习的参数的部分放在构造函数中
#以下范例为 Pytorch 中 torch.nn.Linear 的源码的简化版本
#可以看到它将需要学习的参数放在 __init__ 构造函数中, 并在 forward 中调用 F.linear 函数来实现计算逻辑
class Linear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


# ==============================================
# Step 3. torch.nn.Module 管理子模块 in PyTorch
# ==============================================
# 一般情况下, 很少直接使用 torch.nn.Parameter 来定义参数构建模型, 而是通过一些拼装一些常用的模型层来构造模型
# 这些模型层也是继承自 torch.nn.Module 的对象,本身也包括参数, 属于自己要定义的模块的子模块
# torch.nn.Module 提供了一些方法可以管理这些子模块:
# 1. children() method: 返回生成器, 包括模块下的所有子模块
# 2. named_children() method: 返回一个生成器, 包括模块下的所有子模块, 以及它们的名字
# 3. modules() method: 返回一个生成器, 包括模块下的所有各个层级的模块, 包括模块本身
# 4. named_modules() method: 返回一个生成器, 包括模块下的所有各个层级的模块以及它们的名字, 包括模块本身

# 其中 children() 方法和 named_children() 方法较多使用
# modules() 方法和 named_modules() 方法较少使用, 其功能可以通过多个 named_children() 的嵌套使用实现。

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=10000, embedding_dim=3, padding_idx=1)
        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv_1", torch.nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5))
        self.conv.add_module("pool_1", torch.nn.MaxPool1d(kernel_size=2))
        self.conv.add_module("relu_1", torch.nn.ReLU())
        self.conv.add_module("conv_2", torch.nn.Conv1d(in_channels=16, out_channels=128, kernel_size=2))
        self.conv.add_module("pool_2", torch.nn.MaxPool1d(kernel_size=2))
        self.conv.add_module("relu_2", torch.nn.ReLU())

        self.dense = torch.nn.Sequential()
        self.dense.add_module("flatten", torch.nn.Flatten())
        self.dense.add_module("linear", torch.nn.Linear(6144,1))
        self.dense.add_module("sigmoid", torch.nn.Sigmoid())

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.conv(x)
        y = self.dense(x)
        return y

net = Net()

i = 0
for child in net.children():
    i += 1
    print(child, "\n")
print("child number", i)


i = 0
for name, child in net.named_children():
    i += 1
    print(name, ":", child, "\n")
print("child number", i)


i = 0
for module in net.modules():
    i += 1
    print(module)
print("module number:", i)


# 通过 named_children 方法找到 embedding 层, 并将其参数设置为不可训练 (相当于冻结 embedding 层)
children_dict = {name: module for name, module in net.named_children()}
print(children_dict)
embedding = children_dict["embedding"]
embedding.requires_grad_(False) # 冻结其参数

# 可以看到其第一层的参数已经不可以被训练了
for param in embedding.parameters():
    print(param.requires_grad)
    print(param.numel())

# 不可训练参数数量增加
torchkeras.summary(net, input_shape=(200, ), input_dtype=torch.LongTensor)