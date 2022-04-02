#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 构建模型的 3 中方法, torch.nn.Module, torch.nn.Sequential, 辅助应用模型容器
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-04-01
"""

""" PyTorch 的高阶 API
PyTorch 没有官方的高阶API, 一般通过 torch.nn.Module 来构建模型并编写自定义训练循环
为了更加方便地训练模型, 仿 keras 的 Pytorch 模型接口: torchkeras,  作为 Pytorch 的高阶 API

详细介绍 PyTorch 的高阶API 如下相关的内容:
1. 构建模型的3种方法(继承 torch.nn.Module 基类, 使用 torch.nn.Sequential, 辅助应用模型容器)
2. 训练模型的3种方法(脚本风格, 函数风格, torchkeras.Model类风格)
3. 使用GPU训练模型(单GPU训练, 多GPU训练)
"""

# ------------------------------------------------
# 构建模型的 3 种方法, 可以使用以下 3 种方式构建模型:
# 1. 继承 torch.nn.Module 基类构建自定义模型
# 2. 使用 torch.nn.Sequential 按层顺序构建模型
# 3. 继承 torch.nn.Module 基类构建模型并辅助应用模型容器进行封装 torch.nn.Sequential,torch.nn.ModuleList,torch.nn.ModuleDict
# 其中 第 1 种方式最为常见, 第 2 种方式最简单, 第 3 种方式最为灵活也较为复杂
# 推荐使用第 1 种方式构建模型
# ------------------------------------------------

import torch
from torch import nn
import torchkeras
from collections import OrderedDict


# ------------------------------------------------
# 1. 继承 torch.nn.Module 基类构建自定义模型
# ------------------------------------------------
# 继承 torch.nn.Module 基类构建自定义模型的范例
# 模型中的用到的层一般在 __init__ 函数中定义
# 然后在 forward 方法中定义模型的正向传播逻辑
class Net_1(torch.nn.Module):
    def __init__(self):
        super(Net_1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=0.1)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        y = self.sigmoid(x)

        return y


print(f"\n \033[1;31;47m The Build Model with torch.nn.Module in PyTorch Successfully. \033[0m")
print(f"\033[1;33;40m The Build Model with torch.nn.Module in PyTorch Successfully. \033[0m \n")
net_1 = Net_1()
print(net_1)
torchkeras.summary(net_1, input_shape=(3, 32, 32))


# ------------------------------------------------
# 2. 使用 torch.nn.Sequential 按层顺序构建模型
# ------------------------------------------------
# 使用 torch.nn.Sequential 按层顺序构建模型无需定义 forward 方法, 仅仅适合于简单的模型
# 以下是使用 torch.nn.Sequential 搭建模型的一些等价方法
net_2 = nn.Sequential()
net_2.add_module("conv1", nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3))
net_2.add_module("pool1", nn.MaxPool2d(kernel_size=2, stride=2))
net_2.add_module("conv2", nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5))
net_2.add_module("pool2", nn.MaxPool2d(kernel_size=2, stride=2))
net_2.add_module("dropout", nn.Dropout2d(p=0.1))
net_2.add_module("adaptive_pool", nn.AdaptiveMaxPool2d((1, 1)))
net_2.add_module("flatten", nn.Flatten())
net_2.add_module("linear1", nn.Linear(64, 32))
net_2.add_module("relu", nn.ReLU())
net_2.add_module("linear2", nn.Linear(32, 1))
net_2.add_module("sigmoid", nn.Sigmoid())

print(f"\n \033[1;31;47m The Build Model with torch.nn.Sequential in PyTorch Successfully. \033[0m")
print(f"\033[1;33;40m The Build Model with torch.nn.Sequential in PyTorch Successfully. \033[0m \n")
print(net_2)
print()

# 2 利用变长参数  这种方式构建时不能给每个层指定名称
net_3 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Dropout2d(p=0.1),
    nn.AdaptiveMaxPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

print(f"\n \033[1;31;47m The Build Model with torch.nn.Sequential in PyTorch Successfully. \033[0m")
print(f"\033[1;33;40m The Build Model with torch.nn.Sequential in PyTorch Successfully. \033[0m \n")
print(net_3)
print()

# 3 利用 OrderedDict
net_4 = nn.Sequential(OrderedDict(
          [("conv1",  nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)),
            ("pool1", nn.MaxPool2d(kernel_size=2, stride=2)),
            ("conv2", nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)),
            ("pool2", nn.MaxPool2d(kernel_size=2, stride=2)),
            ("dropout", nn.Dropout2d(p=0.1)),
            ("adaptive_pool", nn.AdaptiveMaxPool2d((1, 1))),
            ("flatten", nn.Flatten()),
            ("linear1", nn.Linear(64, 32)),
            ("relu", nn.ReLU()),
            ("linear2", nn.Linear(32, 1)),
            ("sigmoid", nn.Sigmoid())
          ])
        )

print(f"\n \033[1;31;47m The Build Model with torch.nn.Sequential in PyTorch Successfully. \033[0m")
print(f"\033[1;33;40m The Build Model with torch.nn.Sequential in PyTorch Successfully. \033[0m \n")
print(net_4)
print()
torchkeras.summary(net_4, input_shape=(3,32,32))


# ------------------------------------------------
# 3. 继承 torch.nn.Module 基类构建模型并辅助应用模型容器进行封装 torch.nn.Sequential,torch.nn.ModuleList,torch.nn.ModuleDict
# ------------------------------------------------
# 当模型的结构比较复杂时, 可以应用模型容器 
# torch.nn.Sequential, torch.nn.ModuleList, torch.nn.ModuleDict
# 对模型的部分结构进行封装

# 这样做会让模型整体更加有层次感,有时候也能减少代码量
# 注意, 在下面的范例中每次仅仅使用一种模型容器, 
# 但实际上这些模型容器的使用是非常灵活的,可以在一个模型中任意组合任意嵌套使用
# ------------------------------------------------
# 1. torch.nn.Sequential 作为模型容器
class Net_Sequential(nn.Module):
    def __init__(self):
        super(Net_Sequential, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
            nn.AdaptiveMaxPool2d((1, 1))
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        y = self.dense(x)

        return y 


print(f"\n \033[1;31;47m The Build Model with torch.nn.Sequential Container in PyTorch Successfully. \033[0m")
print(f"\033[1;33;40m The Build Model with torch.nn.Sequential Container in PyTorch Successfully. \033[0m \n")
net_5 = Net_Sequential()
print(net_5)

# ------------------------------------------------
# 2. torch.nn.ModuleList 作为模型容器
# 注意下面中的 ModuleList 不能用 Python 中的列表 list [] 代替
class Net_ModuleList(nn.Module):
    def __init__(self):
        super(Net_ModuleList, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()]
        )

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

print(f"\n \033[1;31;47m The Build Model with torch.nn.ModuleList Container in PyTorch Successfully. \033[0m")
print(f"\033[1;33;40m The Build Model with torch.nn.ModuleList Container in PyTorch Successfully. \033[0m \n")
net_6 = Net_ModuleList()
print(net_6)
torchkeras.summary(net_6, input_shape=(3, 32, 32))


# ------------------------------------------------
# 3. torch.nn.ModuleDict 作为模型容器
# 注意下面中的 ModuleDict 不能用 Python 中的字典 dict {} 代替
class Net_ModuleDict(nn.Module):
    def __init__(self):
        super(Net_ModuleDict, self).__init__()
        self.layers_dict = nn.ModuleDict({
               "conv1": nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
               "pool": nn.MaxPool2d(kernel_size=2, stride=2),
               "conv2": nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
               "dropout": nn.Dropout2d(p=0.1),
               "adaptive": nn.AdaptiveMaxPool2d((1, 1)),
               "flatten": nn.Flatten(),
               "linear1": nn.Linear(64, 32),
               "relu": nn.ReLU(),
               "linear2": nn.Linear(32, 1),
               "sigmoid": nn.Sigmoid()
              })

    def forward(self, x):
        layers = ["conv1", "pool", "conv2", "pool", "dropout", "adaptive",
                  "flatten", "linear1", "relu", "linear2", "sigmoid"]
        for layer in layers:
            x = self.layers_dict[layer](x)
        return x


print(f"\n \033[1;31;47m The Build Model with torch.nn.ModuleList Container in PyTorch Successfully. \033[0m")
print(f"\033[1;33;40m The Build Model with torch.nn.ModuleList Container in PyTorch Successfully. \033[0m \n")
net_7 = Net_ModuleDict()
print(net_7)
torchkeras.summary(net_7,input_shape=(3,32,32))