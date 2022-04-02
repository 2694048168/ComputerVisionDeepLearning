#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: PyTorch API 层次结构 hierarchy
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-03-31
"""

""" PyTorch Hierarchy 的层次结构
PyTorch 中 5-level 不同的层次结构: 即硬件层, 内核层, 低阶API, 中阶API, 高阶API [torchkeras]
并以线性回归和DNN二分类模型为例, 直观对比展示在不同层级实现模型的特点

Pytorch的层次结构从低到高可以分成如下五层:
1. 最底层为硬件层 hardware, PyTorch 支持 CPU、GPU 加入计算资源池
2. 第二层为 C++ 实现的内核 core 
3. 第三层为 Python 实现的操作符, 提供了封装 C++ core 内核的低级 API 指令, 
    主要包括各种张量操作算子, 自动微分, 变量管理.
    如 torch.tensor, torch.cat, torch.autograd.grad, torch.nn.Module

4. 第四层为 Python 实现的模型组件, 对低级 API 进行了函数封装,
    主要包括各种模型层, 损失函数, 优化器, 数据管道等等
    如 torch.nn.Linear, torch.nn.BCE, torch.optim.Adam, torch.utils.data.DataLoader.

5. 第五层为 Python 实现的模型接口, PyTorch 没有官方的高阶API
    为了便于训练模型, 仿照 keras 中的模型接口, 使用了不到 300 行代码, 
    封装了 Pytorch 的高阶模型接口 torchkeras.Model

pip install torchkeras

"""

import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def printbar():
    nowtime = datetime.datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")
    print(f"\n ========================================= {nowtime}")

# ===============================
# low-level API of PyTorch
# ===============================

# ===============================
# middle-level API of PyTorch
# ===============================
# PyTorch 的中阶 API 主要包括各种模型层, 损失函数, 优化器, 数据管道等等

# Example 1. 线性回归模型
print("\033[1;33;40m =================== Example of Linear Regression =================== \033[0m")

# 样本数量
num_samplers = 400

# 生成测试用数据集
X = 10 * torch.rand([num_samplers, 2]) - 5.0  # torch.rand 是均匀分布 
w0 = torch.tensor([[2.0], [-3.0]])
b0 = torch.tensor([[10.0]])
Y = X @ w0 + b0 + torch.normal(0.0, 2.0, size=[num_samplers, 1])  # @表示矩阵乘法, 增加正态扰动

# 数据可视化
plt.figure(figsize=(12, 5))

ax1 = plt.subplot(121)
ax1.scatter(X[:, 0].numpy(), Y[:, 0].numpy(), color="blue", label="samples")
ax1.legend()
plt.xlabel("x1")
plt.ylabel("y", rotation=0)

ax2 = plt.subplot(122)
ax2.scatter(X[:, 1].numpy(), Y[:, 0].numpy(), color="green", label="samples")
ax2.legend()
plt.xlabel("x2")
plt.ylabel("y", rotation=0)

# plt.show()
plt.close()

# -------------------
# 构建数据管道迭代器
dataset_1 = torch.utils.data.TensorDataset(X, Y)
# data_load_1 = torch.utils.data.DataLoader(dataset_1, batch_size=10, shuffle=True, num_workers=2)
data_load_1 = torch.utils.data.DataLoader(dataset_1, batch_size=10, shuffle=True, num_workers=0)

# -------------------
# Define the Model
model = torch.nn.Linear(2, 1)
model.loss_func = torch.nn.MSELoss()
model.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# -------------------
# Training the Model
def train_step(model, features, labels):
    predictions = model(features)
    loss = model.loss_func(predictions, labels)
    loss.backward()
    model.optimizer.step()
    model.optimizer.zero_grad()
    return loss.item()

# --------------------------------------
# test train_step function 效果
features, labels = next(iter(data_load_1))
train_step(model, features, labels)
# --------------------------------------

def train_model(model, epochs):
    for epoch in range(1, epochs + 1):
        for features, labels in data_load_1:
            loss = train_step(model, features, labels)
        if epoch % 2 == 0:
            printbar()
            w = model.state_dict()["weight"]
            b = model.state_dict()["bias"]
            print("epoch = ", epoch, "loss = ", loss)
            print("w =", w)
            print("b =", b)

train_model(model, epochs=20)

# ----------------
# 结果可视化
w, b = model.state_dict()["weight"], model.state_dict()["bias"]

plt.figure(figsize=(12, 5))

ax1 = plt.subplot(121)
ax1.scatter(X[:, 0], Y[:, 0], color="blue", label="samples")
ax1.plot(X[:, 0], w[0, 0] * X[:, 0] + b[0], "-r", linewidth=5.0, label="model")
ax1.legend()
plt.xlabel("x1")
plt.ylabel("y", rotation=0)

ax2 = plt.subplot(122)
ax2.scatter(X[:, 1], Y[:, 0], color="green", label="samples")
ax2.plot(X[:, 1], w[0, 1] * X[:, 1] + b[0], "-r", linewidth=5.0, label="model")
ax2.legend()
plt.xlabel("x2")
plt.ylabel("y", rotation=0)

# plt.show()
plt.close()


# Example 2. DNN 二分类模型
print("\033[1;33;40m =================== Example of Classification with DNN =================== \033[0m")

# 正负样本数量
num_positive, num_negative = 2000, 2000

# 生成正样本, 小圆环分布
r_positive = 5.0 + torch.normal(0.0, 1.0, size=[num_positive, 1]) 
theta_positive = 2 * np.pi * torch.rand([num_positive, 1])
X_positive = torch.cat([r_positive * torch.cos(theta_positive), r_positive * torch.sin(theta_positive)], axis=1)
Y_positive = torch.ones_like(r_positive)

# 生成负样本, 大圆环分布
r_negative = 8.0 + torch.normal(0.0, 1.0, size=[num_negative, 1]) 
theta_negative = 2 * np.pi * torch.rand([num_negative, 1])
X_negative = torch.cat([r_negative * torch.cos(theta_negative), r_negative * torch.sin(theta_negative)], axis=1)
Y_negative = torch.zeros_like(r_negative)

# 汇总样本
X = torch.cat([X_positive, X_negative], axis=0)
Y = torch.cat([Y_positive, Y_negative], axis=0)

# 可视化
plt.figure(figsize=(6, 6))
plt.scatter(X_positive[:, 0].numpy(), X_positive[:, 1].numpy(), color="red")
plt.scatter(X_negative[:, 0].numpy(), X_negative[:, 1].numpy(), c="green")
plt.legend(["positive", "negative"])
# plt.show()
plt.close()

# -------------------
# 构建数据管道迭代器
dataset_2 = torch.utils.data.TensorDataset(X, Y)
# data_load_2 = torch.utils.data.DataLoader(dataset_2, batch_size=10, shuffle=True, num_workers=2)
data_load_2 = torch.utils.data.DataLoader(dataset_2, batch_size=10, shuffle=True, num_workers=0)

# -------------------
# Define the Model
class DNNModel(torch.nn.Module):
    def __init__(self):
        super(DNNModel, self).__init__()
        self.fc1 = torch.nn.Linear(2, 4)
        self.fc2 = torch.nn.Linear(4, 8) 
        self.fc3 = torch.nn.Linear(8, 1)

    # 正向传播
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = torch.nn.Sigmoid()(self.fc3(x))

        return y

    # 损失函数
    def loss_func(self, y_pred, y_true):
        return torch.nn.BCELoss()(y_pred, y_true)

    # 评估函数 (准确率)
    def metric_func(self, y_pred, y_true):
        y_pred = torch.where(y_pred > 0.5, torch.ones_like(y_pred, dtype=torch.float32),
                          torch.zeros_like(y_pred, dtype=torch.float32))

        acc = torch.mean(1 - torch.abs(y_true - y_pred))
        return acc

    # 优化器
    @property
    def optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


model_DNN = DNNModel()

# -----------------------------------------------
# 测试模型结构
(features, labels) = next(iter(data_load_2))
predictions = model_DNN(features)

loss = model_DNN.loss_func(predictions, labels)
metric = model_DNN.metric_func(predictions, labels)

print("init loss:", loss.item())
print("init metric:", metric.item())
# -----------------------------------------------

# -----------------------
# Training the Model
def train_step_DNN(model, features, labels):
    # 正向传播求损失
    predictions = model(features)
    loss = model.loss_func(predictions,labels)
    metric = model.metric_func(predictions,labels)

    # 反向传播求梯度
    loss.backward()

    # 更新模型参数
    model.optimizer.step()
    model.optimizer.zero_grad()

    return loss.item(), metric.item()

# -----------------------------------
# Test train_step function
features, labels = next(iter(data_load_2))
train_step_DNN(model_DNN, features, labels)
# -----------------------------------

def train_model_DNN(model, epochs):
    for epoch in range(1, epochs + 1):
        loss_list, metric_list = [], []
        for features, labels in data_load_2:
            lossi, metrici = train_step_DNN(model, features, labels)
            loss_list.append(lossi)
            metric_list.append(metrici)

        loss = np.mean(loss_list)
        metric = np.mean(metric_list)

        if epoch % 100 == 0:
            printbar()
            print("epoch = ", epoch, "loss = ", loss,"metric = ", metric)


train_model_DNN(model_DNN, epochs=300)

# -------------
# 结果可视化
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
ax1.scatter(X_positive[:, 0], X_positive[:, 1], color="red")
ax1.scatter(X_negative[:, 0], X_negative[:, 1], color="green")
ax1.legend(["positive", "negative"])
ax1.set_title("y_true")

X_positive_pred = X[torch.squeeze(model_DNN.forward(X) >= 0.5)]
X_negative_pred = X[torch.squeeze(model_DNN.forward(X) < 0.5)]

ax2.scatter(X_positive_pred[:, 0], X_positive_pred[:, 1], color="red")
ax2.scatter(X_negative_pred[:, 0], X_negative_pred[:, 1], color="green")
ax2.legend(["positive", "negative"])
ax2.set_title("y_pred")

plt.show()
plt.close()


# ===============================
# high-level API of PyTorch
# ===============================