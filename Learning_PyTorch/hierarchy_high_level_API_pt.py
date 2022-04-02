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
import torchkeras

def printbar():
    nowtime = datetime.datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")
    print(f"\n ========================================= {nowtime}")

# ===============================
# low-level API of PyTorch
# ===============================

# ===============================
# middle-level API of PyTorch
# ===============================

# ===============================
# high-level API of PyTorch
# ===============================
# PyTorch 没有官方的高阶 API,
# 一般需要用户自己实现训练循环, 验证循环, 预测循环

# pip install torchkeras
# 仿照 tf.keras.Model 的功能对 Pytorch 的 nn.Module 进行了封装
# 实现 fit, validate, predict, summary 方法, 相当于用户自定义高阶 API
# ===============================
# 通过继承上述用户自定义 Model 模型接口, 实现线性回归模型

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
dataset_1_train, dataset_1_valid = torch.utils.data.random_split(dataset_1, [int(400 * 0.7), 400 - int(400 * 0.7)])
data_load_1_train = torch.utils.data.DataLoader(dataset_1_train, batch_size=10, shuffle=True, num_workers=0)
data_load_1_valid = torch.utils.data.DataLoader(dataset_1_valid, batch_size=10, num_workers=0)

# -------------------
# Define the Model
class LinearRegression(torchkeras.Model):
    def __init__(self, net=None):
        super(LinearRegression, self).__init__(net)
        self.fc = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)

model = LinearRegression()
model.summary(input_shape=(2, ))

# -----------------------
# Training the Model
# 使用 fit 方法进行训练
def mean_absolute_error(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))

def mean_absolute_percent_error(y_pred, y_true):
    absolute_percent_error = (torch.abs(y_pred - y_true) + 1e-7) / (torch.abs(y_true) + 1e-7)
    return torch.mean(absolute_percent_error)

model.compile(loss_func=torch.nn.MSELoss(),
              optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
              metrics_dict={"mae": mean_absolute_error, "mape": mean_absolute_percent_error})

df_history = model.fit(200, dl_train=data_load_1_train, dl_val=data_load_1_valid, log_step_freq=20)


# ----------------
# 结果可视化
w, b = model.state_dict()["fc.weight"], model.state_dict()["fc.bias"]

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

# ----------------
# 评估模型
df_history.tail()

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    # plt.show()
    # plt.savefig(path2imgs, dpi=120)
    plt.close()

plot_metric(df_history, "loss")
plot_metric(df_history, "mape")

# 评估
print(model.evaluate(data_load_1_valid))

# 预测
data_load = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X))
print(model.predict(data_load)[0 : 10])
print(model.predict(data_load_1_valid)[0:10])


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
datset_2_train, dataset_2_valid = torch.utils.data.random_split(dataset_2, [int(len(dataset_2)*0.7), len(dataset_2) - int(len(dataset_2)*0.7)])

# data_load_2_train = torch.utils.data.DataLoader(dataset_2_train, batch_size=100, shuffle=True, num_workers=2)
data_load_2_train = torch.utils.data.DataLoader(datset_2_train, batch_size=100, shuffle=True, num_workers=0)
data_load_2_valid = torch.utils.data.DataLoader(dataset_2_valid, batch_size=100, num_workers=0)

# -------------------
# Define the Model
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2,4)
        self.fc2 = torch.nn.Linear(4,8) 
        self.fc3 = torch.nn.Linear(8,1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = torch.nn.Sigmoid()(self.fc3(x))

        return y

model = torchkeras.Model(Net())
model.summary(input_shape=(2, ))

# -------------------
# Training the Model
# 准确率
def accuracy(y_pred, y_true):
    y_pred = torch.where(y_pred > 0.5, torch.ones_like(y_pred, dtype=torch.float32),
                      torch.zeros_like(y_pred, dtype=torch.float32))
    acc = torch.mean(1 - torch.abs(y_true - y_pred))

    return acc

model.compile(loss_func=torch.nn.BCELoss(), optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
             metrics_dict={"accuracy": accuracy})

df_history_DNN = model.fit(100, dl_train=data_load_2_train, dl_val=data_load_2_valid, log_step_freq=10)

# -------------------
# 结果可视化
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
ax1.scatter(X_positive[:, 0], X_positive[:, 1], color="red")
ax1.scatter(X_negative[:, 0], X_negative[:, 1], color="green")
ax1.legend(["positive", "negative"])
ax1.set_title("y_true")

X_positive_pred = X[torch.squeeze(model.forward(X) >= 0.5)]
X_negative_pred = X[torch.squeeze(model.forward(X) < 0.5)]

ax2.scatter(X_positive_pred[:, 0], X_positive_pred[:, 1], color="red")
ax2.scatter(X_negative_pred[:, 0], X_negative_pred[:, 1], color="green")
ax2.legend(["positive", "negative"])
ax2.set_title("y_pred")

plt.show()
plt.close()

# 评估模型
plot_metric(df_history_DNN, "loss")
plot_metric(df_history_DNN, "accuracy")

print(model.evaluate(data_load_2_valid))
print(model.predict(data_load_2_valid)[0:10])