#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 训练模型的 3 中方法, 脚本风格, 函数风格, torchkeras.Model 风格
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

# ----------------------------------------------------------------
# Pytorch 通常需要用户编写自定义训练循环, 训练循环的代码风格因人而异
# 3 类典型的训练循环代码风格: 脚本形式训练循环, 函数形式训练循环, 类形式训练循环
# MNIST 数据集的分类模型的训练为例, 演示 3 种训练模型的风格
# ----------------------------------------------------------------

import os, pathlib
import datetime
import torch
from torch import nn
import torchvision
import torchkeras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# ---------------------------
# Data Preparation
# ---------------------------
data_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

path2save_dataset = r"./MNIST"
os.makedirs(path2save_dataset, exist_ok=True)

dataset_train = torchvision.datasets.MNIST(root=pathlib.Path(path2save_dataset),
                                            train=True,
                                            download=True,
                                            transform=data_transform) 

dataset_valid = torchvision.datasets.MNIST(root=pathlib.Path(path2save_dataset),
                                            train=False,
                                            download=True,
                                            transform=data_transform) 

data_load_train = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=0)
data_load_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=128, shuffle=False, num_workers=0)

print("The number of the train dataset is {}".format(len(dataset_train)))
print("The number of the validation dataset is {}".format(len(dataset_valid)))


# 查看部分样本
def show_img_example(dataset_img, num_example):
    """Visualize some image samplers in train dataset or test dataset.
    Args:
        train_img (tuple of tensor): TensorFlow2 tensor with image and label.
        num_sample (int): the number of image samplers must be able to be squared.
    """
    plt.figure(figsize=(8, 8)) 
    for idx in range(num_example):
        img, label = dataset_img[idx]
        img = torch.squeeze(img)
        ax = plt.subplot(int(np.sqrt(num_example)), int(np.sqrt(num_example)), idx+1)
        ax.imshow(img.numpy())
        ax.set_title(f"label = {label}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()
    # plt.savefig(str(pathlib.Path(os.path.join(path2img_save_folder, "visual_img_data.png"))), dpi=120)
    plt.close()

show_img_example(dataset_train, num_example=9)


# ---------------------------
# Define the Model
# ---------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

model = Net()
print(model)
torchkeras.summary(model, input_shape=(1,32,32))


# -------------------------------------------
# Loss function and Metrics for the Model
# -------------------------------------------
def accuracy(y_pred, y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1).data
    return accuracy_score(y_true, y_pred_cls)

model.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
model.loss_func = nn.CrossEntropyLoss()
model.metric_func = accuracy
model.metric_name = "accuracy"


def printbar():
    nowtime = datetime.datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")
    print(f"\n ========================================= {nowtime}")



# --------------------------------------
# Step 1. 脚本风格的训练循环最为常见
# --------------------------------------
print(f"\033[1;33;40m The Training Model with Script-Style in PyTorch Successfully. \033[0m \n")

epochs = 3
log_step_freq = 100

df_history = pd.DataFrame(columns=["epoch", "loss", model.metric_name, "val_loss", "val_"+ model.metric_name]) 
print("\033[1;33;40m ================= Start Training ======================== \033[0m")
printbar()

for epoch in range(1, epochs + 1): 
    # 1, 训练循环
    model.train()
    loss_sum = 0.0
    metric_sum = 0.0

    step = 1
    for step, (features, labels) in enumerate(data_load_train, 1):
        # step 1. 梯度清零
        model.optimizer.zero_grad()

        # step 2. 正向传播求损失
        predictions = model(features)
        loss = model.loss_func(predictions, labels)
        metric = model.metric_func(predictions, labels)

        # step 3. 反向传播求梯度
        loss.backward()
        model.optimizer.step()

        # 打印 batch-level logger 级别日志
        loss_sum += loss.item()
        metric_sum += metric.item()
        if step % log_step_freq == 0:   
            print(("[step = %d] loss: %.4f, "+model.metric_name+": %.4f") % (step, loss_sum/step, metric_sum/step))

    # 2. 验证循环
    model.eval()
    val_loss_sum = 0.0
    val_metric_sum = 0.0

    val_step = 1
    for val_step, (features, labels) in enumerate(data_load_valid, 1):
        with torch.no_grad():
            predictions = model(features)
            val_loss = model.loss_func(predictions, labels)
            val_metric = model.metric_func(predictions, labels)

        val_loss_sum += val_loss.item()
        val_metric_sum += val_metric.item()

    # 3. 记录日志
    info = (epoch, loss_sum/step, metric_sum/step, val_loss_sum/val_step, val_metric_sum/val_step)
    df_history.loc[epoch - 1] = info

    # 打印 epoch-level logger 级别日志
    print(("\nEPOCH = %d, loss = %.4f,"+ model.metric_name +"  = %.4f, val_loss = %.4f, "+"val_"+ model.metric_name+" = %.4f") %info)
    printbar()

printbar()
print("\033[1;33;40m ================= Finishing Training ======================== \033[0m")
print(df_history)



# --------------------------------------
# Step 2. 函数风格的训练循环
# 该风格在脚本形式上作了简单的函数封装
# --------------------------------------
def train_step(model, features, labels):
    # 训练模式, dropout层发生作用
    model.train()

    # 梯度清零
    model.optimizer.zero_grad()

    # 正向传播求损失
    predictions = model(features)
    loss = model.loss_func(predictions,labels)
    metric = model.metric_func(predictions,labels)

    # 反向传播求梯度
    loss.backward()
    model.optimizer.step()

    return loss.item(), metric.item()

@torch.no_grad()
def valid_step(model, features, labels):
    # 预测模式, dropout层不发生作用
    model.eval()

    predictions = model(features)
    loss = model.loss_func(predictions,labels)
    metric = model.metric_func(predictions,labels)

    return loss.item(), metric.item()


# --------------------------------------
# Test train_step function
features, labels = next(iter(data_load_train))
print(train_step(model, features, labels))
print("The Test Passing for train_step function")
# --------------------------------------


def train_model(model, epochs, dl_train, dl_valid, log_step_freq):
    metric_name = model.metric_name
    df_history = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_"+metric_name])
    print("\033[1;33;40m ================= Start Training ======================== \033[0m")
    printbar()

    for epoch in range(1, epochs + 1):  
        # 1 训练循环
        loss_sum = 0.0
        metric_sum = 0.0

        step = 1
        for step, (features, labels) in enumerate(dl_train, 1):
            loss, metric = train_step(model, features, labels)

            # 打印 batch-level logger information 级别日志
            loss_sum += loss
            metric_sum += metric
            if step % log_step_freq == 0:
                print(("[step = %d] loss: %.4f, "+metric_name+": %.4f") % (step, loss_sum/step, metric_sum/step))

        # 2 验证循环
        val_loss_sum = 0.0
        val_metric_sum = 0.0

        val_step = 1
        for val_step, (features, labels) in enumerate(dl_valid, 1):
            val_loss,val_metric = valid_step(model, features, labels)

            val_loss_sum += val_loss
            val_metric_sum += val_metric

        # 3. 记录日志
        info = (epoch, loss_sum/step, metric_sum/step, val_loss_sum/val_step, val_metric_sum/val_step)
        df_history.loc[epoch - 1] = info

        # 打印epoch级别日志
        print(("\nEPOCH = %d, loss = %.4f,"+ metric_name + "  = %.4f, val_loss = %.4f, "+"val_"+ metric_name+" = %.4f")%info)

    printbar()
    print("\033[1;33;40m ================= Finishing Training ======================== \033[0m")
    return df_history

df_history = train_model(model, epochs=3, dl_train=data_load_train, dl_valid=data_load_valid, log_step_freq=100)
print(df_history)


# --------------------------------------
# Step 3. 类形式风格的训练
# 此处使用 torchkeras 中定义的模型接口构建模型,
# 并调用 compile 方法和 fit 方法训练模型
# --------------------------------------
model_keras = torchkeras.Model(Net())

model_keras.compile(loss_func=nn.CrossEntropyLoss(),
             optimizer=torch.optim.Adam(model.parameters(), lr=0.02),
             metrics_dict={"accuracy": accuracy})

df_history = model_keras.fit(3, dl_train=data_load_train, dl_val=data_load_valid, log_step_freq=100) 
print(df_history)