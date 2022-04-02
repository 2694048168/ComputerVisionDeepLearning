#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: PyTorch 的建模流程 图片数据建模 CIFAR-10/CIFAR-100 dataset
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-03-30
"""

""" PyTorch 的建模流程
使用 Pytorch 实现神经网络模型的一般流程包括：
1. 准备数据 date processing
2. 定义模型 define model
3. 训练模型 training model
4. 评估模型 eval model
5. 使用模型 using model
6. 保存模型 saving model

对新手来说,其中最困难的部分实际上是准备数据过程,
在实践中通常会遇到的数据类型包括结构化数据, 图片数据, 文本数据, 时间序列数据
titanic 生存预测问题, cifar2 图片分类问题, imdb电影评论分类问题, 国内新冠疫情结束时间预测问题
演示应用 PyTorch 对这四类数据的建模方法
"""

# 图片数据建模流程 CIFAR Dataset
# ----------------------------------
import os, pathlib
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torch
import torchvision
import torchkeras
# ----------------------------------


def printbar():
    nowtime = datetime.datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")
    print(f"\n ========================================= {nowtime}")

# mac 系统上 pytorch 和 matplotlib 在 jupyter 中同时跑需要更改环境变量
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

# -------------------------
# tep 1. image data preprocessing
# ------------------------------------------------------------------------------------
# CIFAR-2 数据集为 CIFAR-10 数据集的子集，只包括前两种类别 airplane 和 automobile, 这样将问题的规模减小，原理保持不变
# 训练集有 airplane 和 automobile 图片各 5000 张，测试集有 airplane 和 automobile 图片各 1000 张
# CIFAR-2 任务的目标是训练一个模型来对飞机 airplane 和机动车 automobile 两种图片进行分类
# ----------------------------
# PyTorch 中构建图片数据管道通常有三种方法:
# 1. torchvision 中的 datasets.ImageFolder 来读取图片然后用 DataLoader 来并行加载
# 2. 继承 torch.utils.data.Dataset 实现用户自定义读取逻辑然后用 DataLoader 来并行加载
# 3. 读取用户自定义数据集的通用方法, 既可以读取图片数据集, 也可以读取文本数据集
# ------------------------------------------------------------------------------------
path2images_train = r"./cifar2/train/"
path2images_test = r"./cifar2/test/"

transform_train = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
transform_valid = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

dataset_train = torchvision.datasets.ImageFolder(path2images_train, transform=transform_train, target_transform=lambda t: torch.tensor([t]).float())
dataset_valid = torchvision.datasets.ImageFolder(path2images_test, transform=transform_valid, target_transform=lambda t: torch.tensor([t]).float())

print(dataset_train.class_to_idx)

# dataset_load_train = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=4)
dataset_load_train = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=0)
dataset_load_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=32, shuffle=True, num_workers=0)
# dataset_load_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=32, shuffle=True, num_workers=4)


path2img_save_folder = r"./images"
os.makedirs(path2img_save_folder, exist_ok=True)
# visualize some image samplers using matplotlib
def visualize_img_sampler(train_img, num_sample):
    """Visualize some image samplers in train dataset or test dataset.

    Args:
        train_img (tuple of tensor): TensorFlow2 tensor with image and label.
        num_sample (int): the number of image samplers must be able to be squared.
    """
    plt.figure(figsize=(8, 8))
    for idx in range(num_sample):
        img, label = train_img[idx]
        img = img.permute(1, 2, 0) # [C, H, W] ---> [H, W, C]
        ax = plt.subplot(int(np.sqrt(num_sample)), int(np.sqrt(num_sample)), idx+1)
        ax.imshow(img.numpy())
        ax.set_title(f"label = {label.item()}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    # plt.show()
    plt.savefig(str(pathlib.Path(os.path.join(path2img_save_folder, "visual_img_data.png"))), dpi=120)
    plt.close()

visualize_img_sampler(dataset_train, 9)


# --------------------------------------------------------------
# Step 2. Define Model with PyTorch
# --------------------------------------------------------------
# 使用 PyTorch 通常有三种方式构建模型:
# 1. torch.nn.Sequential 按层顺序构建模型
# 2. 继承 torch.nn.Module 基类构建自定义模型
# 3. 继承 torch.nn.Module 基类构建模型并辅助应用模型容器进行封装
# --------------------------------------------------------------

# test torch.nn.AdaptiveMaxPool2d
pool_layer = torch.nn.AdaptiveMaxPool2d((1, 1))
tensor_imgs = torch.randn(10, 8, 32, 32)
print(pool_layer(tensor_imgs).shape )

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.dropout = torch.nn.Dropout2d(p=0.1)
        self.adaptive_pool = torch.nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = torch.nn.Flatten()
        self.linear_1 = torch.nn.Linear(64, 32)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(32, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.pool(x)
        x = self.conv_2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        y = self.sigmoid(x)

        return y

net_instance = Net()
print(net_instance)
torchkeras.summary(net_instance, input_shape=(3, 32, 32))


# ====================================
# Step 3. Training Model with PyTorch
# ====================================
# PyTorch 通常需要用户编写自定义训练循环, 训练循环的代码风格因人而异, 3 类典型的训练循环代码风格: 
# 1. 脚本形式训练循环 
# 2. 函数形式训练循环 ***
# 3. 类形式训练循环
# ====================================
model = net_instance
model.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
model.loss_func = torch.nn.BCELoss()
model.metric_func = lambda y_pred, y_true: roc_auc_score(y_true.data.numpy(), y_pred.data.numpy())
model.metric_name = "auc"

def train_step(model, features, labels):
    # 训练模式, dropout layers 发生作用
    model.train()

    # 梯度清零
    model.optimizer.zero_grad()

    # 前向传播计算损失函数
    predictions = model(features)
    loss = model.loss_func(predictions, labels)
    metric = model.metric_func(predictions, labels)

    # 反向传播计算梯度
    loss.backward()
    model.optimizer.step()

    return loss.item(), metric.item()

def valid_step(model, features, labels):
    # 预测模式, dropout layers 不发生作用
    model.eval()

    # 关闭梯度计算
    with torch.no_grad():
        predictions = model(features)
        loss = model.loss_func(predictions, labels)
        metric = model.metric_func(predictions, labels)

    return loss.item(), metric.item()


# ----------------------------------------
# Test train_step & train_step function
features, labels = next(iter(dataset_load_train))
print("The test of train_step function is {}".format(train_step(model, features, labels)))
print("The test of valid_step function is {}".format(valid_step(model, features, labels)))
# ----------------------------------------

def train_model(model, epochs, data_load_train, data_load_valid, log_step_freq):
    metric_name = model.metric_name
    df_history = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_"+metric_name])
    print("\033[1;33;40m =================== Start Training =================== \033[0m")
    printbar()

    for epoch in range(1, epochs + 1):
        # 1. training loop
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1
        for step, (features, labels) in enumerate(data_load_train, 1):
            loss, metric = train_step(model, features, labels)

            # print batch-level log-information
            loss_sum += loss
            metric_sum += metric
            if step % log_step_freq == 0:
                print(("[step = %d] loss: %.4f, "+metric_name+": %.4f") % (step, loss_sum/step, metric_sum/step))

        # 2. validing loop
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1
        for val_step, (features, labels) in enumerate(data_load_valid, 1):
            loss, metric = valid_step(model, features, labels)

            # print batch-level log-information
            val_loss_sum += loss
            val_metric_sum += metric

        # log information
        info = (epoch, loss_sum/step, metric_sum/step, val_loss_sum/val_step, val_metric_sum/val_step)
        df_history.loc[epoch - 1] = info

        # print epoch-level log-information
        print(("\nEpoch= %d, loss= %.4f, " + metric_name + "= %.4f, val_loss= %.4f, "+" val_" + metric_name+"= %.4f")%info)
        printbar()

    printbar()
    print("\033[1;33;40m =================== Finished Training =================== \033[0m")

    return df_history


epochs = 20
df_history = train_model(model, epochs, dataset_load_train, dataset_load_valid, log_step_freq=50)


# ====================================
# Step 4. Eval Model with PyTorch
# ====================================
print(df_history)

def plot_metric(df_history, metric):
    train_metrics = df_history[metric]
    val_metrics = df_history['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.show()
    # plt.savefig(str(pathlib.Path(os.path.join(img2save_folder, "metic_loss.png"))), dpi=120)
    plt.close()

plot_metric(df_history, "loss")
plot_metric(df_history, "auc")


# ====================================
# Step 5. Using Model with PyTorch
# ====================================
def predict(model, data_load):
    model.eval()
    with torch.no_grad():
        result = torch.cat([model.forward(t[0]) for t in data_load])

    return result.data

# 预测概率 probability
y_pred_probs = predict(model, dataset_load_valid)
print("The probability of Model for Test is {}".format(y_pred_probs))

# 预测类别 classes
y_pred = torch.where(y_pred_probs > 0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
print("The classes of Model for Test is {}".format(y_pred))


# ====================================
# Step 6. Saving Model with PyTorch
# ====================================
# Pytorch 有两种保存模型的方式, 都是通过调用 pickle 序列化方法实现的
# 1. 第一种方法只保存模型参数
# 2. 第二种方法保存完整模型
# 推荐使用第一种, 第二种方法可能在切换设备和目录的时候出现各种问题
# ====================================
# 保存模式参数, 以字典的形式进行保存
path2model = "./models/cifar"
os.makedirs(path2model, exist_ok=True)

print(model.state_dict().keys())
torch.save(model.state_dict(), str(pathlib.Path(os.path.join(path2model, "net_parameter.pkl"))))

net_clone = Net()
net_clone.load_state_dict(torch.load(str(pathlib.Path(os.path.join(path2model, "net_parameter.pkl")))))
y_pred_probs_1 = predict(model, dataset_load_valid)
print("The probability of Model for Test is {}".format(y_pred_probs_1))