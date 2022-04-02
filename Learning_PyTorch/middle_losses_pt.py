#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: Pytorch middle API
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-04-01 (中国标准时间 CST) = 协调世界时(Coordinated Universal Time, UTC) + (时区)08:00
"""

""" Pytorch 的中阶 API
1. 数据管道
2. 模型层
3. 损失函数
4. TensorBoard 可视化

# --------------
损失函数 losses
一般来说,监督学习的目标函数由损失函数和正则化项组成 Objective = Loss + Regularization
PyTorch 中的损失函数一般在训练模型时候指定
注意 PyTorch 中内置的损失函数的参数和 TensorFlow 不同是 y_pred 在前, y_true 在后
而 TensorFlow是 y_true 在前, y_pred 在后

对于回归模型, 通常使用的内置损失函数是均方损失函数 torch.nn.MSELoss
对于二分类模型, 通常使用的是二元交叉熵损失函数 torch.nn.BCELoss 输入已经是sigmoid激活函数之后的结果
torch.nn.BCEWithLogitsLoss (输入尚未经过 torch.nn.Sigmoid 激活函数) 

对于多分类模型, 一般推荐使用交叉熵损失函数 torch.nn.CrossEntropyLoss
y_true需要是一维的, 是类别编码, y_pred 未经过 torch.nn.Softmax激活
此外如果多分类的 y_pred 经过了 torch.nn.LogSoftmax 激活, 可以使用 torhc.nn.NLLLoss 损失函数
(The negative log likelihood loss) 这种方法和直接使用 torch.nn.CrossEntropyLoss 等价

如果有需要, 也可以自定义损失函数, 自定义损失函数需要接收两个张量 y_pred, y_true 作为输入参数,并输出一个标量作为损失函数值
PyTorch 中的正则化项一般通过自定义的方式和损失函数一起添加作为目标函数
如果仅仅使用 L2 正则化, 也可以利用优化器的 weight_decay 参数来实现相同的效果
"""

import torch
import torchkeras
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Step 1. 内置损失函数
# --------------------------
"""内置的损失函数一般有类的实现和函数的实现两种形式
如: torch.nn.BCE 和 F.binary_cross_entropy 都是二元交叉熵损失函数, 前者是类的实现形式, 后者是函数的实现形式

实际上类的实现形式通常是调用函数的实现形式并用 torch.nn.Module 封装后得到的
一般常用的是类的实现形式, 它们封装在 torch.nn 模块下, 并且类名以 Loss 结尾
常用的一些内置损失函数说明如下:
1. nn.MSELoss 均方误差损失, 也叫做 L2 损失, 用于回归
2. nn.L1Loss L1损失, 也叫做绝对值误差损失, 用于回归
3. nn.SmoothL1Loss 平滑L1损失, 当输入在 -1 到 1 之间时, 平滑为 L2 损失, 用于回归
4. nn.BCELoss 二元交叉熵, 用于二分类, 输入已经过 nn.Sigmoid 激活, 对不平衡数据集可以用 weigths 参数调整类别权重
5. nn.BCEWithLogitsLoss 二元交叉熵, 用于二分类, 输入未经过 nn.Sigmoid 激活
6. nn.CrossEntropyLoss 交叉熵, 用于多分类, 要求 label 为稀疏编码,
    输入未经过 nn.Softmax 激活, 对不平衡数据集可以用 weigths 参数调整类别权重

7. nn.NLLLoss 负对数似然损失, 用于多分类, 要求 label 为稀疏编码, 输入经过 nn.LogSoftmax激活
8. nn.CosineSimilarity 余弦相似度,可用于多分类
9. nn.AdaptiveLogSoftmaxWithLoss 一种适合非常多类别且类别分布很不均衡的损失函数, 会自适应地将多个小类别合成一个 cluster

<<PyTorch的十八个损失函数>> https://zhuanlan.zhihu.com/p/61379965
"""
# --------------------------
y_pred = torch.tensor([[10.0, 0.0, -10.0], [8.0, 8.0, 8.0]])
y_true = torch.tensor([0, 2])

# 直接调用交叉熵损失
ce = nn.CrossEntropyLoss()(y_pred, y_true)
print(ce)

# 等价于先计算 nn.LogSoftmax 激活, 再调用 NLLLoss
y_pred_logsoftmax = nn.LogSoftmax(dim=1)(y_pred)
nll = nn.NLLLoss()(y_pred_logsoftmax, y_true)
print(nll)


# --------------------------
# Step 2. 自定义损失函数
# --------------------------
"""自定义损失函数
自定义损失函数接收两个张量 y_pred, y_true 作为输入参数, 并输出一个标量作为损失函数值
也可以对 torch.nn.Module 进行子类化, 重写 forward 方法实现损失的计算逻辑, 从而得到损失函数的类的实现

Focal Loss 的自定义实现示范, Focal Loss 是一种对 binary_crossentropy 的改进损失函数形式
它在样本不均衡和存在较多易分类的样本时相比 binary_crossentropy 具有明显的优势
它有两个可调参数, alpha 参数和 gamma 参数, 其中 alpha 参数主要用于衰减负样本的权重, gamma 参数主要用于衰减容易训练样本的权重
从而让模型更加聚焦在正样本和困难样本上, 这就是为什么这个损失函数叫做Focal Loss

5分钟理解Focal Loss与GHM——解决样本不平衡利器》 
https://zhuanlan.zhihu.com/p/80594704

Focal Loss implementation, Two Paper of Focal Loss and Improved
https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf
https://arxiv.org/abs/1811.05181

"""
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        bce = torch.nn.BCELoss(reduction="none")(y_pred, y_true)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        loss = torch.mean(alpha_factor * modulating_factor * bce)
        return loss


#困难样本
y_pred_hard = torch.tensor([[0.5],[0.5]])
y_true_hard = torch.tensor([[1.0],[0.0]])

#容易样本
y_pred_easy = torch.tensor([[0.9],[0.1]])
y_true_easy = torch.tensor([[1.0],[0.0]])

focal_loss = FocalLoss()
bce_loss = nn.BCELoss()

print("focal_loss(hard samples):", focal_loss(y_pred_hard, y_true_hard))
print("bce_loss(hard samples):", bce_loss(y_pred_hard, y_true_hard))
print("focal_loss(easy samples):", focal_loss(y_pred_easy, y_true_easy))
print("bce_loss(easy samples):", bce_loss(y_pred_easy, y_true_easy))
# 可见 focal_loss 让容易样本的权重衰减到原来的 0.0005/0.1054 = 0.00474
# 而让困难样本的权重只衰减到原来的 0.0866/0.6931=0.12496
# 因此相对而言, focal_loss 可以衰减容易样本的权重

# FocalLoss 的使用完整范例可以参考下面中
# 自定义 L1 和 L2 正则化项中的范例, 
# 该范例既演示了自定义正则化项的方法, 也演示了 FocalLoss 的使用方法
# -------------------------------------
# Step 3. 自定义 L1 and L2 正则化项
# -------------------------------------
# 通常认为 L1 正则化可以产生稀疏权值矩阵, 即产生一个稀疏模型, 可以用于特征选择
# 而 L2 正则化可以防止模型过拟合 overfitting,一定程度上, L1 也可以防止过拟合
# 下面以一个二分类问题为例, 演示给模型的目标函数添加自定义 L1 和 L2 正则化项的方法
# -------------------------------------
# 正负样本数量
num_positive, num_negative = 200, 6000

# 生成正样本,小圆环分布
r_positive = 5.0 + torch.normal(0.0, 1.0, size=[num_positive, 1]) 
theta_positive = 2 * np.pi * torch.rand([num_positive, 1])
X_positive = torch.cat([r_positive * torch.cos(theta_positive), r_positive * torch.sin(theta_positive)], axis=1)
Y_positive = torch.ones_like(r_positive)

# 生成负样本,大圆环分布
r_negative = 8.0 + torch.normal(0.0, 1.0, size=[num_negative, 1]) 
theta_negative = 2* np.pi * torch.rand([num_negative, 1])
X_negative = torch.cat([r_negative * torch.cos(theta_negative), r_negative * torch.sin(theta_negative)], axis=1)
Y_negative = torch.zeros_like(r_negative)

# 汇总样本
X = torch.cat([X_positive, X_negative], axis=0)
Y = torch.cat([Y_positive, Y_negative], axis=0)

# 可视化
plt.figure(figsize=(6, 6))
plt.scatter(X_positive[:, 0], X_positive[:, 1], color="red")
plt.scatter(X_negative[:, 0], X_negative[:, 1], color="green")
plt.legend(["positive", "negative"])
# plt.show()
plt.close()


ds = torch.utils.data.TensorDataset(X, Y)
ds_train, ds_valid = torch.utils.data.random_split(ds, [int(len(ds)*0.7), len(ds) - int(len(ds)*0.7)])
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=100, shuffle=True, num_workers=0)
dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=100, num_workers=0)


class DNNModel(torchkeras.Model):
    def __init__(self):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 8) 
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = nn.Sigmoid()(self.fc3(x))
        return y

model = DNNModel()
model.summary(input_shape=(2, ))

# -------------------------Training the Model
# 准确率
def accuracy(y_pred, y_true):
    y_pred = torch.where(y_pred > 0.5, torch.ones_like(y_pred, dtype=torch.float32),
                      torch.zeros_like(y_pred, dtype=torch.float32))
    acc = torch.mean(1 - torch.abs(y_true - y_pred))
    return acc

# L2正则化
def L2Loss(model, alpha):
    l2_loss = torch.tensor(0.0, requires_grad=True)
    for name, param in model.named_parameters():
        if 'bias' not in name: #一般不对偏置项使用正则
            l2_loss = l2_loss + (0.5 * alpha * torch.sum(torch.pow(param, 2)))
    return l2_loss

# L1正则化
def L1Loss(model, beta):
    l1_loss = torch.tensor(0.0, requires_grad=True)
    for name, param in model.named_parameters():
        if 'bias' not in name:
            l1_loss = l1_loss +  beta * torch.sum(torch.abs(param))
    return l1_loss

# 将 L2 正则和 L1 正则添加到 FocalLoss 损失, 一起作为目标函数
def focal_loss_with_regularization(y_pred, y_true):
    focal = FocalLoss()(y_pred, y_true) 
    l2_loss = L2Loss(model, 0.001) #注意设置正则化项系数
    l1_loss = L1Loss(model, 0.001)
    total_loss = focal + l2_loss + l1_loss
    return total_loss


model.compile(loss_func=focal_loss_with_regularization,
              optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
             metrics_dict={"accuracy": accuracy})

df_history = model.fit(30, dl_train=dl_train, dl_val=dl_valid, log_step_freq=30)


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


# -------------------------------------
# Step 4. 通过优化器实现 L2 正则化项
# -------------------------------------
# 如果仅仅需要使用 L2 正则化, 那么也可以利用优化器的 weight_decay 参数来实现
# weight_decay 参数可以设置参数在训练过程中的衰减, 这和 L2 正则化的作用效果等价
"""
before L2 regularization:
gradient descent: w = w - lr * dloss_dw 

after L2 regularization:
gradient descent: w = w - lr * (dloss_dw+beta*w) = (1-lr*beta)*w - lr*dloss_dw

so (1 - lr * beta) is the weight decay ratio.
"""
# PyTorch 的优化器支持一种称之为 Per-parameter options 的操作
#  就是对每一个参数进行特定的学习率, 权重衰减率指定, 以满足更为细致的要求
weight_params = [param for name, param in model.named_parameters() if "bias" not in name]
bias_params = [param for name, param in model.named_parameters() if "bias" in name]

optimizer = torch.optim.SGD([{'params': weight_params, 'weight_decay':1e-5},
                             {'params': bias_params, 'weight_decay':0}],
                            lr=1e-2, momentum=0.9)
