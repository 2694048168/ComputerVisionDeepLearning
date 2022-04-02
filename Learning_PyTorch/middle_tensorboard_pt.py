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

# -------------- TensorBoard 可视化
在炼丹过程中,如果能够使用丰富的图像来展示模型的结构,指标的变化,参数的分布,输入的形态等信息,
无疑会提升对问题的洞察力,并增加许多炼丹的乐趣

TensorBoard 正是这样一个神奇的炼丹可视化辅助工具,
原是 TensorFlow 的生态系统的组件, 但它也能够很好地和 PyTorch 进行配合,
甚至在 PyTorch 中使用 TensorBoard 比 TensorFlow 中使用 TensorBoard 还要来的更加简单和自然

PyTorch 中利用 TensorBoard 可视化的大概过程如下:
step 1. 首先在 PyTorch 中指定一个目录, 创建一个 torch.utils.tensorboard.SummaryWriter 日志写入器
step 2. 然后根据需要可视化的信息, 利用日志写入器将相应信息日志写入指定的目录
step 3. 最后就可以传入日志目录作为参数启动 TensorBoard, 然后就可以在 TensorBoard 进行可视化查看各种信息

主要介绍 PyTorch 中利用 TensorBoard 进行如下方面信息的可视化的方法:
1. 可视化模型结构： writer.add_graph
2. 可视化指标变化： writer.add_scalar
3. 可视化参数分布： writer.add_histogram
4. 可视化原始图像： writer.add_image 或 writer.add_images
5. 可视化人工绘图： writer.add_figure
"""

import os, pathlib
import datetime
import torch 
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchkeras
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------
# 1. 可视化模型结构： writer.add_graph
# --------------------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.dropout = nn.Dropout2d(p=0.1)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        y = self.sigmoid(x)
        return y

net = Net()
print(net)
torchkeras.summary(net, input_shape=(3, 32, 32))


# ------------------------------------------------------
# TensorBoard
# https://tensorflow.google.cn/tensorboard/get_started
# ------------------------------------------------------
# %tensorboard --logdir path2log_folder_tensorboard # in notebook
# tensorboard --logdir path2log_folder_tensorboard # in command

path2log_folder_tensorboard = r"./tensorboard/visual"
os.makedirs(path2log_folder_tensorboard, exist_ok=True)

# python 3 建议使用 pathlib 修正由于操作系统的引起的路径分隔符不同问题 (正斜杠 "\\" and 反斜杠 "/" )
stamp_time = datetime.datetime.now().strftime(r"%Y%m%d-%H%M%S")
tensorboard_log_folder = str(pathlib.Path(os.path.join(path2log_folder_tensorboard, stamp_time))) 

writer_graph = SummaryWriter(pathlib.Path(tensorboard_log_folder))
writer_graph.add_graph(net, input_to_model=torch.rand(1, 3, 32, 32))
writer_graph.close()


# --------------------------------------------------------------
# 2. 可视化指标变化： writer.add_scalar
# --------------------------------------------------------------
# 有时候在训练过程中, 如果能够实时动态地查看 loss 和各种 metric 的变化曲线
# 那么无疑可以帮助我们更加直观地了解模型的训练情况
# 注意, writer.add_scalar 仅能对标量的值的变化进行可视化,因此它一般用于对 loss 和 metric 的变化进行可视化分析

# f(x) = a * x^2 + b * x + c 的最小值
x = torch.tensor(0.0, requires_grad=True) # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)

optimizer = torch.optim.SGD(params=[x], lr=0.01)

def func(x):
    result = a * torch.pow(x, 2) + b * x + c 
    return result

writer_scalar = SummaryWriter(tensorboard_log_folder)
for i in range(500):
    optimizer.zero_grad()
    y = func(x)
    y.backward()
    optimizer.step()
    writer_scalar.add_scalar("x", x.item(), i) # 日志中记录 x 在第 step i 的值
    writer_scalar.add_scalar("y", y.item(), i) # 日志中记录 y 在第 step i 的值

writer_scalar.close()
print("y = ", func(x).data, ";", "x = ", x.data)


# --------------------------------------------------------------
# 3. 可视化参数分布： writer.add_histogram
# --------------------------------------------------------------
# 如果需要对模型的参数 (一般非标量) 在训练过程中的变化进行可视化, 可以使用 writer.add_histogram
# 它能够观测张量值分布的直方图随训练步骤的变化趋势
# --------------------------------------------------------------
# 创建正态分布的张量模拟参数矩阵
def norm(mean, std):
    t = std * torch.randn((100, 20)) + mean
    return t

writer_histogram = SummaryWriter(tensorboard_log_folder)
for step, mean in enumerate(range(-10, 10, 1)):
    w = norm(mean, 1)
    writer_histogram.add_histogram("w", w, step)
    writer_histogram.flush()

writer_histogram.close()


# --------------------------------------------------------------
# 4. 可视化原始图像： writer.add_image 或 writer.add_images
# --------------------------------------------------------------
# 如果做图像相关的任务, 也可以将原始的图片在 TensorBoard 中进行可视化展示
# 如果只写入一张图片信息, 可以使用 writer.add_image
# 如果要写入多张图片信息, 可以使用 writer.add_images
# 也可以用 torchvision.utils.make_grid 将多张图片拼成一张图片, 然后用 writer.add_image 写入
# 注意, 传入的是代表图片信息的 PyTorch 中的张量数据

transform_train = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]) 
transform_valid = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

ds_train = torchvision.datasets.ImageFolder("./cifar2/train/",
            transform=transform_train, target_transform=lambda t: torch.tensor([t]).float())

ds_valid = torchvision.datasets.ImageFolder("./cifar2/test/",
            transform=transform_train, target_transform=lambda t: torch.tensor([t]).float())

print(ds_train.class_to_idx)

dl_train = torch.utils.data.DataLoader(ds_train, batch_size=50, shuffle=True, num_workers=0)
dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=50, shuffle=True, num_workers=0)

dl_train_iter = iter(dl_train)
images, labels = dl_train_iter.next()

# 仅查看一张图片
writer_image = SummaryWriter(tensorboard_log_folder)
writer_image.add_image('images[0]', images[0])
writer_image.close()

# 将多张图片拼接成一张图片, 中间用黑色网格分割
writer_imgs = SummaryWriter(tensorboard_log_folder)
# create grid of images
img_grid = torchvision.utils.make_grid(images)
writer_imgs.add_image('image_grid', img_grid)
writer_imgs.close()

# 将多张图片直接写入
writer_images = SummaryWriter(tensorboard_log_folder)
writer_images.add_images("images", images, global_step=0)
writer_images.close()


# --------------------------------------------------------------
# 5. 可视化人工绘图： writer.add_figure
# --------------------------------------------------------------
# 如果将 matplotlib 绘图的结果再 tensorboard 中展示, 可以使用 add_figure
# 注意, writer.add_image 不同的是, writer.add_figure 需要传入 matplotlib 的 figure 对象
print(ds_train.class_to_idx)

# visualize some image samplers using matplotlib
figure_plt = plt.figure(figsize=(8, 8))
def visualize_img_sampler(ds_train_img, num_sample):
    """Visualize some image samplers in train dataset or test dataset.

    Args:
        train_img (tuple of tensor): TensorFlow2 tensor with image and label.
        num_sample (int): the number of image samplers must be able to be squared.
    """
    for idx in range(num_sample):
        img, label = ds_train_img[idx]
        img = img.permute(1, 2, 0)
        ax = plt.subplot(int(np.sqrt(num_sample)), int(np.sqrt(num_sample)), idx+1)
        ax.imshow(img.numpy())
        ax.set_title(f"label = {label}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()
    plt.close()

visualize_img_sampler(ds_train, num_sample=9)

writer_figure = SummaryWriter(tensorboard_log_folder)
writer_figure.add_figure('figure', figure_plt, global_step=0)
writer_figure.close()