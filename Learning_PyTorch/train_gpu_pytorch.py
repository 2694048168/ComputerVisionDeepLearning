#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 使用 GPU 训练模型 in PyTorch
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

# -------------------------------------------
# 使用 GPU 训练模型, 深度学习的训练过程常常非常耗时
# 训练过程的耗时主要来自于两个部分: 一部分来自数据准备; 另一部分来自参数迭代
# 当数据准备过程还是模型训练时间的主要瓶颈时, 可以使用更多进程来准备数据
# 当参数迭代过程成为训练时间的主要瓶颈时, 通常的方法是应用 GPU 来进行加速
# Pytorch 中使用 GPU 加速模型非常简单, 只要将模型和数据移动到 GPU 上, 核心代码只有以下几行
# -------------------------------------------
"""
# 定义模型
model = Model() 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device) # 移动模型到 cuda

# 训练模型
train_model()

features = features.to(device) # 移动数据到 cuda
labels = labels.to(device) # 或者 labels = labels.cuda() if torch.cuda.is_available() else labels

"""

# -----------------------------------------------------------
# 如果要使用多个 GPU 训练模型,
# 也非常简单, 只需要在将模型设置为数据并行风格模型
# 则模型移动到 GPU 上之后, 会在每一个 GPU 上拷贝一个副本
# 并把数据平分到各个 GPU 上进行训练, 核心代码如下:
# -----------------------------------------------------------
""" 
# 定义模型
model = Model()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model) # 包装为并行风格模型

# 训练模型
train_model()

features = features.to(device) # 移动数据到cuda
labels = labels.to(device) # 或者 labels = labels.cuda() if torch.cuda.is_available() else labels

"""

import os, pathlib
import time
import torch 
from torch import nn 
import torchvision
import torchkeras
from sklearn.metrics import accuracy_score


# ------------------------------
# 1. check the GPU information
print("The GPU is availabe is : {}".format(torch.cuda.is_available()))

num_gpu = torch.cuda.device_count() 
print("The number of GPU in this Host are : {}".format(num_gpu))

# 2. tensor in CPU and GPU
tensor = torch.rand(100, 100)
# tensor_gpu = tensor.to("cuda:0")
tensor_gpu = tensor.cuda()
print("The device of tensor_gpu is : {}".format(tensor_gpu.device))
print(tensor_gpu.is_cuda)

# tensor_cpu = tensor_gpu.to("cpu")
tensor_cpu = tensor_gpu.cpu()
print("The device of tensor_cpu is : {}".format(tensor_cpu.device))
print(tensor_cpu.is_cuda)

# 3. 将模型中的全部张量移动到 gpu 上
net = nn.Linear(2, 1)
print("The parameters of Model is in GPU : ", end="")
print(next(net.parameters()).is_cuda)

net.to("cuda:0") # 将模型中的全部参数张量依次到 GPU 上, 注意无需重新赋值为 net = net.to("cuda:0")
print("The parameters of Model is in GPU : ", end="")
print(next(net.parameters()).is_cuda)
print(next(net.parameters()).device)

# 4. 创建支持多个 gpu 数据并行的模型
linear = nn.Linear(2, 1)
model = nn.DataParallel(linear)
print(model.device_ids)
print(next(model.module.parameters()).device) 

path2save_model = r"./models/gpu"
os.makedirs(path2save_model, exist_ok=True)

# 注意保存参数时要指定保存 model.module 的参数
torch.save(model.module.state_dict(), pathlib.Path(os.path.join(path2save_model, "model_paramter.pkl"))) 

linear = nn.Linear(2,1)
linear.load_state_dict(torch.load(pathlib.Path(os.path.join(path2save_model, "model_paramter.pkl")))) 

# 5. 清空 cuda 缓存 该方法在 cuda 超内存时十分有用
torch.cuda.empty_cache()


# --------------------------------------------------------------
# 矩阵乘法范例 分别使用 CPU 和 GPU 作一个矩阵乘法, 并比较其计算效率
# --------------------------------------------------------------
print(f"\033[1;33;40m The matrix multiplication on GPU and CPU in PyTorch Successfully. \033[0m \n")

a = torch.rand((10000,200))
b = torch.rand((200,10000))
tic = time.time()
c = torch.matmul(a,b)
toc = time.time()

print("The time cost on CPU : ", end="")
print(toc - tic)
print(a.device)
print(b.device)

# 使用 gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
a = torch.rand((10000, 200),device = device) #可以指定在GPU上创建张量
b = torch.rand((200, 10000)) #也可以在CPU上创建张量后移动到GPU上
b = b.to(device) #或者 b = b.cuda() if torch.cuda.is_available() else b 
tic = time.time()
c = torch.matmul(a,b)
toc = time.time()

print("The time cost on GPU : ", end="")
print(toc-tic)
print(a.device)
print(b.device)


# --------------------------------------------------------------
# 线性回归 对比使用 CPU 和 GPU 训练一个线性回归模型的效率
# --------------------------------------------------------------
print(f"\033[1;33;40m The linear regression Model on GPU and CPU in PyTorch Successfully. \033[0m \n")

# 准备数据
n = 1000000 #样本数量
X = 10*torch.rand([n,2])-5.0  #torch.rand是均匀分布 
w0 = torch.tensor([[2.0,-3.0]])
b0 = torch.tensor([[10.0]])
Y = X@w0.t() + b0 + torch.normal( 0.0,2.0,size = [n,1])  # @表示矩阵乘法,增加正态扰动


# 定义模型
class LinearRegression(nn.Module): 
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn_like(w0))
        self.b = nn.Parameter(torch.zeros_like(b0))

    #正向传播
    def forward(self,x): 
        return x@self.w.t() + self.b

linear = LinearRegression()

# 训练模型
optimizer = torch.optim.Adam(linear.parameters(),lr = 0.1)
loss_func = nn.MSELoss()

def train(epochs):
    tic = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        Y_pred = linear(X) 
        loss = loss_func(Y_pred,Y)
        loss.backward() 
        optimizer.step()
        if epoch % 100 == 0:
            print({"epoch":epoch,"loss":loss.item()})
    toc = time.time()
    print("time used:",toc-tic)

# train(500)

# -------------GPU---------------
# 移动到 GPU
print("torch.cuda.is_available() = ", torch.cuda.is_available())
X = X.cuda()
Y = Y.cuda()
print("X.device:", X.device)
print("Y.device:", Y.device)

# 移动模型到 GPU 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
linear.to(device)
#查看模型是否已经移动到GPU上
print("if on cuda:",next(linear.parameters()).is_cuda)

train(500)


# --------------------------------
# single GPU to train model
# --------------------------------
# pip install -U torchkeras 
# --------------------------------
print(f"\033[1;33;40m The MNIST Model on single GPU in PyTorch Successfully. \033[0m \n")

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


class CnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size = 3),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            nn.Dropout2d(p = 0.1),
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,10)]
        )

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

net = CnnModel()
model = torchkeras.Model(net)
model.summary(input_shape=(1,32,32))


def accuracy(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
    return accuracy_score(y_true.cpu().numpy(),y_pred_cls.cpu().numpy()) 
    # 注意此处要将数据先移动到cpu上，然后才能转换成numpy数组

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.compile(loss_func = nn.CrossEntropyLoss(),
             optimizer= torch.optim.Adam(model.parameters(),lr = 0.02),
             metrics_dict={"accuracy":accuracy},device = device) # 注意此处compile时指定了device

df_history = model.fit(1, dl_train=data_load_train, dl_val=data_load_valid, log_step_freq=100) 
print(df_history)

# save the model parameters
torch.save(model.state_dict(), "./models/model_parameter_MNIST.pkl")

model_clone = torchkeras.Model(CnnModel())
model_clone.load_state_dict(torch.load("./models/model_parameter_MNIST.pkl"))

model_clone.compile(loss_func = nn.CrossEntropyLoss(),
             optimizer= torch.optim.Adam(model.parameters(),lr = 0.02),
             metrics_dict={"accuracy":accuracy},device = device) # 注意此处compile时指定了device

print(model_clone.evaluate(data_load_valid))

# ---------------------------------------
# nvidia-smi -l 1 // 实时查看 GPU 情况
# ---------------------------------------

# --------------------------------
# multi GPUs to train model
# --------------------------------
# pip install -U torchkeras 
# --------------------------------
print(f"\033[1;33;40m The MNIST Model on multi GPUs in PyTorch Successfully. \033[0m \n")

net_parallel = nn.DataParallel(CnnModel())  # Attention this line!!!
model_parallel = torchkeras.Model(net_parallel)
model_parallel.summary(input_shape=(1,32,32))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_parallel.compile(loss_func = nn.CrossEntropyLoss(),
             optimizer= torch.optim.Adam(model.parameters(),lr = 0.02),
             metrics_dict={"accuracy":accuracy},device = device) # 注意此处compile时指定了device

df_history = model_parallel.fit(1, dl_train=data_load_train, dl_val=data_load_valid, log_step_freq=100) 
print(df_history)

# save the model parameters
torch.save(model_parallel.net.module.state_dict(), "./models/multi_gpu_parameter.pkl")

net_clone = CnnModel()
net_clone.load_state_dict(torch.load("./models/multi_gpu_parameter.pkl"))

model_clone = torchkeras.Model(net_clone)
model_clone.compile(loss_func = nn.CrossEntropyLoss(),
             optimizer= torch.optim.Adam(model.parameters(),lr = 0.02),
             metrics_dict={"accuracy":accuracy},device = device)

print(model_clone.evaluate(data_load_valid))