#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: PyTorch 的建模流程 新冠肺炎病毒 时间序列分析和建模
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

# 时间序列数据建模流程 Covid-19
# ----------------------------------
import os, pathlib
import datetime
import torch
import torchkeras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ----------------------------------


def printbar():
    nowtime = datetime.datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")
    print(f"\n ========================================= {nowtime}")

# mac 系统上 pytorch 和 matplotlib 在 jupyter 中同时跑需要更改环境变量
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

# ---------------------------------------
# tep 1. time series data preprocessing
# 数据集取自 tushare
# ---------------------------------------
path2cvs_file = r"./covid_19.csv"

df_data_raw = pd.read_csv(pathlib.Path(path2cvs_file), sep="\t")
df_data_raw.plot(x="date", y=["confirmed_num", "cured_num", "dead_num"])
plt.xticks(rotation=60)
# plt.show()
plt.close()

df_data = df_data_raw.set_index("date")
df_diff = df_data.diff(periods=1).dropna()
df_diff = df_diff.reset_index("date")

df_diff.plot(x="date", y=["confirmed_num", "cured_num", "dead_num"])
plt.xticks(rotation=60)
# plt.show()
plt.close()
df_diff = df_diff.drop("date", axis=1).astype("float32")

print(df_diff.head())

# ----------------------------------------------------------
# 通过继承 torch.utils.data.Dataset 实现自定义时间序列数据集
# torch.utils.data.Dataset 是一个抽象类,用户想要加载自定义的数据只需要继承这个类,并且覆写其中的两个方法即可
# 1. __len__ : 实现 len(dataset) 返回整个数据集的大小, 必须重写父类的虚函数
# 2. __getitem__ : 用来获取一些索引的数据, 使 dataset[i] 返回数据集中第 i 个样本, 必须重写父类的虚函数
# ----------------------------------------------------------
# 用某日前 8 天窗口数据作为输入预测该日数据
WINDOW_SIZE = 8

class Covid19Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(df_diff) - WINDOW_SIZE

    def __getitem__(self, i):
        x = df_diff.loc[i : i + WINDOW_SIZE - 1, :]
        feature = torch.tensor(x.values)
        y = df_diff.loc[i + WINDOW_SIZE, :]
        label = torch.tensor(y.values)

        return (feature, label)

ds_train = Covid19Dataset()

# 数据较小, 可以将全部训练数据放入到一个 batch 中, 提升性能
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=38)


# --------------------------------------------------------------
# Step 2. Define Model with PyTorch
# --------------------------------------------------------------
# 使用 PyTorch 通常有三种方式构建模型:
# 1. torch.nn.Sequential 按层顺序构建模型
# 2. 继承 torch.nn.Module 基类构建自定义模型
# 3. 继承 torch.nn.Module 基类构建模型并辅助应用模型容器进行封装
# --------------------------------------------------------------
# 由于使用类形式的训练循环, 将模型封装成 torchkeras.Model 类来获得类似 Keras 中高阶模型接口的功能
# Model 类实际上继承自 torch.nn.Module 类
torch.random.seed()

class Block(torch.nn.Module):
    def __init__(self):
        super(Block, self).__init__()

    def forward(self, x, x_input):
        x_out = torch.max((1 + x) * x_input[:, -1, :], torch.tensor(0.0))
        return x_out

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3 层 LSTM
        self.lstm = torch.nn.LSTM(input_size=3, hidden_size=3, num_layers=5, batch_first=True)
        self.linear = torch.nn.Linear(3, 3)
        self.block = Block()

    def forward(self, x_input):
        x = self.lstm(x_input)[0][:, -1, :]
        x = self.linear(x)
        y = self.block(x, x_input)
        return y

net = Net()
model = torchkeras.Model(net)
print(model)
model.summary(input_shape=(8, 3), input_dtype=torch.FloatTensor)


# ====================================
# Step 3. Training Model with PyTorch
# ====================================
# PyTorch 通常需要用户编写自定义训练循环, 训练循环的代码风格因人而异, 3 类典型的训练循环代码风格: 
# 1. 脚本形式训练循环 
# 2. 函数形式训练循环 
# 3. 类形式训练循环 ***
# ====================================
# 仿照 Keras 定义了一个高阶的模型接口 Model,
# 实现 fit, validate, predict, summary 方法, 相当于用户自定义高阶 API
# 注: 循环神经网络调试较为困难, 需要设置多个不同的学习率多次尝试, 以取得较好的效果
def mspe(y_pred, y_true):
    err_percent = (y_true - y_pred)**2 / (torch.max(y_true**2, torch.tensor(1e-7)))
    return torch.mean(err_percent)

model.compile(loss_func=mspe, optimizer=torch.optim.Adagrad(model.parameters(), lr=0.1))

df_history = model.fit(100, dl_train, log_step_freq=10)


# ====================================
# Step 4. Eval Model with PyTorch
# ====================================
# 评估模型一般要设置验证集或者测试集,
# 由于此例数据较少, 仅仅可视化损失函数在训练集上的迭代情况
print(df_history)

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.title('Training '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric])
    plt.show()
    # plt.savefig(str(pathlib.Path(os.path.join(img2save_folder, "metic_loss.png"))), dpi=120)
    plt.close()

plot_metric(df_history, "loss")


# ====================================
# Step 5. Using Model with PyTorch
# ====================================
# 此处使用模型预测疫情结束时间, 即新增确诊病例为 0 的时间
#使用 df_result 记录现有数据以及此后预测的疫情数据
df_result = df_diff[["confirmed_num", "cured_num", "dead_num"]].copy()
print(df_result.tail())
print(type(df_result))

# 预测此后 200 天的新增走势, 将其结果添加到 df_result 中
for i in range(200):
    arr_input = torch.unsqueeze(torch.from_numpy(df_result.values[-38:, :]), axis=0)
    arr_predict = model.forward(arr_input)

    df_predict = pd.DataFrame(torch.floor(arr_predict).data.numpy(), columns=df_result.columns)
    df_result = pd.concat([df_result, df_predict], ignore_index=True)
    # df_result = df_result.append(df_predict, ignore_index=True)


df_result.query("confirmed_num==0").head()
# 第 50 天开始新增确诊降为 0, 第 45 天对应 3 月 10 日, 也就是 5 天后, 即预计 3 月 15 日新增确诊降为 0
# 注: 该预测偏乐观


df_result.query("cured_num==0").head()
# 第 132 天开始新增治愈降为 0, 第 45 天对应 3 月 10 日, 也就是大概 3 个月后, 即 6 月 10 日左右全部治愈
# 注: 该预测偏悲观, 并且存在问题, 如果将每天新增治愈人数加起来, 将超过累计确诊人数

df_result.query("dead_num==0").head()
# 第 50 天开始新增确诊降为 0, 第 45 天对应 3 月 10 日, 也就是 5 天后, 即预计 3 月 15 日新增确诊降为 0
# 注: 该预测偏乐观


# ====================================
# Step 6. Saving Model with PyTorch
# ====================================
# Pytorch 有两种保存模型的方式, 都是通过调用 pickle 序列化方法实现的
# 1. 第一种方法只保存模型参数 ***
# 2. 第二种方法保存完整模型
# 推荐使用第一种, 第二种方法可能在切换设备和目录的时候出现各种问题
# ====================================
path2model = "./models/covid_19"
os.makedirs(path2model, exist_ok=True)

print(model.state_dict().keys())
print()
print(model.net.state_dict().keys())
print()
torch.save(model.net.state_dict(), str(pathlib.Path(os.path.join(path2model, "net_parameter.pkl"))))

net_clone = Net()
net_clone.load_state_dict(torch.load(str(pathlib.Path(os.path.join(path2model, "net_parameter.pkl")))))
model_clone = torchkeras.Model(net_clone)
model_clone.compile(loss_func=mspe)

# 评估模型
model_clone.evaluate(dl_train)