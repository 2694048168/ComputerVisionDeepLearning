#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: PyTorch 的建模流程 结构化数据建模 Titanic dataset
@Python Version: 3.9.7
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-03-30
"""

""" PyTorch 的建模流程
使用 Pytorch 实现神经网络模型的一般流程包括：
1. 准备数据
2. 定义模型
3. 训练模型
4. 评估模型
5. 使用模型
6. 保存模型

对新手来说,其中最困难的部分实际上是准备数据过程,
在实践中通常会遇到的数据类型包括结构化数据, 图片数据, 文本数据, 时间序列数据
titanic 生存预测问题, cifar2 图片分类问题, imdb电影评论分类问题, 国内新冠疫情结束时间预测问题
演示应用 PyTorch 对这四类数据的建模方法
"""

# 结构化数据建模流程 Titanic Dataset
# ----------------------------------
import os, pathlib
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
import torchkeras
# ----------------------------------


def printbar():
    nowtime = datetime.datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")
    print(f"\n ========================================= {nowtime}")

# mac 系统上 pytorch 和 matplotlib 在 jupyter 中同时跑需要更改环境变量
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

# =============================
# Step 1. Processing Data
# =============================
# titanic 数据集的目标是根据乘客信息预测他们在 Titanic 号撞击冰山沉没后能否生存
# 结构化数据一般会使用 Pandas 中的 DataFrame 进行预处理
# =============================
path2csv_train = r"./titanic/train.csv"
path2csv_test = r"./titanic/test.csv"

df_train_raw = pd.read_csv(path2csv_train)
df_test_raw = pd.read_csv(path2csv_test)

print(f"train data shape = {df_train_raw.shape}")
print(f"test data shape = {df_test_raw.shape}")
print(f"train data information:\n {df_train_raw.head()}")

# 字段说明：
# ------------------------------------------------------------------------
# Survived: 0代表死亡，1代表存活【y标签】
# Pclass: 乘客所持票类，有三种值(1,2,3) 【转换成onehot编码】
# Name: 乘客姓名 【舍去】
# Sex: 乘客性别 【转换成bool特征】
# Age: 乘客年龄(有缺失) 【数值特征，添加“年龄是否缺失”作为辅助特征】
# SibSp: 乘客兄弟姐妹/配偶的个数(整数值) 【数值特征】
# Parch: 乘客父母/孩子的个数(整数值)【数值特征】
# Ticket: 票号(字符串)【舍去】
# Fare: 乘客所持票的价格(浮点数，0-500不等) 【数值特征】
# Cabin: 乘客所在船舱(有缺失) 【添加“所在船舱是否缺失”作为辅助特征】
# Embarked: 乘客登船港口:S、C、Q(有缺失)【转换成onehot编码，四维度 S,C,Q,nan】
# ------------------------------------------------------------------------

# 利用 Pandas 的数据可视化功能可以简单地进行探索性数据分析 EDA(Exploratory Data Analysis)
img2save_folder = r"./images"
os.makedirs(img2save_folder, exist_ok=True)

# Step 1 for label distribution
ax_label = df_train_raw["Survived"].value_counts().plot(kind="bar", figsize=(8, 6), fontsize=15, rot=0)
ax_label.set_ylabel("Counts", fontsize=15)
ax_label.set_xlabel("Survived", fontsize=15)
# plt.show()
plt.savefig(os.path.join(img2save_folder, "label_dist.png"), dpi=120)
plt.close()

# Step 2 for age distribution
ax_age = df_train_raw["Age"].plot(kind="hist", bins=20, color="purple", figsize=(8, 6), fontsize=15)
ax_age.set_ylabel("Frequency", fontsize=15)
ax_age.set_xlabel("Age", fontsize=15)
# plt.show()
plt.savefig(os.path.join(img2save_folder, "age_dist.png"), dpi=120)
plt.close()

# Step 3 for correlation of age and label
ax_corr_age_label = df_train_raw.query("Survived==0")["Age"].plot(kind="density", figsize=(8, 6), fontsize=15)
df_train_raw.query("Survived==1")["Age"].plot(kind="density", figsize=(8, 6), fontsize=15)
ax_corr_age_label.legend(["Survived==0", "Survived==1"], fontsize=12)
ax_corr_age_label.set_ylabel("Density", fontsize=15)
ax_corr_age_label.set_xlabel("Age", fontsize=15)
# plt.show()
# pathlib.Path 避免由于操作系统引起的路径格式问题
plt.savefig(str(pathlib.Path(os.path.join(img2save_folder, "corr_age_label.png"))), dpi=120)
plt.close()


# data processing for Titanic Dataset
def preprocessing(df_data):
    df_result= pd.DataFrame()

    # Pclass term
    df_Pclass = pd.get_dummies(df_data['Pclass'])
    df_Pclass.columns = ['Pclass_' +str(x) for x in df_Pclass.columns]
    df_result = pd.concat([df_result, df_Pclass], axis=1)

    # Sex term
    df_Sex = pd.get_dummies(df_data['Sex'])
    df_result = pd.concat([df_result, df_Sex], axis=1)

    # Age term
    df_result['Age'] = df_data['Age'].fillna(0)
    df_result['Age_null'] = pd.isna(df_data['Age']).astype('int32')

    # SibSp, Parch, Fare terms
    df_result['SibSp'] = df_data['SibSp']
    df_result['Parch'] = df_data['Parch']
    df_result['Fare'] = df_data['Fare']

    # Carbin term
    df_result['Cabin_null'] =  pd.isna(df_data['Cabin']).astype('int32')

    # Embarked term
    df_Embarked = pd.get_dummies(df_data['Embarked'], dummy_na=True)
    df_Embarked.columns = ['Embarked_' + str(x) for x in df_Embarked.columns]
    df_result = pd.concat([df_result, df_Embarked], axis=1)

    return df_result

x_train = preprocessing(df_train_raw).values
y_train = df_train_raw['Survived'].values

x_test = preprocessing(df_test_raw).values
y_test = df_test_raw['Survived'].values

print(f"x_train shape = {x_train.shape}")
print(f"y_train shape = {y_train.shape}")
print(f"x_test shape = {x_test.shape}")
print(f"y_test shape = {y_test.shape}")

# -------------------------------------------------------------
# 进一步使用 DataLoader 和 TensorDataset 封装成可以迭代的数据管道
# -------------------------------------------------------------
dataset_train = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(x_train).float(),
                                                                            torch.tensor(y_train).float()),
                                            shuffle=True,
                                            batch_size=8)

dataset_test = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(x_test).float(),
                                                                            torch.tensor(y_test).float()),
                                            shuffle=False,
                                            batch_size=8)

# test the data pipeline
for features, labels in dataset_train:
    print(features, labels)
    break


# ====================================
# Step 2. Define Model with PyTorch
# ====================================
# 使用 PyTorch 通常有三种方式构建模型:
# 1. torch.nn.Sequential 按层顺序构建模型
# 2. 继承 torch.nn.Module 基类构建自定义模型
# 3. 继承 torch.nn.Module 基类构建模型并辅助应用模型容器进行封装
# ====================================
def create_net():
    net = torch.nn.Sequential()
    net.add_module("linear_1", torch.nn.Linear(15, 20))
    net.add_module("relu_1", torch.nn.ReLU())
    net.add_module("linear_2", torch.nn.Linear(20, 15))
    net.add_module("relu_2", torch.nn.ReLU())
    net.add_module("linear_3", torch.nn.Linear(15, 1))
    net.add_module("sigmoid", torch.nn.Sigmoid())

    return net

net = create_net()
print(net)
torchkeras.summary(net, input_shape=(15, ))


# ====================================
# Step 3. Training Model with PyTorch
# ====================================
# PyTorch 通常需要用户编写自定义训练循环, 训练循环的代码风格因人而异, 3 类典型的训练循环代码风格: 
# 1. 脚本形式训练循环 ***
# 2. 函数形式训练循环
# 3. 类形式训练循环
# ====================================
loss_func = torch.nn.BCELoss()
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.01)
# metric_func = lambda y_pred, y_true: sklearn.metrics.accuracy_score(y_true.data.numpy(), y_pred.data.numpy() > 0.5)
metric_func = lambda y_pred, y_true: accuracy_score(y_true.data.numpy(), y_pred.data.numpy() > 0.5)
metric_name = "accuracy"

epochs = 10
log_step_frequency = 30

df_history = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_"+metric_name])
print("\033[1;31;47m =================== Start Training =================== \033[0m")
print("\033[1;33;40m =================== Start Training =================== \033[0m")
printbar()

for epoch in range(1, epochs + 1):
    # step 1. loop training
    net.train()
    loss_sum = 0.0
    metric_sum = 0.0
    step = 1

    for step, (features, labels) in enumerate(dataset_train, 1):
        # 梯度清零
        optimizer.zero_grad()

        # 前向传播计算损失函数
        labels = torch.unsqueeze(labels, dim=-1)
        predictions = net(features)
        loss = loss_func(predictions, labels)
        metric = metric_func(predictions, labels)

        # 反向传播计算梯度
        loss.backward()
        optimizer.step()

        # 输出 batch (step) 级别日志
        loss_sum += loss.item() 
        metric_sum += metric.item()
        if step % log_step_frequency == 0:
            print(f"[step = {step}] loss: {loss_sum/step:.4f} {metric_name}: {metric_sum/step:.4f}")

    # step 2. loop validding
    net.eval()
    val_loss_sum = 0.0
    val_metric_sum = 0.0
    val_step = 1

    for val_step, (features, labels) in enumerate(dataset_test, 1):
        # 关闭梯度计算
        with torch.no_grad():
            predictions = net(features)
            labels = torch.unsqueeze(labels, dim=-1)
            val_loss = loss_func(predictions, labels)
            val_metric = metric_func(predictions, labels)

        val_loss_sum += val_loss.item()
        val_metric_sum += val_metric.item()

    # log information
    info = (epoch, loss_sum/step, metric_sum/step, val_loss_sum/val_step, val_metric_sum/val_step)
    df_history.loc[epoch - 1] = info

    # 输出 epoch 级别日志
    print(("\nEpoch= %d, loss= %.4f, " + metric_name + "= %.4f, val_loss= %.4f, "+" val_" + metric_name+"= %.4f")%info)
    printbar()

printbar()
print("\033[1;31;47m =================== Finished Training =================== \033[0m")
print("\033[1;33;40m =================== Finished Training =================== \033[0m")


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
plot_metric(df_history, "accuracy")


# ====================================
# Step 5. Using Model with PyTorch
# ====================================
# 预测概率 probability
y_pred_probability = net(torch.tensor(x_test[0 : 10]).float()).data
print("The probability of Model for Test is {}".format(y_pred_probability))

# 预测类别 classes
y_pred_classes = torch.where(y_pred_probability > 0.5, torch.ones_like(y_pred_probability), torch.zeros_like(y_pred_probability))
print("The classes of Model for Test is {}".format(y_pred_classes))


# ====================================
# Step 6. Saving Model with PyTorch
# ====================================
# Pytorch 有两种保存模型的方式, 都是通过调用 pickle 序列化方法实现的
# 1. 第一种方法只保存模型参数
# 2. 第二种方法保存完整模型
# 推荐使用第一种, 第二种方法可能在切换设备和目录的时候出现各种问题
# ====================================

# 保存模式参数, 以字典的形式进行保存
path2model = "./models"
os.makedirs(path2model, exist_ok=True)

print(net.state_dict().keys())
torch.save(net.state_dict(), str(pathlib.Path(os.path.join(path2model, "net_parameter.pkl"))))

net_clone = create_net()
net_clone.load_state_dict(torch.load(str(pathlib.Path(os.path.join(path2model, "net_parameter.pkl")))))
pred_probability_1 = net_clone.forward(torch.tensor(x_test[0 : 10]).float()).data
print("The probability of Model for Test is {}".format(pred_probability_1))


# 保存完整模式
torch.save(net, str(pathlib.Path(os.path.join(path2model, "net_model.pkl"))))
net_loaded = torch.load(str(pathlib.Path(os.path.join(path2model, "net_model.pkl"))))
pred_probability_2 = net_loaded(torch.tensor(x_test[0:10]).float()).data
print("The probability of Model for Test is {}".format(pred_probability_2))