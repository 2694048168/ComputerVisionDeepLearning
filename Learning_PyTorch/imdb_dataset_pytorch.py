#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: PyTorch 的建模流程 文本数据建模 IMDB dataset
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

# 文本数据建模流程 IMDB Dataset
# ----------------------------------
import os, pathlib
import datetime
import string, re
import torch
import torchtext
import torchkeras
import matplotlib.pyplot as plt
# ----------------------------------


def printbar():
    nowtime = datetime.datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")
    print(f"\n ========================================= {nowtime}")

# mac 系统上 pytorch 和 matplotlib 在 jupyter 中同时跑需要更改环境变量
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

# -------------------------
# tep 1. text data preprocessing
# ------------------------------------------------------------------------------------
# imdb 数据集的目标是根据电影评论的文本内容预测评论的情感标签
# 训练集有 20000 条电影评论文本, 测试集有 5000 条电影评论文本, 其中正面评论和负面评论都各占一半
# 文本数据预处理较为繁琐, 包括中文切词,构建词典,编码转换,序列填充,构建数据管道等等

# https://zhuanlan.zhihu.com/p/143845017 [torchtext]

# torch 中预处理文本数据一般使用 torchtext 或者自定义 Dataset
# torchtext 功能非常强大, 可以构建文本分类, 序列标注, 问答模型, 机器翻译等 NLP 任务的数据集

# torchtext 常见 API:
# 1. torchtext.data.Example : 用来表示一个样本, 数据和标签
# 2. torchtext.vocab.Vocab : 词汇表, 可以导入一些预训练词向量
# 3. torchtext.data.Datasets : 数据集类, __getitem__ 返回 Example 实例, torchtext.data.TabularDataset 是其子类
# 4. torchtext.data.Field : 用来定义字段的处理方法(文本字段, 标签字段)创建 Example 时的预处理, batch 时的一些处理操作
# 5. torchtext.data.Iterator : 迭代器, 用来生成 batch
# 6. torchtext.datasets : 包含了常见的数据集
# ------------------------------------------------------------------------------------
MAX_WORDS = 10000  # 仅考虑最高频的 10000 个词
MAX_LEN = 200  # 每个样本保留 200 个词的长度
BATCH_SIZE = 20 

# 分词方法
tokenizer = lambda x: re.sub('[%s]' % string.punctuation, "", x).split(" ")

# 过滤掉低频词
def filterLowFreqWords(arr, vocab):
    arr = [[x if x < MAX_WORDS else 0 for x in example] 
           for example in arr]
    return arr

# 1. 定义各个字段的预处理方法
TEXT = torchtext.legacy.data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=MAX_LEN, postprocessing=filterLowFreqWords)

LABEL = torchtext.legacy.data.Field(sequential=False, use_vocab=False)

# 2. 构建表格型 dataset
# torchtext.data.TabularDataset 可读取 csv, tsv, json 等格式
ds_train, ds_test = torchtext.legacy.data.TabularDataset.splits(
        path='./imdb', train='train.tsv', test='test.tsv', format='tsv',
        fields=[('label', LABEL), ('text', TEXT)], skip_header=False)

# 3. 构建词典
TEXT.build_vocab(ds_train)

# 4. 构建数据管道迭代器
train_iter, test_iter = torchtext.legacy.data.Iterator.splits((ds_train, ds_test), 
            sort_within_batch=True, 
            sort_key=lambda x: len(x.text), 
            batch_sizes=(BATCH_SIZE,BATCH_SIZE))

# 查看 example 信息
print(ds_train[0].text)
print(ds_train[0].label)

# 查看词典信息
print(len(TEXT.vocab))

# itos: index to string
print(TEXT.vocab.itos[0]) 
print(TEXT.vocab.itos[1]) 

# stoi: string to index
print(TEXT.vocab.stoi['<unk>']) #unknown 未知词
print(TEXT.vocab.stoi['<pad>']) #padding  填充

# freqs: 词频
print(TEXT.vocab.freqs['<unk>']) 
print(TEXT.vocab.freqs['a']) 
print(TEXT.vocab.freqs['good']) 

# 查看数据管道信息
# 注意有坑：text 第 0 维是句子长度
for batch in train_iter:
    features = batch.text
    labels = batch.label
    print(features)
    print(features.shape)
    print(labels)
    break


# 将数据管道组织成 torch.utils.data.DataLoader 相似的 features, label 输出形式
class DataLoader:
    def __init__(self,data_iter):
        self.data_iter = data_iter
        self.length = len(data_iter)

    def __len__(self):
        return self.length

    def __iter__(self):
        # 注意: 此处调整 features 为 batch first, 并调整 label 的 shape 和 dtype
        for batch in self.data_iter:
            yield(torch.transpose(batch.text, 0, 1),
                  torch.unsqueeze(batch.label.float(), dim=1))

dl_train = DataLoader(train_iter)
dl_test = DataLoader(test_iter)

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

class Net(torchkeras.Model):
    def __init__(self, net=None):
        super(Net, self).__init__(net)

        # 设置 padding_idx 参数后将在训练过程中将填充的 token 始终赋值为 0 向量
        self.embedding = torch.nn.Embedding(num_embeddings=MAX_WORDS, embedding_dim=3, padding_idx=1)
        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv_1", torch.nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5))
        self.conv.add_module("pool_1", torch.nn.MaxPool1d(kernel_size=2))
        self.conv.add_module("relu_1", torch.nn.ReLU())
        self.conv.add_module("conv_2", torch.nn.Conv1d(in_channels=16, out_channels=128, kernel_size=2))
        self.conv.add_module("pool_2", torch.nn.MaxPool1d(kernel_size=2))
        self.conv.add_module("relu_2", torch.nn.ReLU())

        self.dense = torch.nn.Sequential()
        self.dense.add_module("flatten", torch.nn.Flatten())
        self.dense.add_module("linear", torch.nn.Linear(6144,1))
        self.dense.add_module("sigmoid", torch.nn.Sigmoid())

    def forward(self,x):
        x = self.embedding(x).transpose(1,2)
        x = self.conv(x)
        y = self.dense(x)

        return y

model = Net()
print(model)
model.summary(input_shape=(200, ), input_dtype=torch.LongTensor)

# ====================================
# Step 3. Training Model with PyTorch
# ====================================
# PyTorch 通常需要用户编写自定义训练循环, 训练循环的代码风格因人而异, 3 类典型的训练循环代码风格: 
# 1. 脚本形式训练循环 
# 2. 函数形式训练循环
# 3. 类形式训练循环 ***
# ====================================
# 类形式的训练循环, 仿照 Keras 定义了一个高阶的模型接口 Model
# 实现 fit,  validate, predict, summary 方法, 相当于用户自定义高阶 API
def accuracy(y_pred, y_true):
    y_pred = torch.where(y_pred > 0.5, torch.ones_like(y_pred, dtype=torch.float32),
                      torch.zeros_like(y_pred, dtype=torch.float32))
    acc = torch.mean(1 - torch.abs(y_true - y_pred))

    return acc

model.compile(loss_func=torch.nn.BCELoss(), 
            optimizer=torch.optim.Adagrad(model.parameters(), lr=0.01),
            metrics_dict={"accuracy": accuracy})

# 有时候模型训练过程中不收敛, 需要多试几次
df_history = model.fit(20, dl_train, dl_val=dl_test, log_step_freq=200)


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

print(model.evaluate(dl_test))


# ====================================
# Step 5. Using Model with PyTorch
# ====================================
print(model.predict(dl_test))


# ====================================
# Step 6. Saving Model with PyTorch
# ====================================
# Pytorch 有两种保存模型的方式, 都是通过调用 pickle 序列化方法实现的
# 1. 第一种方法只保存模型参数 ***
# 2. 第二种方法保存完整模型
# 推荐使用第一种, 第二种方法可能在切换设备和目录的时候出现各种问题
# ====================================
path2model = "./models/imdb"
os.makedirs(path2model, exist_ok=True)

print(model.state_dict().keys())
torch.save(model.state_dict(), str(pathlib.Path(os.path.join(path2model, "net_parameter.pkl"))))

net_clone = Net()
net_clone.load_state_dict(torch.load(str(pathlib.Path(os.path.join(path2model, "net_parameter.pkl")))))
net_clone.compile(loss_func=torch.nn.BCELoss(), 
            optimizer=torch.optim.Adagrad(net_clone.parameters(), lr=0.01),
            metrics_dict={"accuracy": accuracy})

print(net_clone.evaluate(dl_test))