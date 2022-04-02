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

PyTorch 通常使用 Dataset 和 DataLoader 这两个工具类来构建数据管道
Dataset 定义了数据集的内容, 它相当于一个类似列表的数据结构, 具有确定的长度, 能够用索引获取数据集中的元素
DataLoader 定义了按 batch 加载数据集的方法, 它是一个实现了 __iter__ 方法的可迭代对象, 每次迭代输出一个 batch 的数据
DataLoader 能够控制 batch 的大小, batch 中元素的采样方法, 以及将 batch 结果整理成模型所需输入形式的方法,并且能够使用多进程读取数据

在绝大部分情况下,用户只需实现 Dataset 的 __len__ 方法和 __getitem__ 方法
就可以轻松构建自己的数据集, 并用默认数据管道进行加载
"""

import torch
import torchvision
import torchkeras
from sklearn import datasets
from PIL import Image
import pandas as pd
import re, string
import os
from collections import OrderedDict


# ====================================================
# Step 1. torch.utils.data.Dataset and DataLoader 
# ====================================================
"""
---------------------------------------
1>. 获取一个 batch 数据的步骤:
考虑一下从一个数据集中获取一个 batch 的数据需要哪些步骤
假定数据集的特征和标签分别表示为张量 X 和 Y, 数据集可以表示为 (X, Y), 假定 batch 大小为 m

1. 首先要确定数据集的长度 n, 结果类似: n = 1000
2. 然后从 0 到 n-1 的范围中抽样出 m 个数 (batch大小) 
    假定 m = 4,拿到的结果是一个列表, 类似: indices = [1,4,8,9]
3. 接着从数据集中去取这m个数对应下标的元素,拿到的结果是一个元组列表,
    类似:samples=[(X[1],Y[1]),(X[4],Y[4]),(X[8],Y[8]),(X[9],Y[9])]
4. 最后将结果整理成两个张量作为输出,拿到的结果是两个张量, 类似 batch = (features, labels),
    其中 features=torch.stack([X[1],X[4],X[8],X[9]]),
         labels = torch.stack([Y[1],Y[4],Y[8],Y[9]])

---------------------------------------
2>. Dataset 和 DataLoader 的功能分工
1. 上述第 1 个步骤确定数据集的长度是由 Dataset 的 __len__ 方法实现的
2. 第 2 个步骤从 0 到 n-1 的范围中抽样出 m 个数的方法是由 DataLoader 的 sampler 和 batch_sampler 参数指定的
    sampler 参数指定单个元素抽样方法, 一般无需用户设置, 
    默认在 DataLoader 的参数 shuffle=True 时采用随机抽样, shuffle=False 时采用顺序抽样

    batch_sampler 参数将多个抽样的元素整理成一个列表, 一般无需用户设置,
    默认方法在 DataLoader 的参数 drop_last=True 时会丢弃数据集最后一个长度不能被 batch 大小整除的批次
    在 drop_last=False 时保留最后一个批次

3. 第 3 个步骤的核心逻辑根据下标取数据集中的元素 是由 Dataset 的 __getitem__ 方法实现的
4. 第 4 个步骤的逻辑由 DataLoader 的参数 collate_fn 指定, 一般情况下也无需用户设置

---------------------------------------
3>. Dataset 和 DataLoader 的主要接口
以下是 Dataset 和 DataLoader 的核心接口逻辑伪代码, 不完全和源码一致
"""
class Dataset(object):
    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class DataLoader(object):
    def __init__(self, dataset, batch_size, collate_fn, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.sampler =torch.utils.data.RandomSampler if shuffle else torch.utils.data.SequentialSampler
        self.batch_sampler = torch.utils.data.BatchSampler
        self.sample_iter = self.batch_sampler(
            self.sampler(range(len(dataset))),
            batch_size=batch_size, drop_last=drop_last)

    def __next__(self):
        indices = next(self.sample_iter)
        batch = self.collate_fn([self.dataset[i] for i in indices])
        return batch


# ====================================================
# Step 2. torch.utils.data.Dataset 创建数据集 
# ====================================================
# Dataset 创建数据集常用的方法有:
# 1. 使用 torch.utils.data.TensorDataset 根据 Tensor 创建数据集 (numpy的array,Pandas的DataFrame需要先转换成Tensor)
# 2. 使用 torchvision.datasets.ImageFolder 根据图片目录创建图片数据集
# 3. 继承 torch.utils.data.Dataset 创建自定义数据集

# torch.utils.data.random_split 将一个数据集分割成多份, 常用于分割训练集, 验证集和测试集
# 调用 Dataset 的加法重载运算符 (+) 将多个数据集合并成一个数据集
# ====================================================

# 1----> 根据 Tensor 创建数据集
iris = datasets.load_iris()
ds_iris = torch.utils.data.TensorDataset(torch.tensor(iris.data), torch.tensor(iris.target))

# 分割成训练集和预测集
n_train = int(len(ds_iris) *0.8)
n_valid = len(ds_iris) - n_train
ds_train,ds_valid = torch.utils.data.random_split(ds_iris, [n_train, n_valid])
print(type(ds_iris))
print(type(ds_train))

# 使用 DataLoader 加载数据集
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=8)
dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=8)

for features, labels in dl_train:
    print(features)
    print(labels)
    break

# 演示加法运算符（+）的合并作用
ds_data = ds_train + ds_valid
print('len(ds_train) = ', len(ds_train))
print('len(ds_valid) = ', len(ds_valid))
print('len(ds_train+ds_valid) = ', len(ds_data))
print(type(ds_data))


# 2----> 根据图片目录创建图片数据集
img = Image.open("./images/cat.jpeg")
print(img)
img.show()

# 随机数值翻转
torchvision.transforms.RandomVerticalFlip()(img).show()

# 随机旋转
torchvision.transforms.RandomRotation(45)(img).show()

# 定义图片增强操作
transform_train = torchvision.transforms.Compose([
   torchvision.transforms.RandomHorizontalFlip(),  # 随机水平翻转
   torchvision.transforms.RandomVerticalFlip(),    # 随机垂直翻转
   torchvision.transforms.RandomRotation(45),      # 随机在 45 度角度内旋转
   torchvision.transforms.ToTensor()               # 转换成张量
  ]
)

transform_valid = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# 根据图片目录创建数据集
ds_train = torchvision.datasets.ImageFolder("./cifar2/train/",
            transform=transform_train, target_transform=lambda t: torch.tensor([t]).float())
ds_valid = torchvision.datasets.ImageFolder("./cifar2/test/",
            transform=transform_train, target_transform=lambda t: torch.tensor([t]).float())

print(ds_train.class_to_idx)


# 使用 DataLoader 加载数据集
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=50, shuffle=True, num_workers=0)
dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=50, shuffle=True, num_workers=0)

for features, labels in dl_train:
    print(features.shape)
    print(labels.shape)
    break

for features, labels in dl_valid:
    print(features.shape)
    print(labels.shape)
    break


# 3----> 创建自定义数据集
# 通过继承 Dataset 类创建 imdb 文本分类任务的自定义数据集
# 大概思路如下: 首先对训练集文本分词构建词典, 然后将训练集文本和测试集文本数据转换成 token 单词编码
# 接着将转换成单词编码的训练集数据和测试集数据按样本分割成多个文件, 一个文件代表一个样本
# 最后可以根据文件名列表获取对应序号的样本内容, 从而构建 Dataset 数据集
MAX_WORDS = 10000  # 仅考虑最高频的10000个词
MAX_LEN = 200  # 每个样本保留200个词的长度
BATCH_SIZE = 20 

train_data_path = './imdb/train.tsv'
test_data_path = './imdb/test.tsv'
train_token_path = './imdb/train_token.tsv'
test_token_path =  './imdb/test_token.tsv'
train_samples_path = './imdb/train_samples/'
test_samples_path =  './imdb/test_samples/'

# 首先构建词典, 并保留最高频的 MAX_WORDS 个词
word_count_dict = {}
# 清洗文本
def clean_text(text):
    lowercase = text.lower().replace("\n"," ")
    stripped_html = re.sub('<br />', ' ',lowercase)
    cleaned_punctuation = re.sub('[%s]'%re.escape(string.punctuation),'',stripped_html)
    return cleaned_punctuation

with open(train_data_path,"r",encoding = 'utf-8') as f:
    for line in f:
        label,text = line.split("\t")
        cleaned_text = clean_text(text)
        for word in cleaned_text.split(" "):
            word_count_dict[word] = word_count_dict.get(word,0)+1 

df_word_dict = pd.DataFrame(pd.Series(word_count_dict,name = "count"))
df_word_dict = df_word_dict.sort_values(by = "count",ascending =False)

df_word_dict = df_word_dict[0:MAX_WORDS-2] #  
df_word_dict["word_id"] = range(2,MAX_WORDS) #编号0和1分别留给未知词<unkown>和填充<padding>

word_id_dict = df_word_dict["word_id"].to_dict()
print(df_word_dict.head(10))


# 然后利用构建好的词典, 将文本转换成 token 序号
# 填充文本
def pad(data_list,pad_length):
    padded_list = data_list.copy()
    if len(data_list)> pad_length:
         padded_list = data_list[-pad_length:]
    if len(data_list)< pad_length:
         padded_list = [1]*(pad_length-len(data_list))+data_list
    return padded_list

def text_to_token(text_file,token_file):
    with open(text_file,"r",encoding = 'utf-8') as fin,\
      open(token_file,"w",encoding = 'utf-8') as fout:
        for line in fin:
            label,text = line.split("\t")
            cleaned_text = clean_text(text)
            word_token_list = [word_id_dict.get(word, 0) for word in cleaned_text.split(" ")]
            pad_list = pad(word_token_list,MAX_LEN)
            out_line = label+"\t"+" ".join([str(x) for x in pad_list])
            fout.write(out_line+"\n")

text_to_token(train_data_path, train_token_path)
text_to_token(test_data_path, test_token_path)

# 接着将 token 文本按照样本分割, 每个文件存放一个样本的数据
if not os.path.exists(train_samples_path):
    os.mkdir(train_samples_path)

if not os.path.exists(test_samples_path):
    os.mkdir(test_samples_path)

def split_samples(token_path,samples_dir):
    with open(token_path,"r",encoding = 'utf-8') as fin:
        i = 0
        for line in fin:
            with open(samples_dir+"%d.txt"%i,"w",encoding = "utf-8") as fout:
                fout.write(line)
            i = i+1

split_samples(train_token_path, train_samples_path)
split_samples(test_token_path, test_samples_path)

print(os.listdir(train_samples_path)[0:100])

# 可以创建数据集 Dataset,  从文件名称列表中读取文件内容了
class imdbDataset(Dataset):
    def __init__(self,samples_dir):
        self.samples_dir = samples_dir
        self.samples_paths = os.listdir(samples_dir)

    def __len__(self):
        return len(self.samples_paths)

    def __getitem__(self,index):
        path = self.samples_dir + self.samples_paths[index]
        with open(path,"r",encoding = "utf-8") as f:
            line = f.readline()
            label,tokens = line.split("\t")
            label = torch.tensor([float(label)],dtype = torch.float)
            feature = torch.tensor([int(x) for x in tokens.split(" ")],dtype = torch.long)
            return  (feature,label)

ds_train = imdbDataset(train_samples_path)
ds_test = imdbDataset(test_samples_path)

print(len(ds_train))
print(len(ds_test))

dl_train = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=BATCH_SIZE, num_workers=0)

for features,labels in dl_train:
    print(features)
    print(labels)
    break


# 最后构建模型测试一下数据集管道是否可用
class Net(torchkeras.Model):
    def __init__(self):
        super(Net, self).__init__()
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
        self.dense.add_module("linear", torch.nn.Linear(6144, 1))
        self.dense.add_module("sigmoid", torch.nn.Sigmoid())

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.conv(x)
        y = self.dense(x)
        return y

model = Net()
print(model)
model.summary(input_shape=(200, ), input_dtype=torch.LongTensor)

# 编译模型
def accuracy(y_pred,y_true):
    y_pred = torch.where(y_pred > 0.5, torch.ones_like(y_pred, dtype=torch.float32),
                      torch.zeros_like(y_pred, dtype=torch.float32))
    acc = torch.mean(1 - torch.abs(y_true - y_pred))
    return acc

model.compile(loss_func = torch.nn.BCELoss(), optimizer=torch.optim.Adagrad(model.parameters(), lr=0.02),
             metrics_dict={"accuracy":accuracy})

# 训练模型
df_history = model.fit(2, dl_train, dl_val=dl_test, log_step_freq=200)
print(df_history)


# ====================================================
# Step 3. torch.utils.data.Dataset 加载数据集 
# ====================================================
# DataLoader 能够控制 batch_size 的大小, batch 中元素的采样方法
# 以及将 batch 结果整理成模型所需输入形式的方法, 并且能够使用多进程读取数据
# DataLoader 的函数签名如下: function signatures
"""
torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    sampler=None,
    batch_sampler=None,
    num_workers=0,
    collate_fn=None,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
    multiprocessing_context=None,
)
"""
# 一般情况下,仅仅会配置 dataset, batch_size, shuffle, num_workers, drop_last 这五个参数,其他参数使用默认值即可
# DataLoader 除了可以加载前面讲的 torch.utils.data.Dataset 外,还能够加载另外一种数据集 torch.utils.data.IterableDataset
# 和 Dataset 数据集相当于一种列表结构不同, IterableDataset 相当于一种迭代器结构, 它更加复杂,一般较少使用
# ----------参数说明-------------
# 1. dataset : 数据集
# 2. batch_size : 批次大小
# 3. shuffle : 是否乱序
# 4. sampler : 样本采样函数,一般无需设置
# 5. batch_sampler : 批次采样函数,一般无需设置
# 6. num_workers : 使用多进程读取数据,设置的进程数
# 7. collate_fn : 整理一个批次数据的函数
# 8. pin_memory : 是否设置为锁业内存, 默认为 False, 锁业内存不会使用虚拟内存(硬盘), 从锁业内存拷贝到 GPU上速度会更快
# 9. drop_last : 是否丢弃最后一个样本数量不足 batch_size 批次数据
# 10. timeout : 加载一个数据批次的最长等待时间,一般无需设置
# 11. worker_init_fn : 每个 worker 中 dataset 的初始化函数, 常用于 IterableDataset, 一般不使用
# --------------------------------------------------------------------------------------------------------------
ds = torch.utils.data.TensorDataset(torch.arange(1, 50))
dl = torch.utils.data.DataLoader(ds,
                batch_size = 10,
                shuffle= True,
                num_workers=0,
                drop_last = True)
# 迭代数据
for batch, in dl:
    print(batch)