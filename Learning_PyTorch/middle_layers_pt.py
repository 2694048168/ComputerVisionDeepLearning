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

模型层 layers, 深度学习模型一般由各种模型层组合而成
torch.nn 中内置了非常丰富的各种模型层, 它们都属于 torch.nn.Module 的子类, 具备参数管理功能

1. torch.nn.Linear, torch.nn.Flatten, torch.nn.Dropout, torch.nn.BatchNorm2d
2. torch.nn.Conv2d, torch.nn.AvgPool2d, torch.nn.Conv1d, torch.nn.ConvTranspose2d
3. torch.nn.Embedding, torch.nn.GRU, torch.nn.LSTM
4. torch.nn.Transformer

如果这些内置模型层不能够满足需求, 也可以通过继承 nn.Module 基类构建自定义的模型层
实际上, pytorch 不区分模型和模型层, 都是通过继承 torch.nn.Module 进行构建
因此只要继承 torch.nn.Module 基类并实现 forward 方法即可自定义模型层
"""

"""
--------------------------
内置模型层
--------------------------
一些常用的内置模型层简单介绍如下, 基础层
torch.nn.Linear: 全连接层, 参数个数 = 输入层特征数 x 输出层特征数 (weight) + 输出层特征数(bias)
torch.nn.Flatten: 压平层, 用于将多维张量样本压成一维张量样本
torch.nn.BatchNorm1d: 一维批标准化层, 通过线性变换将输入批次缩放平移到稳定的均值和标准差,可以增强模型对输入不同分布的适应性,加快模型训练速度,有轻微正则化效果,一般在激活函数之前使用, 可以用 afine 参数设置该层是否含有可以训练的参数

torch.nn.BatchNorm2d: 二维批标准化层
torch.nn.BatchNorm3d: 三维批标准化层
torch.nn.Dropout: 一维随机丢弃层,一种正则化手段
torch.nn.Dropout2d: 二维随机丢弃层
torch.nn.Dropout3d: 三维随机丢弃层
torch.nn.Threshold: 限幅层, 当输入大于或小于阈值范围时,截断之
torch.nn.ConstantPad2d: 二维常数填充层, 对二维张量样本填充常数扩展长度
torch.nn.ReplicationPad1d: 一维复制填充层, 对一维张量样本通过复制边缘值填充扩展长度
torch.nn.ZeroPad2d: 二维零值填充层, 对二维张量样本在边缘填充 0 值
torch.nn.GroupNorm: 组归一化, 一种替代批归一化的方法, 将通道分成若干组进行归一, 不受 batch 大小限制, 据称性能和效果都优于BatchNorm
torch.nn.LayerNorm: 层归一化, 较少使用
torch.nn.InstanceNorm2d: 样本归一化, 较少使用

各种归一化技术参考如下知乎文章《FAIR何恺明等人提出组归一化: 替代批归一化, 不受批量大小限制》
https://zhuanlan.zhihu.com/p/34858971

https://arxiv.org/abs/1803.08494 <<Group Normalization>> by Yuxin Wu, Kaiming He

---------------------
卷积网络相关层
---------------------
torch.nn.Conv1d: 普通一维卷积, 常用于文本, 参数个数 = 输入通道数×卷积核尺寸(如3)×卷积核个数 + 卷积核尺寸(如3)
torch.nn.Conv2d: 普通二维卷积, 常用于图像, 参数个数 = 输入通道数×卷积核尺寸(如3乘3)×卷积核个数 + 卷积核尺寸(如3乘3)

----> 通过调整 dilation 参数大于 1, 可以变成空洞卷积, 增大卷积核感受野
----> 通过调整 groups 参数不为 1, 可以变成分组卷积, 分组卷积中不同分组使用相同的卷积核, 显著减少参数数量
----> 当 groups 参数等于通道数时, 相当于 tensorflow 中的二维深度卷积层 tf.keras.layers.DepthwiseConv2D
----> 利用分组卷积和 1 乘 1 卷积的组合操作, 可以构造相当于 Keras 中的二维深度可分离卷积层 tf.keras.layers.SeparableConv2D

torch.nn.Conv3d: 普通三维卷积, 常用于视频, 参数个数 = 输入通道数×卷积核尺寸(如3乘3乘3)×卷积核个数 + 卷积核尺寸(如3乘3乘3)
torch.nn.MaxPool1d: 一维最大池化
torch.nn.MaxPool2d: 二维最大池化, 一种下采样方式, 没有需要训练的参数
torch.nn.MaxPool3d: 三维最大池化

torch.nn.AdaptiveMaxPool2d: 二维自适应最大池化, 无论输入图像的尺寸如何变化, 输出的图像尺寸是固定的
该函数的实现原理, 大概是通过输入图像的尺寸和要得到的输出图像的尺寸来反向推算池化算子的 padding, stride 等参数

torch.nn.FractionalMaxPool2d: 二维分数最大池化, 普通最大池化通常输入尺寸是输出的整数倍, 而分数最大池化则可以不必是整数,
分数最大池化使用了一些随机采样策略, 有一定的正则效果, 可以用它来代替普通最大池化和 Dropout 层

torch.nn.AvgPool2d: 二维平均池化
torch.nn.AdaptiveAvgPool2d: 二维自适应平均池化, 无论输入的维度如何变化, 输出的维度是固定的
torch.nn.ConvTranspose2d: 二维卷积转置层, 俗称反卷积层, 并非卷积的逆操作,
但在卷积核相同的情况下, 当其输入尺寸是卷积操作输出尺寸的情况下, 卷积转置的输出尺寸恰好是卷积操作的输入尺寸, 在语义分割中可用于上采样

torch.nn.Upsample: 上采样层, 操作效果和池化相反, 可以通过 mode 参数控制上采样策略为"nearest"最邻近策略或"linear"线性插值策略

torch.nn.Unfold: 滑动窗口提取层, 其参数和卷积操作 torch.nn.Conv2d 相同
实际上, 卷积操作可以等价于 torch.nn.Unfold 和 torch.nn.Linear 以及 torch.nn.Fold 的一个组合
其中 torch.nn.Unfold 操作可以从输入中提取各个滑动窗口的数值矩阵, 并将其压平成一维, 
利用 torch.nn.Linear 将 torch.nn.Unfold 的输出和卷积核做乘法后, 
再使用 torch.nn.Fold 操作将结果转换成输出图片形状

torch.nn.Fold: 逆滑动窗口提取层

# -------------------------
循环网络相关层
# -------------------------
torch.nn.Embedding: 嵌入层, 一种比 One-hot 更加有效的对离散特征进行编码的方法
一般用于将输入中的单词映射为稠密向量, 嵌入层的参数需要学习

torch.nn.LSTM: 长短记忆循环网络层[支持多层], 最普遍使用的循环网络层
具有携带轨道, 遗忘门,更新门,输出门.
可以较为有效地缓解梯度消失问题, 从而能够适用长期依赖问题
设置 bidirectional=True 时可以得到双向LSTM
需要注意的时,默认的输入和输出形状是(seq, batch, feature), 
如果需要将 batch 维度放在第 0 维, 则要设置 batch_first 参数设置为 True

torch.nn.GRU: 门控循环网络层[支持多层],
LSTM 的低配版, 不具有携带轨道, 参数数量少于 LSTM, 训练速度更快

torch.nn.RNN: 简单循环网络层[支持多层], 容易存在梯度消失,不能够适用长期依赖问题,一般较少使用
torch.nn.LSTMCell: 长短记忆循环网络单元, 和 torch.nn.LSTM 在整个序列上迭代相比, 它仅在序列上迭代一步, 一般较少使用
torch.nn.GRUCell: 门控循环网络单元, 和 torch.nn.GRU 在整个序列上迭代相比, 它仅在序列上迭代一步, 一般较少使用
torch.nn.RNNCell: 简单循环网络单元, 和 torch.nn.RNN 在整个序列上迭代相比, 它仅在序列上迭代一步, 一般较少使用

# ------------------
Transformer 相关层
# ------------------
torch.nn.Transformer: Transformer 网络结构,
Transformer 网络结构是替代循环网络的一种结构, 解决了循环网络难以并行, 难以捕捉长期依赖的缺陷
它是目前 NLP 任务的主流模型的主要构成部分,
Transformer 网络结构由 TransformerEncoder 编码器和 TransformerDecoder 解码器组成,
编码器和解码器的核心是 MultiheadAttention 多头注意力层

torch.nn.TransformerEncoder: Transformer 编码器结构,
由多个 torch.nn.TransformerEncoderLayer 编码器层组成

torch.nn.TransformerDecoder: Transformer 解码器结构
由多个 torch.nn.TransformerDecoderLayer 解码器层组成

torch.nn.TransformerEncoderLayer: Transformer 的编码器层
torch.nn.TransformerDecoderLayer: Transformer 的解码器层
torch.nn.MultiheadAttention: 多头注意力层

Transformer 原理介绍可以参考如下知乎文章《详解Transformer(Attention Is All You Need)》
https://zhuanlan.zhihu.com/p/48508221

https://arxiv.org/abs/1706.03762 <<Attention Is All You Need>> by Ashish Vaswani, Noam Shazeer, Niki Parmar

"""

# --------------------------
# 自定义模型层
# --------------------------
# PyTorch 的内置模型层不能够满足需求, 可以通过继承 torch.nn.Module 基类构建自定义的模型层
# 实际上, pytorch 不区分模型和模型层, 都是通过继承 torch.nn.Module 进行构建
# 因此, 只要继承 torch.nn.Module 基类并实现forward方法即可自定义模型层
# 下面是 Pytorch 的 torch.nn.Linear 层的源码, 可以仿照它来自定义模型层
import math
import torch
from torch import nn
import torch.nn.functional as F


class Linear(nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

# -----------------------------
linear = nn.Linear(20, 30)
inputs = torch.randn(128, 20)
output = linear(inputs)
print(output.size())