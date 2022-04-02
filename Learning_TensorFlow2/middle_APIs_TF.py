#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The middle-level API of TensorFlow2
@Python Version: 3.8.12
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-03-29
"""

""" 
TensorFlow 的中阶 API 主要包括:
1. 数据管道 tf.data
2. 特征列 tf.feature_column
3. 激活函数 tf.nn
4. 模型层 tf.keras.layers
5. 损失函数 tf.keras.losses
6. 评估函数 tf.keras.metrics
7. 优化器 tf.keras.optimizers
8. 回调函数 tf.keras.callbacks
"""

import json
import numpy as np
import tensorflow as tf


# =====================================================================================
""" 激活函数 activation
激活函数在深度学习中扮演着非常重要的角色, 它给网络赋予了非线性, 从而使得神经网络能够拟合任意复杂的函数
如果没有激活函数, 无论多复杂的网络, 都等价于单一的线性变换, 无法对非线性函数进行拟合
深度学习中最流行的激活函数为 relu,  但也有些新推出的激活函数, 例如 swish、GELU 据称效果优于 relu 激活函数.

https://zhuanlan.zhihu.com/p/98472075

https://zhuanlan.zhihu.com/p/98863801

常用激活函数
1. tf.nn.sigmoid: 将实数压缩到 0 到 1 之间, 一般只在二分类的最后输出层使用. 主要缺陷为存在梯度消失问题, 计算复杂度高, 输出不以 0 为中心

2. tf.nn.softmax: sigmoid 的多分类扩展, 一般只在多分类问题的最后输出层使用

3. tf.nn.tanh: 将实数压缩到 -1 到 1 之间, 输出期望为 0. 主要缺陷为存在梯度消失问题, 计算复杂度高

4. tf.nn.relu: 修正线性单元, 最流行的激活函数, 一般隐藏层使用. 主要缺陷是: 输出不以 0 为中心, 输入小于 0 时存在梯度消失问题(death relu)

5. tf.nn.leaky_relu: 对修正线性单元的改进, 解决了死亡 relu 问题

6. tf.nn.elu: 指数线性单元, 对 relu 的改进, 能够缓解死亡 relu 问题

7. tf.nn.selu: 扩展型指数线性单元, 在权重用 tf.keras.initializers.lecun_normal 初始化前提下能够对神经网络进行自归一化, 不可能出现梯度爆炸或者梯度消失问题, 需要和 Dropout 的变种 AlphaDropout 一起使用

8. tf.nn.swish: 自门控激活函数, 谷歌出品, 相关研究指出用 swish 替代 relu 将获得轻微效果提升

9. gelu: 高斯误差线性单元激活函数, 在 Transformer 中表现最好, tf.nn 模块尚没有实现该函数

在模型中使用激活函数
在 keras 模型中使用激活函数一般有两种方式,
1. 一种是作为某些层的 activation 参数指定
2. 另一种是显式添加 tf.keras.layers.Activation 激活层
"""
tf.keras.backend.clear_session()
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(32, input_shape=(None, 16), activation=tf.nn.relu)) # 通过activation参数指定
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation(tf.nn.softmax))  # 显式添加 layers.Activation 激活层
model.summary()


# =====================================================================================
""" 模型层 layers
深度学习模型一般由各种模型层组合而成, tf.keras.layers 内置了非常丰富的各种功能的模型层:
1. tf.keras.layers.Dense
2. tf.keras.layers.Flatten
3. tf.keras.layers.Input
4. tf.keras.layers.DenseFeature
5. tf.keras.layers.Dropout
6. tf.keras.layers.Conv2D
7. tf.keras.layers.MaxPooling2D
8. tf.keras.layers.Conv1D
9. tf.keras.layers.Embedding
10. tf.keras.layers.GRU
11. tf.keras.layers.LSTM
12. tf.keras.layers.Bidirectional

如果这些内置模型层不能够满足需求, 也可以通过编写 tf.keras.Lambda 匿名模型层
或继承 tf.keras.layers.Layer 基类构建自定义的模型层
其中 tf.keras.Lambda 匿名模型层只适用于构造没有学习参数的模型层 例如 attention compute in Transformer

内置模型层, 一些常用的内置模型层
基础层
1. Dense: 密集连接层, 参数个数 = 输入层特征数 x 输出层特征数 (weight) + 输出层特征数 (bias)
2. Activation: 激活函数层, 一般放在 Dense 层后面, 等价于在 Dense 层中指定 activation
3. Dropout: 随机置零层, 训练期间以一定几率将输入置 0, 一种正则化手段
4. BatchNormalization: 批标准化层, 通过线性变换将输入批次缩放平移到稳定的均值和标准差, 可以增强模型对输入不同分布的适应性, 加快模型训练速度, 有轻微正则化效果, 一般在激活函数之前使用
5. SpatialDropout2D: 空间随机置零层, 训练期间以一定几率将整个特征图置 0, 一种正则化手段, 有利于避免特征图之间过高的相关性
6. Input: 输入层, 通常使用 Functional API 方式构建模型时作为第一层
7. DenseFeature: 特征列接入层, 用于接收一个特征列列表并产生一个密集连接层
8. Flatten: 压平层, 用于将多维张量压成一维
9. Reshape: 形状重塑层, 改变输入张量的形状
10. Concatenate: 拼接层, 将多个张量在某个维度上拼接
11. Add: 加法层
12. Subtract: 减法层
13. Maximum: 取最大值层
14. Minimum: 取最小值层

卷积网络相关层
1. Conv1D: 普通一维卷积, 常用于文本, 参数个数 = 输入通道数 x 卷积核尺寸(3) x 卷积核个数
2. Conv2D: 普通二维卷积, 常用于图像, 参数个数 = 输入通道数 x 卷积核尺寸(3x3) x 卷积核个数
3. Conv3D: 普通三维卷积, 常用于视频, 参数个数 = 输入通道数 x 卷积核尺寸(3x3x3) x 卷积核个数
4. SeparableConv2D: 二维深度可分离卷积层, 不同于普通卷积同时对区域和通道操作, 深度可分离卷积先操作区域, 再操作通道. 即先对每个通道做独立卷积操作区域, 再用 1x1 卷积跨通道组合操作通道. 参数个数 = 输入通道数 x 卷积核尺寸 + 输入通道数 x 输出通道数. 深度可分离卷积的参数数量一般远小于普通卷积, 效果一般也更好
5. DepthwiseConv2D: 二维深度卷积层, 仅有 SeparableConv2D 前半部分操作, 即只操作区域, 不操作通道, 一般输出通道数和输入通道数相同, 但也可以通过设置 depth_multiplier 让输出通道为输入通道的若干倍数. 输出通道数 = 输入通道数 x depth_multiplier. 参数个数 = 输入通道数 x 卷积核尺寸 x depth_multiplier
6. Conv2DTranspose: 二维卷积转置层, 俗称反卷积层. 并非卷积的逆操作, 但在卷积核相同的情况下, 当其输入尺寸是卷积操作输出尺寸的情况下, 卷积转置的输出尺寸恰好是卷积操作的输入尺寸
7. LocallyConnected2D: 二维局部连接层, 类似 Conv2D, 唯一的差别是没有空间上的权值共享, 所以其参数个数远高于二维卷积
8. MaxPool2D: 二维最大池化层, 也称作下采样层, 池化层无可训练参数, 主要作用是降尺度
9. AveragePooling2D: 二维平均池化层
10. GlobalMaxPool2D: 全局最大池化层, 每个通道仅保留一个值, 一般从卷积层过渡到全连接层时使用, 是 Flatten 的替代方案
11. GlobalAvgPool2D: 全局平均池化层, 每个通道仅保留一个值

循环网络相关层
1. Embedding: 嵌入层, 一种比 One-hot 更加有效的对离散特征进行编码的方法, 一般用于将输入中的单词映射为稠密向量, 嵌入层的参数需要学习
2. LSTM: 长短记忆循环网络层, 最普遍使用的循环网络层, 具有携带轨道, 遗忘门, 更新门, 输出门. 可以较为有效地缓解梯度消失问题, 从而能够适用长期依赖问题, 设置 return_sequences=True 时可以返回各个中间步骤输出, 否则只返回最终输出
3. GRU: 门控循环网络层, LSTM 的低配版, 不具有携带轨道, 参数数量少于 LSTM, 训练速度更快
4. SimpleRNN: 简单循环网络层, 容易存在梯度消失, 不能够适用长期依赖问题, 一般较少使用
5. ConvLSTM2D: 卷积长短记忆循环网络层, 结构上类似 LSTM, 但对输入的转换操作和对状态的转换操作都是卷积运算
6. Bidirectional: 双向循环网络包装器, 可以将 LSTM, GRU 等层包装成双向循环网络, 从而增强特征提取能力
7. RNN: RNN 基本层, 接受一个循环网络单元或一个循环单元列表, 通过调用 tf.keras.backend.rnn 函数在序列上进行迭代从而转换成循环网络层
8. LSTMCell: LSTM 单元, 和 LSTM 在整个序列上迭代相比, 它仅在序列上迭代一步, 可以简单理解 LSTM 即 RNN 基本层包裹 LSTMCell
9. GRUCell: GRU 单元, 和 GRU 在整个序列上迭代相比, 它仅在序列上迭代一步
10. SimpleRNNCell: SimpleRNN 单元, 和 SimpleRNN 在整个序列上迭代相比, 它仅在序列上迭代一步
11. AbstractRNNCell: 抽象 RNN 单元, 通过对它的子类化用户可以自定义 RNN 单元, 再通过 RNN 基本层的包裹实现用户自定义循环网络层
12. Attention: Dot-product 类型注意力机制层, 可以用于构建注意力模型
13. AdditiveAttention: Additive 类型注意力机制层, 可以用于构建注意力模型
14. TimeDistributed: 时间分布包装器, 包装后可以将 Dense、Conv2D 等作用到每一个时间片段上

自定义模型层
如果自定义模型层没有需要被训练的参数, 一般推荐使用 Lamda 层实现
如果自定义模型层有需要被训练的参数, 则可以通过对 Layer 基类子类化实现
Lambda 层由于没有需要被训练的参数, 只需要定义正向传播逻辑即可, 使用比 Layer 基类子类化更加简单
Lambda 层的正向逻辑可以使用 Python 的 lambda 函数来表达, 也可以用 def 关键字定义函数来表达
Layer 的子类化一般需要重新实现初始化方法, Build 方法和 Call 方法, 简化的线性层范例类似Dense
"""
my_power = tf.keras.layers.Lambda(lambda x: tf.math.pow(x, 2))
result = my_power(tf.range(5))
print(result)

# https://zhuanlan.zhihu.com/p/149532177 (*args and **kwargs in Python)
class Linear(tf.keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units

    # build method 一般定义 Layer 所需要被训练的参数
    def build(self, input_shape):
        self.w = self.add_weight("w", shape=(input_shape[-1], self.units),
                                 initializer="random_normal",
                                 trainable=True) # 注意必须要有参数名称 "w", 否则会报错
        
        self.b = self.add_weight("b", shape=(self.units, ),
                                 initializer="random_normal",
                                 trainable=True)

        super(Linear, self).build(input_shape) # 相当于设置 self.built = True

    # call method 一般定义正向传播逻辑, __call__ method 调用它
    @tf.function
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    # 如果要让自定义的 Layer 通过 Functional API 组合成模型时可以被保存为 h5 模型
    # 需要自定义 get_config method
    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({"units": self.units})
        return config

linear = Linear(units=8)
print(linear.built)
# 指定 input_shape, 显式调用 build 方法, 第 0 维代表样本数量, 用 None 填充
linear.build(input_shape = (None, 16)) 
print(linear.built)
print(linear.compute_output_shape(input_shape=(None, 16)))

linear_2 = Linear(units=16)
print(linear_2.built)
# 如果 built=False, 调用 __call__ 时会先调用 build 方法, 再调用 call 方法
linear_2(tf.random.uniform((100, 64))) 
print(linear_2.built)
config = linear_2.get_config()
print(config)


tf.keras.backend.clear_session()
model = tf.keras.models.Sequential()
# 注意该处的 input_shape 会被模型加工, 无需使用 None 代表样本数量维
model.add(Linear(units = 1, input_shape=(2, )))  
print("model.input_shape: ", model.input_shape)
print("model.output_shape: ", model.output_shape)
model.summary()


model.compile(optimizer="sgd", loss="mse", metrics=["mae"])
print(model.predict(tf.constant([[3.0, 2.0], [4.0, 5.0]])))
# 保存成 h5 模型 "./Models"
model.save("./Models/linear_model.h5", save_format="h5")
model_loaded_keras = tf.keras.models.load_model("./Models/linear_model.h5", custom_objects={"Linear": Linear})
tf.print("\033[1;33;40m The Model saving with Keras format in ./Models/linear_model successfully. \033[0m")
print(model_loaded_keras.predict(tf.constant([[3.0, 2.0], [4.0, 5.0]])))

# 保存成 tf 模型
model.save("./Models/linear_model", save_format="tf")
model_loaded_tf = tf.keras.models.load_model("./Models/linear_model")
tf.print("\033[1;33;40m The Model saving with TensorFlow format in ./Models/linear_model successfully. \033[0m")
print(model_loaded_tf.predict(tf.constant([[3.0, 2.0], [4.0, 5.0]])))


# =====================================================================================
""" 损失函数 losses
一般来说, 监督学习的目标函数由损失函数和正则化项组成 Objective = Loss + Regularization
对于 keras 模型, 目标函数中的正则化项一般在各层中指定, 例如使用 Dense 的 kernel_regularizer 和 bias_regularizer 等参数指定权重使用L1 或者 L2 正则化项, 此外还可以用 kernel_constraint 和 bias_constraint 等参数约束权重的取值范围, 这也是一种正则化手段

损失函数在模型编译时候指定, 对于回归模型, 通常使用的损失函数是均方损失函数 mean_squared_error(MSE)
对于二分类模型, 通常使用的是二元交叉熵损失函数 binary_crossentropy
对于多分类模型, 如果 label 是 one-hot 编码的, 则使用类别交叉熵损失函数 categorical_crossentropy;
如果 label 是类别序号编码的, 则需要使用稀疏类别交叉熵损失函数 sparse_categorical_crossentropy

如果有需要, 也可以自定义损失函数, 自定义损失函数需要接收两个张量 y_true, y_pred 作为输入参数, 并输出一个标量作为损失函数值

内置损失函数
内置的损失函数一般有类的实现和函数的实现两种形式
CategoricalCrossentropy 和 categorical_crossentropy 都是类别交叉熵损失函数, 前者是类的实现形式, 后者是函数的实现形式
常用的一些内置损失函数说明:
1. mean_squared_error, 均方误差损失, 用于回归, 简写为 mse, 类与函数实现形式分别为 MeanSquaredError 和 MSE
2. mean_absolute_error, 平均绝对值误差损失, 用于回归, 简写为 mae, 类与函数实现形式分别为 MeanAbsoluteError 和 MAE
3. mean_absolute_percentage_error, 平均百分比误差损失, 用于回归, 简写为 mape, 类与函数实现形式分别为 MeanAbsolutePercentageError 和 MAPE
4. Huber, Huber损失, 只有类实现形式, 用于回归, 介于 mse 和 mae 之间, 对异常值比较鲁棒, 相对 mse 有一定的优势
5. binary_crossentropy, 二元交叉熵, 用于二分类, 类实现形式为 BinaryCrossentropy
6. categorical_crossentropy, 类别交叉熵, 用于多分类, 要求 label 为 onehot 编码, 类实现形式为 CategoricalCrossentropy
7. sparse_categorical_crossentropy, 稀疏类别交叉熵, 用于多分类, 要求 label 为序号编码形式, 类实现形式为 SparseCategoricalCrossentropy
8. hinge, 合页损失函数, 用于二分类, 最著名的应用是作为支持向量机 SVM 的损失函数, 类实现形式为 Hinge
9. kld, 相对熵损失, 也叫KL散度, 常用于最大期望算法 EM 的损失函数, 两个概率分布差异的一种信息度量, 类与函数实现形式分别为 KLDivergence 或 KLD
10. cosine_similarity, 余弦相似度, 可用于多分类, 类实现形式为 CosineSimilarity


自定义损失函数
自定义损失函数接收两个张量 y_true, y_pred 作为输入参数, 并输出一个标量作为损失函数值
也可以对 tf.keras.losses.Loss 进行子类化, 重写 call 方法实现损失的计算逻辑, 从而得到损失函数的类的实现

Focal Loss的自定义实现, Focal Loss 是一种对 binary_crossentropy 的改进损失函数形式
它在样本不均衡和存在较多易分类的样本时相比 binary_crossentropy 具有明显的优势
它有两个可调参数, alpha 参数和 gamma 参数, 其中 alpha 参数主要用于衰减负样本的权重, gamma 参数主要用于衰减容易训练样本的权重
从而让模型更加聚焦在正样本和困难样本上, 这就是为什么这个损失函数叫做 Focal Loss

https://zhuanlan.zhihu.com/p/80594704
"""
# 损失函数和正则化项
tf.keras.backend.clear_session()
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, input_dim=64,
                                kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                activity_regularizer=tf.keras.regularizers.l1(0.01),
                                kernel_constraint=tf.keras.constraints.MaxNorm(max_value=2, axis=0)))
model.add(tf.keras.layers.Dense(10,
                                kernel_regularizer=tf.keras.regularizers.l1_l2(0.01, 0.01), activation="sigmoid"))

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["AUC"])                                
model.summary()


# Focal Loss implementation
# --------------Two Paper of Focal Loss and Improved-----------------------------------------------
# https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf
# https://arxiv.org/abs/1811.05181
# -------------------------------------------------------------------------------------------------
def focal_loss(gamma=2., alpha=0.75):
    def focal_loss_fixed(y_true, y_pred):
        binary_cross_entropy = tf.losses.binary_crossentropy(y_true, y_pred)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        loss = tf.reduce_sum(alpha_factor * modulating_factor * binary_cross_entropy, axis=-1)

        return loss
    return focal_loss_fixed

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.75, name="focal_loss"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        binary_cross_entropy = tf.losses.binary_crossentropy(y_true, y_pred)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        loss = tf.reduce_sum(alpha_factor * modulating_factor * binary_cross_entropy, axis=-1)

        return loss


# =====================================================================================
""" 评估指标 metrics
损失函数除了作为模型训练时候的优化目标, 也能够作为模型好坏的一种评价指标. 但通常人们还会从其它角度评估模型的好坏
这就是评估指标, 通常损失函数都可以作为评估指标, 如 MAE, MSE, CategoricalCrossentropy 等也是常用的评估指标
但评估指标不一定可以作为损失函数, 例如 AUC, Accuracy, Precision, 因为评估指标不要求连续可导, 而损失函数通常要求连续可导
编译模型时，可以通过列表形式指定多个评估指标, 如果有需要, 也可以自定义评估指标
自定义评估指标需要接收两个张量 y_true, y_pred 作为输入参数, 并输出一个标量作为评估值
也可以对 tf.keras.metrics.Metric 进行子类化, 重写初始化方法, update_state方法, result方法实现评估指标的计算逻辑, 从而得到评估指标的类的实现形式

由于训练的过程通常是分批次训练的, 评估指标要跑完一个 epoch 才能够得到整体的指标结果, 因此类形式的评估指标更为常见, 即需要编写初始化方法以创建与计算指标结果相关的一些中间变量, 编写 update_state方法在每个 batch 后更新相关中间变量的状态, 编写 result 方法输出最终指标结果

如果编写函数形式的评估指标, 则只能取 epoch 中各个 batch 计算的评估指标结果的平均值作为整个 epoch 上的评估指标结果, 这个结果通常会偏离整个 epoch 数据一次计算的结果

一. 常用的内置评估指标
1. MeanSquaredError, 均方误差, 用于回归, 可以简写为 MSE, 函数形式为 mse
2. MeanAbsoluteError, 平均绝对值误差, 用于回归, 可以简写为 MAE, 函数形式为 mae
3. MeanAbsolutePercentageError, 平均百分比误差, 用于回归, 可以简写为 MAPE, 函数形式为 mape
4. RootMeanSquaredError, 均方根误差, 用于回归
5. Accuracy, 准确率, 用于分类, 可以用字符串 "Accuracy" 表示, 要求 y_true 和 y_pred 都为类别序号编码
6. Precision, 精确率, 用于二分类, Precision = TP/(TP+FP), 查看混淆矩阵 confusion matrix
7. Recall, 召回率, 用于二分类, Recall = TP/(TP+FN)
8. TruePositives, 真正例, 用于二分类
9. TrueNegatives, 真负例, 用于二分类
10. FalsePositives, 假正例, 用于二分类
11. FalseNegatives, 假负例, 用于二分类
12. AUC, ROC曲线 (TPR vs FPR)下的面积, 用于二分类, 直观解释为随机抽取一个正样本和一个负样本, 正样本的预测值大于负样本的概率
13. CategoricalAccuracy, 分类准确率, 与 Accuracy 含义相同, 要求 y_true(label) 为 onehot 编码形式
14. SparseCategoricalAccuracy, 稀疏分类准确率, 与 Accuracy 含义相同, 要求 y_true(label) 为序号编码形式
15. MeanIoU, Intersection-Over-Union, 常用于图像分割
16. TopKCategoricalAccuracy, 多分类 TopK 准确率, 要求 y_true(label) 为 one-hot 编码形式
17. SparseTopKCategoricalAccuracy, 稀疏多分类 TopK 准确率, 要求 y_true(label) 为序号编码形式
18. Mean, 平均值
19. Sum, 求和

二. 自定义评估指标
以金融风控领域常用的 KS 指标, 示范自定义评估指标
KS 指标适合二分类问题, 其计算方式为 KS = max(TPR - FPR), 其中 TPR=TP/(TP+FN), FPR = FP/(FP+TN)
TPR 曲线实际上就是正样本的累积分布曲线(CDF), FPR 曲线实际上就是负样本的累积分布曲线(CDF)
KS 指标就是正样本和负样本累积分布曲线差值的最大值
"""
# 函数形式的自定义评估指标
@tf.function
def ks(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1, ))
    y_pred = tf.reshape(y_pred, (-1, ))
    length = tf.shape(y_true)[0]
    t = tf.math.top_k(y_pred, k=length, sorted=False)
    y_pred_sorted = tf.gather(y_pred, t.indices)
    y_true_sorted = tf.gather(y_true, t.indices)
    cum_positive_ratio = tf.truediv(tf.cumsum(y_true_sorted), tf.reduce_sum(y_true_sorted))
    cum_negative_ratio = tf.truediv(tf.cumsum(1 - y_true_sorted), tf.reduce_sum(1 - y_true_sorted))
    ks_value = tf.reduce_max(tf.abs(cum_positive_ratio - cum_negative_ratio))

    return ks_value


# 类的形式实现自定义评估指标
class KS(tf.keras.metrics.Metric):
    def __init__(self, name="ks", **kwargs):
        super(KS, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(
            name="tp",
            shape=(101, ),
            initializer="zeros"
        )
        self.false_positives = self.add_weight(
            name="fp",
            shape=(101, ),
            initializer="zeros"
        )

    @tf.function
    def update_state(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, (-1, )), tf.bool)
        y_pred = tf.cast(100 * tf.reshape(y_pred, (-1, )), tf.int32)

        for i in tf.range(0, tf.shape(y_true)[0]):
            if y_true[i]:
                self.true_positives[y_pred[i]].assign(self.true_positives[y_pred[i]] + 1.0)
            else:
                self.false_positives[y_pred[i]].assign(self.false_positives[y_pred[i]] + 1.0)

        return (self.true_positives, self.false_positives)

    @tf.function
    def result(self):
        cum_positive_ratio = tf.truediv(tf.cumsum(self.true_positives), tf.reduce_sum(self.true_positives))
        cum_negative_ratio = tf.truediv(tf.cumsum(self.false_positives), tf.reduce_sum(self.false_positives))
        ks_value = tf.reduce_max(tf.abs(cum_positive_ratio - cum_negative_ratio))

        return ks_value


# ======== Test ========
y_true = tf.constant([[1], [1], [1], [0], [1], [1], [1], [0], [0], [0], [1], [0], [1], [0]])
y_pred = tf.constant([[0.6], [0.1], [0.4], [0.5], [0.7], [0.7], [0.7],
                      [0.4], [0.4], [0.5], [0.8], [0.3], [0.5], [0.3]])

tf.print(ks(y_true, y_pred))

ks_instance = KS()
ks_instance.update_state(y_true, y_pred)
tf.print(ks_instance.result())


# =====================================================================================
""" 优化器 optimizers
机器学习界有一群炼丹师, 每天的日常是:拿来药材(数据), 架起八卦炉(模型), 点着六味真火(优化算法), 就摇着蒲扇等着丹药出炉了.
机器学习一样,模型优化算法的选择直接关系到最终模型的性能,有时候效果不好,未必是特征的问题或者模型设计的问题,很可能就是优化算法的问题.
深度学习优化算法大概经历 SGD -> SGDM -> NAG ->Adagrad -> Adadelta(RMSprop) -> Adam -> Nadam

https://zhuanlan.zhihu.com/p/32230623

对于一般新手炼丹师, 优化器直接使用 Adam, 并使用其默认参数就OK
一些爱写论文的炼丹师由于追求评估指标效果, 可能会偏爱前期使用 Adam 优化器快速下降, 后期使用 SGD 并精调优化器参数得到更好的结果
此外目前也有一些前沿的优化算法, 据称效果比 Adam 更好, 例如 LazyAdam, Look-ahead, RAdam, Ranger.

一. 优化器的使用
优化器主要使用 apply_gradients method 方法传入变量和对应梯度从而来对给定变量进行迭代,
或者直接使用 minimize method 方法对目标函数进行迭代优化.
更常见的使用是在编译时将优化器传入 keras 的 Model, 通过调用 model.fit 实现对 Loss 的的迭代优化.
初始化优化器时会创建一个变量 optimier.iterations 用于记录迭代的次数, 因此优化器和 tf.Variable 一样, 需要在 @tf.function 外创建.

二. 内置优化器
tf.keras.optimizers 子模块中, 基本上都有对应的类的实现
1. SGD, 默认参数为纯 SGD, 设置 momentum 参数不为 0 实际上变成 SGDM, 考虑一阶动量, 设置 nesterov 为 True 后变成 NAG, 即 Nesterov Accelerated Gradient, 在计算梯度时计算的是向前走一步所在位置的梯度

2. Adagrad, 考虑二阶动量, 对于不同的参数有不同的学习率, 即自适应学习率. 缺点是学习率单调下降, 可能后期学习速率过慢乃至提前停止学习

3. RMSprop, 考虑二阶动量, 对于不同的参数有不同的学习率, 即自适应学习率, 对 Adagrad 进行了优化, 通过指数平滑只考虑一定窗口内的二阶动量

4. Adadelta, 考虑二阶动量, 与 RMSprop 类似, 但是更加复杂一些, 自适应性更强

5. Adam, 同时考虑一阶动量和二阶动量, 可以看成 RMSprop 上进一步考虑一阶动量

6. Nadam,  在 Adam 基础上进一步考虑 Nesterov Acceleration
"""
@tf.function
def printbar():
    today_time = tf.timestamp() % (24*60*60)
    hours = tf.cast(today_time // 3600 + 8, tf.int32) % tf.constant(24)
    minutes = tf.cast((today_time % 3600) // 60, tf.int32)
    seconds = tf.cast(tf.floor(today_time % 60), tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m)) == 1:
            return tf.strings.format("0{}", m)
        else:
            return tf.strings.format("{}", m)

    timestring = tf.strings.join([timeformat(hours), timeformat(minutes), timeformat(seconds)], separator=":")
    tf.print("================================= \033[1;33;40m It is now a Beijing Time : \033[0m", timestring)


# 计算 f(x) = a*x^2 + b*x + c 的最小值
# method 1. optimizer.apply_gradients
tf.print("\033[1;33;40m Method for optimizer.apply_gradients : \033[0m")
x = tf.Variable(0.0, name="x", dtype=tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

@tf.function
def minimizer_func():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)

    while tf.constant(True):
        with tf.GradientTape() as tape:
            y = a * tf.pow(x, 2) + b * x + c

        dy_dx = tape.gradient(y, x)
        optimizer.apply_gradients(grads_and_vars=[(dy_dx, x)])

        # 迭代终止条件
        epsilon = 0.00001
        if tf.abs(dy_dx) < tf.constant(epsilon):
            break

        if tf.math.mod(optimizer.iterations, 100)==0:
            printbar()
            tf.print("step = ", optimizer.iterations)
            tf.print("x = ", x)
            tf.print("")

    y = a * tf.pow(x, 2) + b * x + c

    return y

tf.print("y =", minimizer_func())
tf.print("x =", x)

# method 2. 使用 optimizer.minimize
tf.print("\033[1;33;40m Method for optimizer.minimize : \033[0m")
def func():   
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a * tf.pow(x, 2) + b * x + c
    return y

@tf.function
def train(epochs=1000):  
    for _ in tf.range(epochs):  
        optimizer.minimize(func, [x])
    tf.print("epoch = ", optimizer.iterations)
    return func()

train(1000)
tf.print("y = ", func())
tf.print("x = ", x)


# method 3. 使用 model.fit
tf.print("\033[1;33;40m Method for model.fit : \033[0m")
tf.keras.backend.clear_session()

class FakeModel(tf.keras.models.Model):
    def __init__(self, a, b, c):
        super(FakeModel, self).__init__()
        self.a = a
        self.b = b
        self.c = c

    def build(self):
        self.x = tf.Variable(0.0, name="x")
        self.built = True

    def call(self, features):
        loss  = self.a * (self.x)**2 + self.b * (self.x) + self.c
        return(tf.ones_like(features) * loss)

def my_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)


model = FakeModel(tf.constant(1.0), tf.constant(-2.0), tf.constant(1.0))
model.build()
model.summary()
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss=my_loss)
history = model.fit(tf.zeros((100,2)), tf.ones(100), batch_size=1, epochs=10)  # 迭代 1000次

tf.print("x = ", model.x)
tf.print("loss = ", model(tf.constant(0.0)))


# =====================================================================================
""" 回调函数 callbacks
tf.keras 的回调函数实际上是一个类, 一般是在 model.fit 时作为参数指定,
用于控制在训练过程开始或者在训练过程结束,
在每个 epoch 训练开始或者训练结束,
在每个 batch 训练开始或者训练结束时执行一些操作, 例如收集一些日志信息, 改变学习率等超参数, 提前终止训练过程等等

同样地, 针对 model.evaluate 或者 model.predict 也可以指定 callbacks 参数,
用于控制在评估或预测开始或者结束时,
在每个 batch 开始或者结束时执行一些操作, 但这种用法相对少见

大部分时候, tf.keras.callbacks 子模块中定义的回调函数类已经足够使用了,
如果有特定的需要, 也可以通过对 tf.keras.callbacks.Callbacks 实施子类化构造自定义的回调函数
所有回调函数都继承至 tf.keras.callbacks.Callbacks 基类, 拥有 params 和 model 这两个属性:
其中 params 是一个dict, 记录了训练相关参数 [例如 verbosity, batch size, number of epochs 等等]
model 即当前关联的模型的引用

此外,对于回调类中的一些方法如 on_epoch_begin, on_batch_end, 还会有一个输入参数 logs, 提供有关当前 epoch 或者 batch 的一些信息
并能够记录计算结果, 如果 model.fit 指定了多个回调函数类, 这些 logs 变量将在这些回调函数类的同名函数间依顺序传递


一. 内置回调函数
1. BaseLogger, 收集每个 epoch 上 metrics 在各个 batch 上的平均值, 对 stateful_metrics 参数中的带中间状态的指标直接拿最终值无需对各个 batch 平均, 指标均值结果将添加到 logs 变量中, 该回调函数被所有模型默认添加, 且是第一个被添加的.

2. History, 将 BaseLogger 计算的各个 epoch 的 metrics 结果记录到 history 这个 dict 变量中, 作为 model.fit 的返回值, 该回调函数被所有模型默认添加, 在 BaseLogger 之后被添加

3. EarlyStopping, 当被监控指标在设定的若干个 epoch 后没有提升, 则提前终止训练
4. TensorBoard, 为 Tensorboard 可视化保存日志信息, 支持评估指标, 计算图, 模型参数等的可视化
5. ModelCheckpoint, 在每个 epoch 后保存模型
6. ReduceLROnPlateau, 如果监控指标在设定的若干个 epoch 后没有提升, 则以一定的因子减少学习率
7. TerminateOnNaN, 如果遇到 loss 为 NaN, 提前终止训练
8. LearningRateScheduler, 学习率控制器, 给定学习率 lr 和 epoch 的函数关系, 据该函数关系在每个 epoch 前调整学习率
9. CSVLogger, 将每个 epoch 后的 logs 结果记录到 CSV 文件中
10. ProgbarLogger, 将每个 epoch 后的 logs 结果打印到标准输出流中


二. 自定义回调函数
1. 使用 tf.keras.callbacks.LambdaCallback 编写较为简单的回调函数
2. 可以通过对 tf.keras.callbacks.Callback 子类化编写更加复杂的回调函数逻辑
3. 如果需要深入学习 tf.Keras 中的回调函数, 不要犹豫阅读内置回调函数的源代码
"""

# 示范使用 LambdaCallback 编写较为简单的回调函数
json_log = open('./keras_log.json', mode='wt', buffering=1)
json_logging_callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: json_log.write(
        json.dumps(dict(epoch=epoch, **logs)) + '\n'),
    on_train_end=lambda logs: json_log.close()
)


# 示范通过 tf.keras.callbacks.Callback 子类化编写回调函数 [LearningRateScheduler的源代码]
class LearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:  
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            lr = self.schedule(epoch, lr)
        except TypeError:  # Support for old API for backward compatibility
            lr = self.schedule(epoch)
        if not isinstance(lr, (tf.Tensor, float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        if isinstance(lr, ops.Tensor) and not lr.dtype.is_floating:
            raise ValueError('The dtype of Tensor should be float')
        K.set_value(self.model.optimizer.lr, tf.keras.backend.get_value(lr))
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler reducing learning '
                 'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)

tf.print("\033[1;33;40m Callbacks in TensorFlow is successfull. \033[0m")