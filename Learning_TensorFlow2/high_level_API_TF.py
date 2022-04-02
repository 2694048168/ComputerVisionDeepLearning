#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The high-level API of TensorFlow2
@Python Version: 3.8.12
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-03-29
"""

import os, pathlib
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf

""" 
TensorFlow 的高阶 API
high-level API of TensorFlow 主要是 tensorflow.keras.models

1. 模型的构建: Sequential、functional API、Model子类化
2. 模型的训练: 内置 fit 方法, 内置 train_on_batch 方法, 自定义训练循环, 单 GPU 训练模型, 多 GPU 训练模型, TPU训练模型
3. 模型的部署: tensorflow serving 部署模型, 使用 spark(scala) 调用 tensorflow 模型

构建模型的3种方法
1. 使用 Sequential 按层顺序构建模型
2. 使用函数式 API 构建任意结构模型
3. 继承 Model 基类构建自定义模型

对于顺序结构的模型, 优先使用 Sequential 方法构建
如果模型有多输入或者多输出, 或者模型需要共享权重, 或者模型具有残差连接等非顺序结构, 推荐使用函数式API进行创建
如果无特定必要, 尽可能避免使用 Model 子类化的方式构建模型, 这种方式提供了极大的灵活性, 但也有更大的概率出错
"""
train_token_path = r"./imdb/train_token.csv"
test_token_path = r"./imdb/test_token.csv"

MAX_WORDS = 10000  # We will only consider the top 10,000 words in the dataset
MAX_LEN = 200  # We will cut reviews after 200 words
BATCH_SIZE = 20 

# 构建管道
def parse_line(line):
    t = tf.strings.split(line, "\t")
    label = tf.reshape(tf.cast(tf.strings.to_number(t[0]), tf.int32), (-1,))
    features = tf.cast(tf.strings.to_number(tf.strings.split(t[1], " ")), tf.int32)
    return (features, label)

ds_train=  tf.data.TextLineDataset(filenames=[train_token_path]) \
   .map(parse_line, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
   .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
   .prefetch(tf.data.experimental.AUTOTUNE)

ds_test=  tf.data.TextLineDataset(filenames=[test_token_path]) \
   .map(parse_line, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
   .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
   .prefetch(tf.data.experimental.AUTOTUNE)


# -------------------------------
# way-1 with Sequential
tf.keras.backend.clear_session()

model_sequential = tf.keras.Sequential()
model_sequential.add(tf.keras.layers.Embedding(MAX_WORDS, 7, input_length=MAX_LEN))
model_sequential.add(tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation="relu"))
model_sequential.add(tf.keras.layers.MaxPool1D(2))
model_sequential.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu"))
model_sequential.add(tf.keras.layers.MaxPool1D(2))
model_sequential.add(tf.keras.layers.Flatten())
model_sequential.add(tf.keras.layers.Dense(1, activation="sigmoid"))

model_sequential.compile(optimizer='Nadam',
            loss='binary_crossentropy',
            metrics=["accuracy", "AUC"])

model_sequential.summary()

# ==== Error-bug ==== in current TensorFlow version
# baselogger_callback = tf.keras.callbacks.BaseLogger(stateful_metrics=["auc"])

path2log_folder = r"./tensorboard/keras_model"
os.makedirs(path2log_folder, exist_ok=True)
logdir = str(pathlib.Path(os.path.join(path2log_folder, datetime.datetime.now().strftime(r"%Y%m%d-%H%M%S"))))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

history_sequential = model_sequential.fit(ds_train,
                                epochs=6,
                                validation_data=ds_test,
                                callbacks=[tensorboard_callback])

print(type(history_sequential.history))
print(history_sequential.history.keys())

def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+ metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.show()
    # plt.savefig("./images/high_level_1.png", dpi=120)
    plt.close()

plot_metric(history_sequential, "auc")
plot_metric(history_sequential, "accuracy")
plot_metric(history_sequential, "loss")


# -------------------------------
# way-2 with functional API
tf.keras.backend.clear_session()

inputs = tf.keras.layers.Input(shape=[MAX_LEN])
x = tf.keras.layers.Embedding(MAX_WORDS, 7)(inputs)

branch1 = tf.keras.layers.SeparableConv1D(64, 3, activation="relu")(x)
branch1 = tf.keras.layers.MaxPool1D(3)(branch1)
branch1 = tf.keras.layers.SeparableConv1D(32, 3, activation="relu")(branch1)
branch1 = tf.keras.layers.GlobalMaxPool1D()(branch1)

branch2 = tf.keras.layers.SeparableConv1D(64, 5, activation="relu")(x)
branch2 = tf.keras.layers.MaxPool1D(5)(branch2)
branch2 = tf.keras.layers.SeparableConv1D(32, 5, activation="relu")(branch2)
branch2 = tf.keras.layers.GlobalMaxPool1D()(branch2)

branch3 = tf.keras.layers.SeparableConv1D(64, 7, activation="relu")(x)
branch3 = tf.keras.layers.MaxPool1D(7)(branch3)
branch3 = tf.keras.layers.SeparableConv1D(32, 7, activation="relu")(branch3)
branch3 = tf.keras.layers.GlobalMaxPool1D()(branch3)

concat = tf.keras.layers.Concatenate()([branch1, branch2, branch3])
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(concat)

model_functional = tf.keras.models.Model(inputs=inputs, outputs=outputs)
model_functional.summary()

model_functional.compile(optimizer='Nadam',
            loss='binary_crossentropy',
            metrics=['accuracy', "AUC"])

history_functional = model_functional.fit(ds_train,
                                epochs=6,
                                validation_data=ds_test,
                                callbacks=[tensorboard_callback])

plot_metric(history_functional, "auc")
plot_metric(history_functional, "accuracy")
plot_metric(history_functional, "loss")


# ---------------------------------------------
# way-3 with instance of tf.keras.Model class
tf.keras.backend.clear_session()

# 先自定义一个残差模块, 为自定义 Layer
class ResBlock(tf.keras.layers.Layer):
    def __init__(self, kernel_size, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self,input_shape):
        self.conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=self.kernel_size, activation="relu", padding="same")
        self.conv2 = tf.keras.layers.Conv1D(filters=32, kernel_size=self.kernel_size, activation="relu", padding="same")
        self.conv3 = tf.keras.layers.Conv1D(filters=input_shape[-1], kernel_size=self.kernel_size, activation="relu", padding="same")
        self.maxpool = tf.keras.layers.MaxPool1D(2)
        super(ResBlock,self).build(input_shape) # 相当于设置 self.built=True

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = tf.keras.layers.Add()([inputs, x])
        x = self.maxpool(x)
        return x

    # 如果要让自定义的 Layer 通过 Functional API 组合成模型时可以序列化,需要自定义 get_config 方法
    def get_config(self):
        config = super(ResBlock, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config


# 测试 ResBlock
resblock = ResBlock(kernel_size=3)
resblock.build(input_shape=(None,200,7))
print(resblock.compute_output_shape(input_shape=(None, 200, 7)))


# 自定义模型,实际上也可以使用 Sequential 或者 FunctionalAPI
class ImdbModel(tf.keras.models.Model):
    def __init__(self):
        super(ImdbModel, self).__init__()

    def build(self, input_shape):
        self.embedding = tf.keras.layers.Embedding(MAX_WORDS, 7)
        self.block1 = ResBlock(7)
        self.block2 = ResBlock(5)
        self.dense = tf.keras.layers.Dense(1, activation="sigmoid")
        super(ImdbModel,self).build(input_shape)

    def call(self, x):
        x = self.embedding(x)
        x = self.block1(x)
        x = self.block2(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.dense(x)
        return x


model_instance = ImdbModel()
model_instance.build(input_shape =(None,200))
model_instance.summary()

model_instance.compile(optimizer='Nadam',
            loss='binary_crossentropy',
            metrics=['accuracy', "AUC"])

history_instance = model_instance.fit(ds_train,
                                    validation_data=ds_test,
                                    epochs=6,
                                    callbacks=[tensorboard_callback])

plot_metric(history_instance, "auc")
plot_metric(history_instance, "accuracy")
plot_metric(history_instance, "loss")