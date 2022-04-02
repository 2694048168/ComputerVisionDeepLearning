#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The high-level API of TensorFlow2
@Python Version: 3.8.12
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-03-29
"""

""" 
TensorFlow 的高阶 API
high-level API of TensorFlow 主要是 tensorflow.keras.models

1. 模型的构建: Sequential、functional API、Model子类化
2. 模型的训练: 内置 fit 方法, 内置 train_on_batch 方法, 自定义训练循环, 单 GPU 训练模型, 多 GPU 训练模型, TPU训练模型
3. 模型的部署: tensorflow serving 部署模型, 使用 spark(scala) 调用 tensorflow 模型

使用单 GPU 训练模型
深度学习的训练过程常常非常耗时,一个模型训练几个小时,训练几天也是常有的事情,有时候甚至要训练几十天
训练过程的耗时主要来自于两个部分, 一部分来自数据准备, 另一部分来自参数迭代
1. 当数据准备过程还是模型训练时间的主要瓶颈时, 可以使用更多进程来准备数据
2. 当参数迭代过程成为训练时间的主要瓶颈时, 通常的方法是应用 GPU/Google TPU 来进行加速

https://zhuanlan.zhihu.com/p/68509398

无论是内置 fit 方法, 还是自定义训练循环, 从 CPU 切换成 单 GPU 训练模型都是非常方便的, 无需更改任何代码.
当存在可用的 GPU 时, 如果不特意指定 device, tensorflow 会自动优先选择使用 GPU 来创建张量和执行张量计算

但如果是在公司或者学校实验室的服务器环境, 存在 多个GPU 和 多个使用者时, 
为了不让单个同学的任务占用全部 GPU 资源导致其他同学无法使用,
tensorflow 默认获取 全部 GPU 的 全部内存资源权限, 但实际上只使用一个 GPU 的部分资源,
通常会在开头增加几行代码以控制每个任务使用的 GPU编号和显存大小, 以便其他同学也能够同时训练模型
"""

import tensorflow as tf


print(f"TensorFlow Version : {tf.__version__}")
print(f"Host Computer supported Devices : {tf.config.list_physical_devices()}")

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


# -------------------------
# GPU setting
# -------------------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    gpu_0 = gpus[0]
    tf.config.experimental.set_memory_growth(gpu_0, True) # 显存按需申请使用

    # 或者也可以设置 GPU 显存为固定使用量(例如：4G)
    # tf.config.experimental.set_virtual_device_configuration(gpu0, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

    tf.config.set_visible_devices([gpu_0], "GPU")


# 比较 GPU 和 CPU 的计算速度
tf.print("\033[1;33;40m matrix multiplication on GPU \033[0m")
printbar()
with tf.device("/gpu:0"):
    tf.random.set_seed(42)
    a = tf.random.uniform((10000, 100), minval=0, maxval=3.0)
    b = tf.random.uniform((100, 100000), minval=0, maxval=3.0)
    c = a @ b
    tf.print(tf.reduce_sum(tf.reduce_sum(c, axis=0), axis=0))
printbar()

tf.print("\033[1;33;40m matrix multiplication on CPU \033[0m")
printbar()
with tf.device("/cpu:0"):
    tf.random.set_seed(42)
    a = tf.random.uniform((10000, 100), minval=0, maxval=3.0)
    b = tf.random.uniform((100,100000), minval=0, maxval=3.0)
    c = a @ b
    tf.print(tf.reduce_sum(tf.reduce_sum(c, axis=0), axis=0))
printbar()

# Preprocessing data for Model
MAX_LEN = 300
BATCH_SIZE = 32
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.reuters.load_data()
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_LEN)

MAX_WORDS = x_train.max() + 1
CAT_NUM = y_train.max() + 1

ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
          .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
          .prefetch(tf.data.experimental.AUTOTUNE).cache()

ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
          .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
          .prefetch(tf.data.experimental.AUTOTUNE).cache()


# Define the Model
tf.keras.backend.clear_session()

def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(MAX_WORDS, 7, input_length=MAX_LEN))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation="relu"))
    model.add(tf.keras.layers.MaxPool1D(2))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu"))
    model.add(tf.keras.layers.MaxPool1D(2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(CAT_NUM, activation="softmax"))

    return model

model = create_model()
model.summary()

# Training the Model
optimizer = tf.keras.optimizers.Nadam()
loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = loss_func(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)
    train_metric.update_state(labels, predictions)

@tf.function
def valid_step(model, features, labels):
    predictions = model(features)
    batch_loss = loss_func(labels, predictions)
    valid_loss.update_state(batch_loss)
    valid_metric.update_state(labels, predictions)

def train_model(model, ds_train, ds_valid, epochs):
    for epoch in tf.range(1, epochs + 1):
        for features, labels in ds_train:
            train_step(model, features, labels)

        for features, labels in ds_valid:
            valid_step(model, features, labels)

        logs = 'Epoch={}, Loss:{}, Accuracy:{}, Valid Loss:{}, Valid Accuracy:{}'
        if epoch % 1 == 0:
            printbar()
            tf.print(tf.strings.format(logs,
            (epoch,train_loss.result(),train_metric.result(),valid_loss.result(),valid_metric.result())))
            tf.print("")

        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()

train_model(model, ds_train, ds_test, 10)