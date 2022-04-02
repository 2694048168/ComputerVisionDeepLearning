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

使用多 GPU 训练模型: 分布式训练 (数据并行, 模型并行, 数据和模型同时并行)
如果使用多 GPU 训练模型, 推荐使用内置 fit 方法, 较为方便, 仅需添加 2 行代码
MirroredStrategy 过程简介:
- 训练开始前, 该策略在所有 N 个计算设备上均各复制一份完整的模型
- 每次训练传入一个批次的数据时, 将数据分成 N 份, 分别传入 N 个计算设备(即数据并行)
- N 个计算设备使用本地变量(镜像变量)分别计算自己所获得的部分数据的梯度
- 使用分布式计算的 All-reduce 操作, 在计算设备间高效交换梯度数据并进行求和,使得最终每个设备都有了所有设备的梯度之和
- 使用梯度求和的结果更新本地变量(镜像变量)
- 当所有设备均更新本地变量后, 进行下一轮训练(即该并行策略是同步的)
"""

import os
import tensorflow as tf


print(f"TensorFlow Version : {tf.__version__}")
print(f"Host Computer supported Devices : {tf.config.list_physical_devices()}")
# 使用 1 个物理 GPU 模拟出两个逻辑 GPU 进行多 GPU 训练
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # 设置两个逻辑 GPU 模拟多 GPU 训练
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


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


def compile_model(model):
    model.compile(optimizer=tf.keras.optimizers.Nadam(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.SparseTopKCategoricalAccuracy(5)]) 
    return(model)


# Training Model with multi-GPU 增加以下两行代码
strategy = tf.distribute.MirroredStrategy()  
with strategy.scope(): 
    model = create_model()
    model.summary()
    model = compile_model(model)

history = model.fit(ds_train, validation_data=ds_test, epochs=10)


# =============================================================================
# 使用 TPU of Google 训练模型
# 如果想尝试使用 Google Colab 上的 TPU 来训练模型, 也是非常方便, 仅需添加 6 行代码
# 在 Colab 笔记本中: 修改->笔记本设置->硬件加速器 中选择 TPU
# =============================================================================

# =============================================================================
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.experimental.TPUStrategy(resolver)
# with strategy.scope():
#     model = create_model()
#     model.summary()
#     model = compile_model(model)

# history = model.fit(ds_train, validation_data=ds_test, epochs=10)
# =============================================================================