#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The Hierarchy of TensorFlow2
@Image of Hierarchy: ./images/hierarchy_tensorflow.png 
@Brief: https://developers.google.com/machine-learning/crash-course/first-steps-with-tensorflow/toolkit
@Python Version: 3.8.12
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-03-25
"""

"""
TensorFlow 5个不同的层次结构 Hierarchy: 即硬件层 hardware, 内核层 kernel, 低阶API, 中阶API, 高阶API
1. 最底层为硬件层 hardware, TensorFlow 支持 CPU、GPU TPU 加入计算资源池
2. C++实现的内核 kernel, kernel 可以跨平台分布运行
3. Python实现的操作符, 提供封装C++内核的低级API指令, 主要包括各种张量操作算子、计算图、自动微分. tf.Variable, tf.constant, tf.function, tf.GradientTape,t f.nn.softmax

4. Python实现的模型组件, 对低级API进行函数封装, 主要包括各种模型层, 损失函数, 优化器, 数据管道, 特征列等等. tf.keras.layers, tf.keras.losses, tf.keras.metrics, tf.keras.optimizers, tf.data.DataSet, tf.feature_column

5. Python实现的模型成品, 一般为按照 OOP方式 封装的高级API, 主要为 tf.keras.models 提供的模型的类接口
"""

import os, pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# ===============================
# low-level API of TensorFLow
# ===============================

# ===============================
# reusable libaries API of TensorFLow
# ===============================

# ===============================
# high-level API of TensorFLow
# ===============================
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

printbar()

# -------------------------------
# linear regression model
# -------------------------------
num_samplers = 400
x = tf.random.uniform([num_samplers, 2], minval=-10, maxval=10)
w0 = tf.constant([[2.0], [-3.0]])
b0 = tf.constant([[3.0]])
y = x @ w0 + b0 + tf.random.normal([num_samplers, 1], mean=0.0, stddev=2.0)

plt.figure(figsize=(12, 5))

ax1 = plt.subplot(121)
ax1.scatter(x[:, 0], y[:, 0], color="blue")
plt.xlabel("x1")
plt.ylabel("y", rotation=0)

ax2 = plt.subplot(122)
ax2.scatter(x[:, 1], y[:, 0], color="green")
plt.xlabel("x2")
plt.ylabel("y", rotation=0)

# plt.show()
plt.close()

# Define the Model
tf.keras.backend.clear_session()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(2, )))
model.summary()

# Training the Model with fit method 
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.fit(x, y, batch_size=10, epochs=200)

tf.print(f"The w of Model is {model.layers[0].kernel}")
tf.print(f"The b of Model is {model.layers[0].bias}")

# result visualization
w, b = model.variables

plt.figure(figsize=(12, 5))

ax1 = plt.subplot(121)
ax1.scatter(x[:, 0], y[:, 0],color="blue", label="samplers")
ax1.plot(x[:, 0], w[0] * x[:, 0] + b[0], "-r", linewidth=5.0, label="model")
ax1.legend()
plt.xlabel("x1")
plt.ylabel("y", rotation=0)

ax2 = plt.subplot(122)
ax2.scatter(x[:, 1], y[:, 0],color="green", label="samplers")
ax2.plot(x[:, 1], w[1] * x[:, 1] + b[0], "-r", linewidth=5.0, label="model")
ax2.legend()
plt.xlabel("x2")
plt.ylabel("y", rotation=0)

# plt.show()
plt.close()


# -------------------------------------------------
# using neural network for binary classification
# -------------------------------------------------
print("=============== \033[1;31;40m The Second Demo with Neural Network \033[0m ==================")

num_positive, num_negative = 2000, 2000
num_examples = num_negative + num_positive

radius_positive = 3.0 + tf.random.truncated_normal([num_positive, 1], 0.0, 1.0)
theta_positive = tf.random.uniform([num_positive, 1], 0.0, 2*np.pi)
x_positive = tf.concat([radius_positive * tf.cos(theta_positive), radius_positive * tf.sin(theta_positive)], axis=1)
y_positive = tf.ones_like(radius_positive)

radius_negative = 7.0 + tf.random.truncated_normal([num_negative, 1], 0.0, 1.0)
theta_negative = tf.random.uniform([num_negative, 1], 0.0, 2*np.pi)
x_negative = tf.concat([radius_negative * tf.cos(theta_negative), radius_negative * tf.sin(theta_negative)], axis=1)
y_negative = tf.zeros_like(radius_negative)

x = tf.concat([x_positive, x_negative], axis=0)
y = tf.concat([y_positive, y_negative], axis=0)

# sample shuffle
data = tf.concat([x, y], axis=1)
data = tf.random.shuffle(data)
x = data[:, :2]
y = data[:, 2:]

plt.figure(figsize=(6, 6))
plt.scatter(x_positive[:, 0].numpy(), x_positive[:, 1].numpy(), color="red")
plt.scatter(x_negative[:, 0].numpy(), x_negative[:, 1].numpy(), color="green")
plt.legend(["positive", "negative"])
# plt.show()
plt.close()

# Build input data pipeline for Model
# cache(缓存)性能, 使用 .cache() method: 当计算缓存空间足够时, 将 preprocess 的数据存储在缓存空间中将大幅提高计算速度
dataset_train = tf.data.Dataset.from_tensor_slices((x[0: num_examples * 3//4, :], y[0: num_examples * 3//4, :])).shuffle(buffer_size=1000).batch(20).prefetch(tf.data.experimental.AUTOTUNE).cache()

dataset_valid = tf.data.Dataset.from_tensor_slices((x[num_examples * 3//4: , :], y[num_examples * 3//4: , :])).batch(20).prefetch(tf.data.experimental.AUTOTUNE).cache()

# Define the Model
tf.keras.backend.clear_session()
class DNN_Model(tf.keras.models.Model):
    def __init__(self):
        super(DNN_Model, self).__init__()

    def build(self, input_shape):
        self.dense_1 = tf.keras.layers.Dense(4, activation="relu", name="dense_1")
        self.dense_2 = tf.keras.layers.Dense(8, activation="relu", name="dense_2")
        self.dense_3 = tf.keras.layers.Dense(1, activation="sigmoid", name="dense_3")
        super(DNN_Model, self).build(input_shape)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float32)])
    def call(self, x):
        x = self.dense_1(x)
        x = self.dense_2(x)
        y = self.dense_3(x)
        return y

model = DNN_Model()
model.build(input_shape=(None, 2))
model.summary()


# Training the Model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_func = tf.keras.losses.BinaryCrossentropy()

train_loss = tf.keras.metrics.Mean(name="train_loss")
train_metric = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

valid_loss = tf.keras.metrics.Mean(name="valid_loss")
valid_metric = tf.keras.metrics.BinaryAccuracy(name="valid_accuracy")


@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = loss_func(labels, predictions)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss.update_state(loss)
    train_metric.update_state(labels, predictions)


@tf.function
def valid_step(model, features, labels):
    predictions = model(features)
    batch_loss = loss_func(labels, predictions)
    valid_loss.update_state(batch_loss)
    valid_metric.update_state(labels, predictions)


@tf.function
def train_model(model, dataset_train, dataset_valid, epochs):
    for epoch in tf.range(1, epochs + 1):
        for features, labels in dataset_train:
            train_step(model, features, labels)

        for features, labels in dataset_valid:
            valid_step(model, features, labels)

        logs = 'Epoch={}, Loss:{}, Accuracy:{}, Valid Loss:{}, Valid Accuracy:{}'
        if epoch % 10 == 0:
            printbar()
            tf.print(tf.strings.format(logs, (epoch, train_loss.result(), train_metric.result(), valid_loss.result(), valid_metric.result())))

        train_loss.reset_states()
        train_metric.reset_states()
        valid_loss.reset_states()
        valid_metric.reset_states()

train_model(model, dataset_train, dataset_valid, epochs=100)


# results the visual
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
ax1.scatter(x_positive[:, 0], x_positive[:, 1], c="r")
ax1.scatter(x_negative[:, 0], x_negative[:, 1], c="g")
ax1.legend(["positive", "negative"])
ax1.set_title("y_true")

x_positive_pred = tf.boolean_mask(x, tf.squeeze(model(x) >= 0.5), axis=0)
x_negative_pred = tf.boolean_mask(x, tf.squeeze(model(x) < 0.5), axis=0)

ax2.scatter(x_positive_pred[:, 0], x_positive_pred[:, 1], c="r")
ax2.scatter(x_negative_pred[:, 0], x_negative_pred[:, 1], c="g")
ax2.legend(["positive", "negative"])
ax2.set_title("y_pred")

# plt.show()
plt.savefig(str(pathlib.Path(os.path.join("./images", "cicle_fitting.png"))), dpi=120)
plt.close()
