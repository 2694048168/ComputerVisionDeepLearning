#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The Hierarchy of TensorFlow2
@Image of Hierarchy: ./images/hierarchy_tensorflow.png 
@Brief: https://developers.google.com/machine-learning/crash-course/first-steps-with-tensorflow/toolkit
@Python Version: 3.8.12
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-03-21
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
X = tf.random.uniform([num_samplers, 2], minval=-10, maxval=10)
w0 = tf.constant([[2.0], [-3.0]])
b0 = tf.constant([[3.0]])
Y = X @ w0 + b0 + tf.random.normal([num_samplers, 1], mean=0.0, stddev=2.0)

# data visualization
plt.figure(figsize=(10, 5))
ax_1 = plt.subplot(121)
ax_1.scatter(X[:, 0], Y[:, 0], color="b")
plt.title("Data Visualization for feature X1")
plt.xlabel("x1")
plt.ylabel("y", rotation=0)

ax_2 = plt.subplot(122)
ax_2.scatter(X[:, 1], Y[:, 0], color="g")
plt.title("Data Visualization for feature X2")
plt.xlabel("x2")
plt.ylabel("y", rotation=0)

# plt.show()
plt.savefig(str(pathlib.Path(os.path.join("./images", "linear_regress.png"))), dpi=120)
plt.close()

# Building a data pipeline iterators
def data_iter(features, labels, batch_size=8):
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)
    for idx in range(0, num_examples, batch_size):
        indexes = indices[idx: min(idx + batch_size, num_examples)]
        yield tf.gather(features, indexes), tf.gather(labels, indexes)

batch_size = 8
(features, labels) = next(data_iter(X, Y, batch_size))
print(f"The features of data is {features}")
print(f"The lables of data is {labels}")

# Define the Model
w = tf.Variable(tf.random.normal(w0.shape))
b = tf.Variable(tf.zeros_like(b0, dtype=tf.float32))

class LinearRegression:
    def __call__(self, x):
        return x @ w + b

    def loss_func(self, y_true, y_pred):
        return tf.reduce_mean((y_true - y_pred)**2 / 2)

model = LinearRegression()

# Training Model
@tf.function # comment this line, the use of dynamic compute graph mode
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = model.loss_func(labels, predictions)

    dloss_dw, dloss_db = tape.gradient(loss, [w, b])
    learning_rate = 0.01
    w.assign(w - learning_rate * dloss_dw)
    b.assign(b - learning_rate * dloss_db)

    return loss

loss_step = train_step(model, features, labels)
print(f"The loss of Model on a step is {loss_step}")

# 可以通过查看训练所需的时间，验证静态图和动态图的时间效率
@tf.function
def train_model(model, epochs):
    for epoch in tf.range(1, epochs + 1):
        for features, labels in data_iter(X, Y, 10):
            loss = train_step(model, features, labels)

        if epoch % 50 == 0:
            printbar()
            tf.print(f"epoch = {epoch}, loss = {loss}")
            tf.print(f"w = {w}, b = {b}")

train_model(model, epochs=200)

# the effect of the visualization of linear fitting
plt.figure(figsize=(12, 5))

ax1 = plt.subplot(121)
ax1.scatter(X[:, 0], Y[:, 0], color="b", label="samples")
ax1.plot(X[:, 0], w[0] * X[:, 0] + b[0], "-r", linewidth=5.0, label="model")
ax1.legend()
plt.xlabel("x1")
plt.ylabel("y", rotation=0)

ax2 = plt.subplot(122)
ax2.scatter(X[:, 1], Y[:, 0], color="g", label="samples")
ax2.plot(X[:, 1], w[1] * X[:,1] + b[0], "-r", linewidth=5.0, label="model")
ax2.legend()
plt.xlabel("x2")
plt.ylabel("y", rotation = 0)

# plt.show()
plt.savefig(str(pathlib.Path(os.path.join("./images", "linear_regress_fitting.png"))), dpi=120)
plt.close()

# -------------------------------------------------
# using neural network for binary classification
# -------------------------------------------------
print("=============== \033[1;31;40m The Second Demo with Neural Network \033[0m ==================")
num_positive, num_negative = 2000, 2000

# generate positive samplers with small circle distribution
radius_positive = 5.0 + tf.random.truncated_normal([num_positive, 1], 0.0, 1.0)
theta_positive = tf.random.uniform([num_positive, 1], 0.0, 2*np.pi)
x_positive = tf.concat([radius_positive * tf.cos(theta_positive), radius_positive * tf.sin(theta_positive)], axis=1)
y_positive = tf.ones_like(radius_positive)

# generate negative samplers with big circle distribution
radius_negative = 8.0 + tf.random.truncated_normal([num_negative, 1], 0.0, 1.0)
theta_negative = tf.random.uniform([num_negative, 1], 0.0, 2*np.pi)
x_negative = tf.concat([radius_negative * tf.cos(theta_negative), radius_negative * tf.sin(theta_negative)], axis=1)
y_negative = tf.zeros_like(radius_negative)

x = tf.concat([x_positive, x_negative], axis=0)
y = tf.concat([y_positive, y_negative], axis=0)

plt.figure(figsize=(6, 6))
plt.scatter(x_positive[:, 0].numpy(), x_positive[:, 1].numpy(), color="r")
plt.scatter(x_negative[:, 0].numpy(), x_negative[:, 1].numpy(), color="g")
plt.legend(["positive", "negative"])
# plt.show()
plt.savefig(str(pathlib.Path(os.path.join("./images", "circle_data.png"))), dpi=120)
plt.close()


# Building data pipeline iterators 
def data_iter_DNN(features, labels, batch_size=8):
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        indexes = indices[i: min(i + batch_size, num_examples)]
        yield tf.gather(features, indexes), tf.gather(labels, indexes)


# test the data pipeline iterators
features, labels = next(data_iter_DNN(x, y, 10))
print(f"The shape of features is : {features.shape}")
print(f"The value of features is : {features}")
print(f"The shape of labels is : {labels.shape}")
print(f"The value of labels is : {labels}")

# Define the Model
class DNN_Model(tf.Module):
    def __init__(self, name=None):
        super(DNN_Model, self).__init__(name=name)
        self.w1 = tf.Variable(tf.random.truncated_normal([2, 4]), dtype=tf.float32)
        self.b1 = tf.Variable(tf.zeros([1, 4]), dtype=tf.float32)
        self.w2 = tf.Variable(tf.random.truncated_normal([4, 8]), dtype=tf.float32)
        self.b2 = tf.Variable(tf.zeros([1, 8]), dtype=tf.float32)
        self.w3 = tf.Variable(tf.random.truncated_normal([8, 1]), dtype=tf.float32)
        self.b3 = tf.Variable(tf.zeros([1, 1]), dtype=tf.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float32)])
    def __call__(self, x):
        x = tf.nn.relu(x @ self.w1 + self.b1)
        x = tf.nn.relu(x @ self.w2 + self.b2)
        y = tf.nn.sigmoid(x @ self.w3 + self.b3)
        return y
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float32), tf.TensorSpec(shape=[None, 1], dtype=tf.float32)])
    def loss_func(self, y_true, y_pred):
        # 将预测值限制在 1e-7 以上, 1 - 1e-7 以下, avoid log(0) error
        epsilon_limit = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon_limit, 1.0 - epsilon_limit)
        binary_cross_entropy = - y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        return tf.reduce_mean(binary_cross_entropy)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float32), tf.TensorSpec(shape=[None, 1], dtype=tf.float32)])
    def metric_func(self, y_true, y_pred):
        y_pred = tf.where(y_pred > 0.5, tf.ones_like(y_pred, dtype=tf.float32), tf.zeros_like(y_pred, dtype=tf.float32))
        accuracy = tf.reduce_mean(1 - tf.abs(y_true - y_pred))
        return accuracy


# instance of class DNN_Model
model = DNN_Model()
predictions = model(features)
loss = model.loss_func(labels, predictions)
metric = model.metric_func(labels, predictions)

tf.print("The initialize loss of the Model is = {}".format(loss))
tf.print("The initialize metric of the Model is = {}".format(metric))
tf.print("The trainable variables of the Model are = {}".format(len(model.trainable_variables)))


# Training Model
@tf.function
def train_step(model, features, labels):
    # 1. feedward to compute tape gradient with AutoGrad
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = model.loss_func(labels, predictions)

    # 2. backward to compute gradients
    grads = tape.gradient(loss, model.trainable_variables)
 
    # 3. gradient descent (SGD)
    learning_rate = 0.001
    for param, dloss_param in zip(model.trainable_variables, grads):
        param.assign(param - learning_rate * dloss_param)

    # 4. compute metrics
    metric = model.metric_func(labels, predictions)

    return loss, metric


def train_model(model, epochs=1000):
    for epoch in tf.range(1, epochs + 1):
        for features, labels in data_iter_DNN(x, y, 100):
            loss, metric = train_step(model, features, labels)

        if epoch % 50 == 0:
            printbar()
            tf.print(f"The epoch = {epoch}, loss = {loss}, accuracy = {metric}")

# call train_model
train_model(model)


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


# ===============================
# reusable libaries API of TensorFLow
# ===============================

# ===============================
# high-level API of TensorFLow
# ===============================