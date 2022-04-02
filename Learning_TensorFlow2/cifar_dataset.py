#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: TensorFlow2 for CIFAR dataset
@Brief: CIFAR 是一个具有 10分类(CIFASR-10) 和 100分类(CIFASR-100) 的图像分类任务数据集
@Dataset: https://gitee.com/Python_Ai_Road/eat_tensorflow2_in_30_days/tree/master/data/cifar2
@Dataset: https://github.com/lyhue1991/eat_tensorflow2_in_30_days/tree/master/data/cifar2
@Python Version: 3.8.12
@Author: Wei Li
@Date: 2022-03-17
"""

# Overview of the source code
""" -------------------------------------------------
Stage 1. Dataset Processing
Stage 2. Define Models
Stage 3. Training Models
Stage 4. Evaluate Models
Stage 5. Using Modles
Stage 6. Saving Models and Load Models
-------------------------------------------------"""

import os
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

# -------------------------
# image data preprocessing
# -------------------------
# CIFAR-2 数据集为 CIFAR-10 数据集的子集，只包括前两种类别 airplane 和 automobile, 这样将问题的规模减小，原理保持不变
# 训练集有 airplane 和 automobile 图片各 5000 张，测试集有 airplane 和 automobile 图片各 1000 张
# CIFAR-2 任务的目标是训练一个模型来对飞机 airplane 和机动车 automobile 两种图片进行分类
# ----------------------------
# TensorFlow2 for image data preprocessing way-1 tf.keras.preprocessing.image.ImageDataGenerator
# TensorFlow2 for image data preprocessing way-2 tf.image and tf.data.Dataset
# ----------------------------
batch_size = 128
path2img_folder = r"./cifar2"
path2img_save_folder = r"./images"
os.makedirs(path2img_save_folder, exist_ok=True)

def load_image(img_path, size=(32, 32)):
    """Image jpeg file preprocessing from disk with TensorFlow2

    Args:
        img_path (string): path to image file.
        size (tuple, optional): resolution or size of image. Defaults to (32, 32).

    Returns:
        tuple tensor: the input image tensor and the output label tensor.
    """
    # automobile set its label to 1, airplane set its label to 0
    label = tf.constant(1, tf.int8) if tf.strings.regex_full_match(img_path, ".*automobile.*") else tf.constant(0, tf.int8)

    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img) 
    img = tf.image.resize(img, size) / 255.

    return (img, label)

# -----------------------------------------
# trick tips: using num_parallel_calls to 并行化进行预处理
# trick tips: using prefetch to 预存数据，避免 I/O 阻塞影响时间
# -----------------------------------------
train_img = tf.data.Dataset.list_files(os.path.join(path2img_folder, "train/*/*.jpg"))
train_img = train_img.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_img = train_img.shuffle(buffer_size=10*batch_size).batch(batch_size)
train_img = train_img.prefetch(tf.data.experimental.AUTOTUNE)

test_img = tf.data.Dataset.list_files(os.path.join(path2img_folder, "test/*/*.jpg")).map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# visualize some image samplers using matplotlib
def visualize_img_sampler(train_img, num_sample):
    """Visualize some image samplers in train dataset or test dataset.

    Args:
        train_img (tuple of tensor): TensorFlow2 tensor with image and label.
        num_sample (int): the number of image samplers must be able to be squared.
    """
    plt.figure(figsize=(8, 8))
    for idx, (img, label) in enumerate(train_img.unbatch().take(num_sample)):
        ax = plt.subplot(int(np.sqrt(num_sample)), int(np.sqrt(num_sample)), idx+1)
        ax.imshow(img.numpy())
        ax.set_title(f"label = {label}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()
    # plt.savefig(os.path.join(path2img_save_folder, "visual_img_data.png"), dpi=120)
    plt.close()

# visualize_img_sampler(train_img, 25)

# ------------------------------------------------------------
# PyCharm IDE 可以直接 Debug 查看每一个变量的类型和维度等各种信息
# VsCode 编辑器里面个人喜欢用 print 查看变量的类型和维度等信息
# ------------------------------------------------------------
# print(f"x_train_image tensor shape is = {train_img}")
# print(f"x_train_image tensor type is = {type(train_img)}")

# for x_img, y_label in train_img.take(1):
#     print(f"x_train_image each batch shape is = {x_img.shape}")
#     print(f"y_train_image each batch shape is = {y_label.shape}")
# ------------------------------------------------------------

# -------------------------
# Define Model for this task
# -------------------------
tf.keras.backend.clear_session()

def functional_model(num_classes):
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3))(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(5, 5))(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Dropout(rate=0.1)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model

model = functional_model(num_classes=1)
# model.summary()


# -------------------------------------
# Training builded Model on dataset
# -------------------------------------
path2log_folder_tensorboard = r"./tensorboard/cifar"
os.makedirs(path2log_folder_tensorboard, exist_ok=True)

# python 3 建议使用 pathlib 修正由于操作系统的引起的路径分隔符不同问题 (正斜杠 "\\" and 反斜杠 "/" )
stamp_time = datetime.datetime.now().strftime(r"%Y%m%d-%H%M%S")
tensorboard_log_folder = str(Path(os.path.join(path2log_folder_tensorboard, stamp_time))) 

tensorboard_callback = tf.keras.callbacks.TensorBoard(tensorboard_log_folder, histogram_freq=1)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.binary_crossentropy, 
    metrics=["accuracy"])

history = model.fit(train_img,
    epochs=100, 
    validation_data=test_img,
    callbacks=[tensorboard_callback], 
    workers=4)


# ------------------------------------------------------
# TensorBoard
# https://tensorflow.google.cn/tensorboard/get_started
# ------------------------------------------------------
# %tensorboard --logdir path2log_folder_tensorboard # in notebook
# tensorboard --logdir path2log_folder_tensorboard # in command

# print(f"------------------------------------------------------")
# print(f"The type of Model Return History is {type(history)}")
# print(f"The dir of Model Return History is {dir(history)}")
# print(f"The dir of Model Return History is {type(history.history)}")
# print(f"The dir of Model Return History is {dict(history.history)}")
# print(f"The dir of Model Return History is {dir(history.history)}")
# print(f"------------------------------------------------------")
df_history = pd.DataFrame(history.history)
df_history.index = range(1, len(df_history)+1)
df_history.index.name = "epoch"
# print(df_history)

# using matplotlib package to visualize loss and accuracy
def plot_metric(history, metric):
    train_metric  = history.history[metric]
    val_metric = history.history["val_" + metric]
    epochs = range(1, len(train_metric)+1)

    plt.plot(epochs, train_metric, "bo--")
    plt.plot(epochs, val_metric, "ro-")
    plt.title(f"Training and validation {metric}")
    plt.xlabel("Epochs")
    plt.ylabel(f"{metric}")
    plt.legend([f"train_{metric}", f"val_{metric}"])
    plt.show()
    # plt.savefig(os.path.join(path2img_save_folder, "metric_cifar.png"), dpi=120)
    plt.close()

plot_metric(history, "loss")
plot_metric(history, "accuracy")

val_loss, val_accuracy = model.evaluate(test_img, workers=4)
print(f"The loss on test dataset is {val_loss}")
print(f"The accuracy on test dataset is {val_accuracy}")

# -------------------------------
# Using  the trained Model
# -------------------------------
result = model.predict(test_img)
print(f"The predict of Model is \n {result}")
print("-----------------------------------")

for x_img, y_label in test_img.take(1):
    print(model.predict_on_batch(x_img[0:20]))
print("-----------------------------------")


# -----------------------------------
# Save Model for deploy application
# -----------------------------------
path2model_save_folder = r"./Models/CIFAR"
os.makedirs(path2model_save_folder, exist_ok=True)

# 终端显示颜色的格式 for python
# \033[显示方式;字体色;背景色m ...... \033[0m
# \033[1;31;47m {string} \033[0m
# \033[1;33;40m {string} \033[0m
# model.save(path2model_save_folder, save_format="tf")
print(f"The Trained Model Save at \033[1;31;47m {path2model_save_folder} \033[0m with TensorFlow Format Successfully.")
print(f"The Trained Model Save at \033[1;33;40m {path2model_save_folder} \033[0m with TensorFlow Format Successfully.")

model_loaded = tf.keras.models.load_model(path2model_save_folder)
result = model_loaded.predict(test_img)
print(f"The predict of Model is \n {result}")
print("-----------------------------------")