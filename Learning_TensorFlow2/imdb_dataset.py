#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: TensorFlow2 for text IMDB dataset
@Brief: IMDB dataset 目标是根据电影评论的文本内容预测评论的情感标签
        训练集有 20000 条电影评论文本, 测试集有 5000 条电影评论文本, 其中正面评论和负面评论都各占一半.
        文本数据预处理较为繁琐, 包括中文切词, 构建词典, 编码转换, 序列填充, 构建数据管道等等.
@Dataset: https://gitee.com/Python_Ai_Road/eat_tensorflow2_in_30_days/tree/master/data/imdb
@Dataset: https://github.com/lyhue1991/eat_tensorflow2_in_30_days/tree/master/data/imdb
@Python Version: 3.8.12
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-03-18
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

import tensorflow as tf
import re, string
import os, pathlib


# ----------------------------
# Stage 1. Dataset Processing
# ---------------------------
# TensorFlow 中完成文本数据预处理的常用方案有两种
# 第一种是利用 tf.keras.preprocessing 中的 Tokenizer 词典构建工具和 tf.keras.utils.Sequence 构建文本数据生成器管道
# 第二种是使用 tf.data.Dataset 搭配 tf.keras.layers.experimental.preprocessing.TextVectorization 预处理层
path2train_text = r"./imdb/train.csv"
path2test_text = r"./imdb/test.csv"

# 考虑最高频率的 10000 个单词
Max_Frequency_Word = 10000
# 每个样本保留 200 个单词的长度
Max_Length = 200
Batch_Size = 64

def split_line(line):
    arr = tf.strings.split(line, "\t")
    label = tf.expand_dims(tf.cast(tf.strings.to_number(arr[0]), tf.int32), axis=0)
    text = tf.expand_dims(arr[1], axis=0)

    return (text, label)

train_raw_data = tf.data.TextLineDataset(filenames=[path2train_text]).map(split_line, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(buffer_size=1024).batch(Batch_Size).prefetch(tf.data.experimental.AUTOTUNE)

test_raw_data = tf.data.TextLineDataset(filenames=[path2test_text]).map(split_line, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(Batch_Size).prefetch(tf.data.experimental.AUTOTUNE)

def clean_text(text):
    lowercase = tf.strings.lower(text)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    cleaned_punctuation  = tf.strings.regex_replace(stripped_html, "[%s]" % re.escape(string.punctuation), "")

    return cleaned_punctuation

vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    standardize=clean_text,
    split="whitespace",
    max_tokens=Max_Frequency_Word - 1,
    output_mode="int",
    output_sequence_length=Max_Length)

text_data = train_raw_data.map(lambda text, label: text)
vectorize_layer.adapt(text_data)
print(vectorize_layer.get_vocabulary()[0:100])


train_data = train_raw_data.map(lambda text, label: (vectorize_layer(text), label)).prefetch(tf.data.experimental.AUTOTUNE)
test_data = test_raw_data.map(lambda text, label: (vectorize_layer(text), label)).prefetch(tf.data.experimental.AUTOTUNE)


# ----------------------------
# Stage 2. Define Models
# ---------------------------
tf.keras.backend.clear_session()

class CNNModel(tf.keras.models.Model):
    def __init__(self):
        super(CNNModel, self).__init__()

    def build(self, input_shape):
        self.embedding = tf.keras.layers.Embedding(Max_Frequency_Word, 7, input_length=Max_Length)
        self.conv_1 = tf.keras.layers.Conv1D(16, kernel_size=5, name="conv_1", activation="relu")
        self.pool_1 = tf.keras.layers.MaxPool1D(name="pool_1")
        self.conv_2 = tf.keras.layers.Conv1D(128, kernel_size=2, name="conv_2", activation="relu")
        self.pool_2 = tf.keras.layers.MaxPool1D(name="pool_2")
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1, activation="sigmoid")
        super(CNNModel, self).build(input_shape)

    def call(self, x):
        x = self.embedding(x)
        x = self.conv_1(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = self.flatten(x)
        x = self.dense(x)

        return x

    def summary(self):
        x_input = tf.keras.layers.Input(shape=Max_Length)
        output = self.call(x_input)
        model = tf.keras.Model(inputs=x_input, outputs=output)
        model.summary()

model = CNNModel()
model.build(input_shape=(None, Max_Length))
model.summary()


# ----------------------------
# Stage 3. Training Models
# ---------------------------
# 使用修饰器，使得构建静态图，而不需要 session
@tf.function
def printbar():
    today_ts = tf.timestamp() % (24*60*60)
    hour = tf.cast(today_ts // 3600 + 8, tf.int32) % tf.constant(24)
    minute = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m)) == 1:
            return (tf.strings.format("0{}", m))
        else:
            return (tf.strings.format("{}", m))

    timestring = tf.strings.join([timeformat(hour), timeformat(minute), timeformat(second)], separator=":")
    tf.print("===================="*5 + timestring)

# configure of training phase
optimizer = tf.keras.optimizers.Nadam()
loss_func = tf.keras.losses.BinaryCrossentropy()

train_loss = tf.keras.metrics.Mean(name="train_loss")
train_metric = tf.keras.metrics.BinaryAccuracy(name="train_accuracy")

valid_loss = tf.keras.metrics.Mean(name="valid_loss")
valid_metric = tf.keras.metrics.BinaryAccuracy(name="valid_accuracy")

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
    predictions = model(features, training=False)
    batch_loss = loss_func(labels, predictions)
    valid_loss.update_state(batch_loss)
    valid_metric.update_state(labels, predictions)

def train_model(model, train_data, valid_data, epochs):
    for epoch in tf.range(1, epochs+1):
        for features, labels in train_data:
            train_step(model, features, labels)

        for features, labels in valid_data:
            valid_step(model, features, labels)

        # logs 模板需要根据 metric 具体情况修改
        logs = 'Epoch = {}, Loss: {}, Accuracy: {}, Valid Loss: {}, Valid Accuracy: {}'
        if epoch % 1 == 0:
            printbar()
            tf.print(tf.strings.format(logs, (epoch, train_loss.result(), train_metric.result(), valid_loss.result(), valid_metric.result())))
            tf.print("") # new line
        
        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()

# ===================================
train_model(model, train_data, test_data, epochs=6)


# ----------------------------
# Stage 4. Evaluate Models
# ---------------------------
# 通过自定义训练循环训练的模型没有经过编译 compile，无法直接使用 model.evaluate(ds_valid) 方法
def evaluate_model(model, valid_data):
    for features, labels in valid_data:
        valid_step(model, features, labels)

    logs = 'Valid Loss: {}, Valid Accuracy: {}'
    tf.print(tf.strings.format(logs, (valid_loss.result(), valid_metric.result())))

    valid_loss.reset_states()
    valid_metric.reset_states()
    
evaluate_model(model, test_data)

# ----------------------------
# Stage 5. Using Modles
# ---------------------------
# 可以使用以下方法:
# 1. model.predict(test_data)
# 2. model(test_data)
# 3. model.call(test_data)
# 4. model.predict_on_batch(test_data)
# 推荐优先使用 model.predict(test_data) 方法，既可以对 Dataset, 也可以对 Tensor 使用
tf.print(model.predict(test_data))

for test_sample, _ in test_data.take(1):
    tf.print(model(test_sample))
    tf.print(model.call(test_sample))


# ----------------------------
# Stage 6. Saving Models and Load Models
# ---------------------------
path2model_save = r"./Models/imdb"
os.makedirs(path2model_save, exist_ok=True)

model.save(pathlib.Path(path2model_save), save_format="tf")
print(f"The Trained Model Save at \033[1;33;40m {path2model_save} \033[0m with TensorFlow Format Successfully.")

print("-------------------------------------------------------")
model_loaded = tf.keras.models.load_model(pathlib.Path(path2model_save))
model.compile(optimizer=optimizer, loss=loss_func)
evaluate_model(model_loaded, test_data)