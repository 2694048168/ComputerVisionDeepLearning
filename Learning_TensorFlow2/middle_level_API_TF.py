#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The middle-level API of TensorFlow2
@Python Version: 3.8.12
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-03-28
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

import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import tensorflow as tf


# ========================
# data pipeline tf.data
# ========================
# 如果需要训练的数据大小不大, 例如不到 1G, 那么可以直接全部读入内存中进行训练, 这样一般效率最高
# 但如果需要训练的数据很大, 例如超过 10G, 无法一次载入内存, 那么通常需要在训练的过程中分批逐渐读入
# 使用 tf.data API 可以构建数据输入管道, 轻松处理大量的数据, 不同的数据格式, 以及不同的数据转换

# 可以从 Numpy array, Pandas DataFrame, Python generator, csv文件, 文本文件, 文件路径, tfrecords 文件等方式构建数据管道
# 其中通过 Numpy array, Pandas DataFrame, 文件路径构建数据管道是最常用的方法
# tfrecords 文件方式构建数据管道较为复杂, 需要对样本构建 tf.Example 后压缩成字符串写到 tfrecords 文件, 
# 读取后再解析成 tf.Example. tfrecords 文件的优点是压缩后文件较小, 便于网络传播, 加载速度较快

iris = datasets.load_iris()

# 1. numpy nd-array to build data pipeline for Model
print("\033[1;33;40m Numpy ndarray to build data pipeline for Model with TensorFlow \033[0m")
dataset_1 = tf.data.Dataset.from_tensor_slices((iris["data"], iris["target"]))
for feature, label in dataset_1.take(5):
    print(feature, label)

# 2. pandas DataFrame to build data pipeline for Model
print("\033[1;33;40m Pandas DataFrame to build data pipeline for Model with TensorFlow \033[0m")
dataset_df = pd.DataFrame(iris["data"], columns=iris.feature_names)
dataset_2 = tf.data.Dataset.from_tensor_slices((dataset_df.to_dict("list"), iris["target"]))
for feature, label in dataset_2.take(5):
    print(feature, label)

# 3. Python generator to build data pipeline for Model
print("\033[1;33;40m Python Generator to build data pipeline for Model with TensorFlow \033[0m")
path2image_data = r"./cifar2/test"
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255).flow_from_directory(
    path2image_data,
    target_size=(32, 32),
    batch_size=16,
    class_mode="binary"
)

classdict = image_generator.class_indices
print(classdict)

def generator():
    for feature, label in image_generator:
        yield (feature, label)

dataset_3 = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.int32))
plt.figure(figsize=(6, 6))
for idx, (img, label) in enumerate(dataset_3.unbatch().take(9)):
    ax = plt.subplot(3, 3, idx+1)
    ax.imshow(img.numpy())
    ax.set_title(f"label = {label}")
    ax.set_xticks([])
    ax.set_yticks([]) 
# plt.show()
plt.savefig("./images/img_example.png", dpi=120)
plt.close()

# 4. csv file to build data pipeline for Model
print("\033[1;33;40m CSV file to build data pipeline for Model with TensorFlow \033[0m")
dataset_4 = tf.data.experimental.make_csv_dataset(
    file_pattern=["./titanic/train.csv", "./titanic/test.csv"],
    batch_size=3,
    label_name="Survived",
    na_value="",
    num_epochs=1,
    ignore_errors=True
)

for data, label in dataset_4.take(2):
    print(data, label)


# 5. tex file to build data pipeline for Model
print("\033[1;33;40m TXT file to build data pipeline for Model with TensorFlow \033[0m")
dataset_5 = tf.data.TextLineDataset(
    filenames=["./titanic/train.csv", "./titanic/test.csv"]).skip(1)  # 省略第一行 header

for line in dataset_5.take(5):
    print(line)


# 6. file path to build data pipeline for Model
print("\033[1;33;40m file path to build data pipeline for Model with TensorFlow \033[0m")
dataset_6 = tf.data.Dataset.list_files("./cifar2/train/*/*.jpg")
for file in dataset_6.take(5):
    print(file)

def load_image(img_path, size=(32, 32)):
    label = 1 if tf.strings.regex_full_match(img_path, ".*/automobile/.*") else 0
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, size)
    return (img, label)

for idx, (img, label) in enumerate(dataset_6.map(load_image).take(2)):
    plt.figure(idx)
    plt.imshow((img/255.0).numpy())
    plt.title(f"lable = {label}")
    plt.xticks([])
    plt.yticks([])
# plt.show()
plt.close()


# 7. tfrecords file to build data pipeline for Model
print("\033[1;33;40m tfrecords file to build data pipeline for Model with TensorFlow \033[0m")
def create_tfrecords(in_path, out_path):
    """将输入的数据文件保存为 TFRecord 格式进行保存.

    Args:
        in_path (string): 原始数据路径
        out_path (string): TFRecord 文件输出路径
    """
    # with tf.io.TFRecordWriter(example_path) as file_writer:
    writer = tf.io.TFRecordWriter(out_path)
    dirs = os.listdir(in_path)
    for index, name in enumerate(dirs):
        class_path = in_path + "/" + name + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = tf.io.read_file(img_path)
            # img = tf.image.decode_image(img)
            # img = tf.image.encode_jpeg(img) # 统一成 jpeg 格式压缩
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.numpy()]))
                })
            )
            writer.write(example.SerializeToString())
    writer.close()

create_tfrecords(in_path="./cifar2/test", out_path="./cifar2/cifar_test.tfrecords")
print(f"The TFRecord file Saving at \033[1;31;47m ./cifar2/cifar_test.tfrecords \033[0m Successfully.")

def parse_example(proto):
    description = {
        "img_raw": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64)}

    example = tf.io.parse_single_example(proto, description)
    img = tf.image.decode_jpeg(example["img_raw"]) # 注意 images 格式 jpeg
    img = tf.image.resize(img, (32, 32))
    label = example["label"]

    return (img, label)

dataset_7 = tf.data.TFRecordDataset("./cifar2/cifar_test.tfrecords").map(parse_example).shuffle(3000)
plt.figure(figsize=(6, 6))
for idx, (img, lable) in enumerate(dataset_7.take(9)):
    ax = plt.subplot(3, 3, idx+1)
    ax.imshow((img/255.0).numpy())
    ax.set_title(f"label = {label}")
    ax.set_xticks([])
    ax.set_yticks([])
# plt.show()
plt.close()


# ----------------------------------
# application of data conversion
# ----------------------------------
# Dataset 数据结构应用非常灵活, 本质上是一个 Sequece 序列,
# 其每个元素可以是各种类型, 例如张量, 列表, 字典, 也可以是 Dataset
# Dataset 包含了非常丰富的数据转换功能:
# 1. map: 将转换函数映射到数据集每一个元素
# 2. flat_map: 将转换函数映射到数据集的每一个元素, 并将嵌套的 Dataset 压平
# 3. interleave: 效果类似 flat_map, 但可以将不同来源的数据夹在一起
# 4. filter: 过滤掉某些元素
# 5. zip: 将两个长度相同的 Dataset 横向铰合
# 6. concatenate: 将两个Dataset纵向连接
# 7. reduce: 执行归并操作
# 8. batch: 构建批次, 每次放一个批次, 比原始数据增加一个维度, 其逆操作为 unbatch
# 9. padded_batch: 构建批次, 类似 batch, 但可以填充到相同的形状
# 10. window: 构建滑动窗口, 返回 Dataset of Dataset
# 11. shuffle: 数据顺序洗牌
# 12. repeat: 重复数据若干次, 不带参数时, 重复无数次
# 13. shard: 采样, 从某个位置开始隔固定距离采样一个元素
# 14. take: 采样, 从开始位置取前几个元素
# ----------------------------------

dataset_8 = tf.data.Dataset.from_tensor_slices(["Hello World", "Hello China", "Hello Beijing"])

dataset_8_map = dataset_8.map(lambda x: tf.strings.split(x, " "))
for word in dataset_8_map:
    tf.print(word)
    print(word)
print()

dataset_8_flatmap = dataset_8.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(tf.strings.split(x, " ")))
for word in dataset_8_flatmap:
    print(word)
print()

dataset_8_interleave = dataset_8.interleave(lambda x: tf.data.Dataset.from_tensor_slices(tf.strings.split(x, " ")))
for word in dataset_8_interleave:
    print(word)
print()

# 找出含有字母 a 或 B 的元素
dataset_8_filter = dataset_8.filter(lambda x: tf.strings.regex_full_match(x, ".*[a|B].*"))
for word in dataset_8_filter:
    print(word)
print()

# zip
dataset_9 = tf.data.Dataset.range(0, 3)
dataset_10 = tf.data.Dataset.range(3, 6)
dataset_11 = tf.data.Dataset.range(6, 9)
dataset_zip = tf.data.Dataset.zip((dataset_9, dataset_10, dataset_11))
print(type(dataset_zip))
for x, y, z in dataset_zip:
    print(x.numpy())
    print(y.numpy())
    print(z.numpy())
print()

# condatenate
dataset_concat = tf.data.Dataset.concatenate(dataset_9, dataset_10)
print(type(dataset_concat))
for x in dataset_concat:
    print(x)
print()


# reduce: 执行归并操作
dataset_12 = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5.0])
result = dataset_12.reduce(0.0, lambda x, y: tf.add(x, y))
print(result)
print()

# batch: 构建批次, 每次放一个批次, 比原始数据增加一个维度, 其逆操作为 unbatch
dataset_13 = tf.data.Dataset.range(12)
dataset_batch = dataset_13.batch(4)
for x in dataset_batch:
    print(x)
print()

# padded_batch:构建批次, 类似 batch, 但可以填充到相同的形状
elements = [[1, 2], [3, 4, 5], [6, 7], [8]]
dataset_14 = tf.data.Dataset.from_generator(lambda: iter(elements), tf.int32)
dataset_padded_batch = dataset_14.padded_batch(2, padded_shapes=[4, ])
for x in dataset_padded_batch:
    print(x)    
print()


# window: 构建滑动窗口 (Swin Transformer 的实现), 返回 Dataset of Dataset
dataset_15 = tf.data.Dataset.range(12)
# window 返回的是 Dataset of Dataset, 可以用 flat_map 压平
dataset_window = dataset_15.window(3, shift=1).flat_map(lambda x: x.batch(3, drop_remainder=True)) 
for x in dataset_window:
    print(x)
print()

# shuffle: 数据顺序洗牌
dataset_16 = tf.data.Dataset.range(12)
dataset_shuffle = dataset_16.shuffle(buffer_size=5)
for x in dataset_shuffle:
    print(x)
print()

# repeat: 重复数据若干次, 不带参数时, 重复无数次
dataset_17 = tf.data.Dataset.range(3)
dataset_repeat = dataset_17.repeat(3)
for x in dataset_repeat:
    print(x)
print()

# shard:采样, 从某个位置开始隔固定距离采样一个元素
dataset_18 = tf.data.Dataset.range(12)
dataset_shard = dataset_18.shard(3, index=1)
for x in dataset_shard:
    print(x)
print()

# take:采样, 从开始位置取前几个元素
dataset_19 = tf.data.Dataset.range(12)
dataset_take = dataset_19.take(3)
print(list(dataset_take.as_numpy_iterator()))


# -----------------------------
# boost pipe performance
# -----------------------------
# 训练深度学习模型常常会非常耗时,一部分来自数据准备, 一部分来自参数迭代
# 参数迭代过程的耗时通常依赖于 GPU 来提升
# 而数据准备过程的耗时则可以通过构建高效的数据管道进行提升
# 以下是一些构建高效数据管道的建议:
# 1. 使用 prefetch 方法让数据准备和参数迭代两个过程相互并行
# 2. 使用 interleave 方法可以让数据读取过程多进程执行, 并将不同来源数据夹在一起
# 3. 使用 map 时设置 num_parallel_calls 让数据转换过程多进程执行
# 4. 使用 cache 方法让数据在第一个 epoch 后缓存到内存中, 仅限于数据集不大情形
# 5. 使用 map 转换时, 先 batch, 然后采用向量化的转换方法对每个 batch 进行转换
# ----------------------------------------------------------------------------
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


# 数据的准备和参数迭代两个过程默认情况下是串行的
def generator_data(): # 模拟数据准备
    for idx in range(10):
        # 假设每次准备数据需要 2 seconds
        time.sleep(2)
        yield idx

dataset_20 = tf.data.Dataset.from_generator(generator_data, output_types=tf.int32)

def train_step_data(): # 模拟参数迭代
    # 假设每一步训练需要 1 seconds
    time.sleep(1)

# 训练过程预计耗时 10*2 + 10*1 = 30s
tf.print(tf.constant("============ Start Training ============"))

printbar()
for x in dataset_20:
    train_step_data()
printbar()

tf.print(tf.constant("============ End Training ============"))


# 使用 prefetch 方法让数据准备和参数迭代两个过程相互并行
# 训练过程预计耗时 max(10*2, 10*1) = 20s
tf.print(tf.constant("============ Start Training ============"))

printbar()
# tf.data.experimental.AUTOTUNE 可以让程序自动选择合适的参数
for x in dataset_20.prefetch(buffer_size=tf.data.experimental.AUTOTUNE):
    train_step_data()
printbar()

tf.print(tf.constant("============ End Training ============"))


# 使用 interleave 方法可以让数据读取过程多进程执行, 并将不同来源数据夹在一起
dataset_file = tf.data.Dataset.list_files("./titanic/*.csv")
dataset_21 = dataset_file.flat_map(lambda x: tf.data.TextLineDataset(x).skip(1))
for line in dataset_21.take(4):
    print(line)

ds_files = tf.data.Dataset.list_files("./titanic/*.csv")
dataset_22 = ds_files.interleave(lambda x:tf.data.TextLineDataset(x).skip(1))
for line in dataset_22.take(8):
    print(line)

# 使用 map 时设置num_parallel_calls 让数据转换过程多进行执行
dataset_23 = tf.data.Dataset.list_files("./cifar2/train/*/*.jpg")
# 单进程转换
tf.print(tf.constant("======== start transformation ========"))
printbar()
dataset_23_map = dataset_23.map(load_image)
for _ in dataset_23_map:
    pass
printbar()
tf.print(tf.constant("======== ending transformation ========"))

# 多进程转换
tf.print(tf.constant("======== start transformation ========"))
printbar()
dataset_map_parallel = dataset_23.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
for _ in dataset_map_parallel:
    pass
printbar()
tf.print(tf.constant("======== ending transformation ========"))


# 使用 cache 方法让数据在第一个 epoch 后缓存到内存中, 仅限于数据集不大情形
# 模拟数据准备
def generator():
    for i in range(5):
        #假设每次准备数据需要2s
        time.sleep(2) 
        yield i 
dataset_24 = tf.data.Dataset.from_generator(generator, output_types=(tf.int32))

# 模拟参数迭代
def train_step():
    #假设每一步训练需要0s
    pass

# 训练过程预计耗时 (5*2 + 5*0)*3 = 30s
tf.print(tf.constant("start training..."))
printbar()
for epoch in tf.range(3):
    for x in dataset_24:
        train_step()
    printbar()
    tf.print("epoch =", epoch, " ended")
printbar()
tf.print(tf.constant("end training..."))

# 使用 cache 方法让数据在第一个epoch后缓存到内存中，仅限于数据集不大情形。
dataset_25 = tf.data.Dataset.from_generator(generator,output_types = (tf.int32)).cache()
# 训练过程预计耗时 (5*2 + 5*0) + (5*0 + 5*0)*2 = 10s
printbar()
tf.print(tf.constant("start training..."))
for epoch in tf.range(3):
    for x in dataset_25:
        train_step()
    printbar()
    tf.print("epoch =", epoch, " ended")
printbar()
tf.print(tf.constant("end training..."))


# 使用 map 转换时, 先 batch, 然后采用向量化的转换方法对每个 batch 进行转换
dataset_26 = tf.data.Dataset.range(100000)
dataset_map_batch = dataset_26.map(lambda x: x**2).batch(20) # map ---> batch
printbar()
tf.print(tf.constant("start scalar transformation..."))
for x in dataset_map_batch:
    pass
printbar()
tf.print(tf.constant("end scalar transformation..."))

dataset_batch_map = dataset_26.batch(20).map(lambda x: x**2) # batch ---> map
printbar()
tf.print(tf.constant("start scalar transformation..."))
for x in dataset_batch_map:
    pass
printbar()
tf.print(tf.constant("end scalar transformation..."))