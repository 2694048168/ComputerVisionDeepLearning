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

""" 
特征列用法概述
使用特征列可以将类别特征转换为 one-hot 编码特征, 将连续特征构建分桶特征, 以及对多个特征生成交叉特征等等
要创建特征列, 请调用 tf.feature_column 模块的函数. 该模块中常用的九个函数, 所有九个函数都会返回一个 Categorical-Column 或一个 Dense-Column 对象, 但却不会返回 bucketized_column, 后者继承自这两个类
注意: 所有的 Catogorical Column 类型最终都要通过 indicator_column 转换成 Dense Column 类型才能传入模型

1. numeric_column 数值列, 最常用
2. bucketized_column 分桶列, 由数值列生成, 可以由一个数值列出多个特征, one-hot编码
3. categorical_column_with_identity 分类标识列, one-hot编码, 相当于分桶列每个桶为1个整数的情况
4. categorical_column_with_vocabulary_list 分类词汇列, one-hot编码, 由list指定词典
5. categorical_column_with_vocabulary_file 分类词汇列, 由文件file指定词典
6. categorical_column_with_hash_bucket 哈希列, 整数或词典较大时采用
7. indicator_column 指标列, 由 Categorical Column 生成, one-hot编码
8. embedding_column 嵌入列, 由 Categorical Column 生成, 嵌入矢量分布参数需要学习. 嵌入矢量维数建议取类别数量的 4 次方根.
9. crossed_column 交叉列, 可以由除 categorical_column_with_hash_bucket 的任意分类列构成
"""

# 特征列使用范例, 使用特征列解决 Titanic 生存问题
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


def print_log(info):
    nowtime = datetime.datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")
    print(f"\n ========================================= {nowtime}")
    print(f"[INFO]----> {info}\n\n")

# ==============================
# 1. 构建数据管道
# ==============================
print_log("Step1: prepare dataset")
df_train_raw = pd.read_csv("./titanic/train.csv")
df_test_raw = pd.read_csv("./titanic/test.csv")
df_raw = pd.concat([df_train_raw, df_test_raw])

def prepare_df_data(df_raw):
    df_data = df_raw.copy()
    df_data.columns = [x.lower() for x in df_data.columns]
    df_data = df_data.rename(columns={'survived': 'label'})
    df_data = df_data.drop(['passengerid', 'name'], axis=1)
    for col, dtype in dict(df_data.dtypes).items():
        # 判断是否包含缺失值
        if df_data[col].hasnans:
            # 添加标识是否缺失列
            df_data[col + '_nan'] = pd.isna(df_data[col]).astype('int32')
            # 填充
            if dtype not in [np.object, np.str, np.unicode]:
                df_data[col].fillna(df_data[col].mean(), inplace=True)
            else:
                df_data[col].fillna('', inplace=True)
    return df_data

df_data = prepare_df_data(df_raw)
df_train = df_data.iloc[0:len(df_train_raw), :]
df_test = df_data.iloc[len(df_train_raw):, :]

# 从 dataframe 导入数据 
def df_to_dataset(df, shuffle=True, batch_size=32):
    df_data = df.copy()
    if 'label' not in df_data.columns:
        dataset_titanic = tf.data.Dataset.from_tensor_slices(df_data.to_dict(orient='list'))
    else: 
        labels = df_data.pop('label')
        dataset_titanic = tf.data.Dataset.from_tensor_slices((df_data.to_dict(orient='list'), labels))  
    if shuffle:
        dataset_titanic = dataset_titanic.shuffle(buffer_size=len(df_data))
    dataset_titanic = dataset_titanic.batch(batch_size)

    return dataset_titanic

ds_train = df_to_dataset(df_train)
ds_test = df_to_dataset(df_test)


# ==========================
# 2. 定义特征列
# ==========================
print_log("Step2: make feature columns")
feature_columns = []
# 数值列
for col in ['age','fare','parch','sibsp'] + [
    c for c in df_data.columns if c.endswith('_nan')]:
    feature_columns.append(tf.feature_column.numeric_column(col))

# 分桶列
age = tf.feature_column.numeric_column('age')
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# 类别列
# 注意: 所有的 Catogorical Column 类型最终都要通过 indicator_column 转换成 Dense Column 类型才能传入模型
sex = tf.feature_column.indicator_column(
      tf.feature_column.categorical_column_with_vocabulary_list(
      key='sex',vocabulary_list=["male", "female"]))
feature_columns.append(sex)

pclass = tf.feature_column.indicator_column(
      tf.feature_column.categorical_column_with_vocabulary_list(
      key='pclass',vocabulary_list=[1,2,3]))
feature_columns.append(pclass)

ticket = tf.feature_column.indicator_column(
     tf.feature_column.categorical_column_with_hash_bucket('ticket',3))
feature_columns.append(ticket)

embarked = tf.feature_column.indicator_column(
      tf.feature_column.categorical_column_with_vocabulary_list(
      key='embarked',vocabulary_list=['S','C','B']))
feature_columns.append(embarked)

# 嵌入列
cabin = tf.feature_column.embedding_column(
    tf.feature_column.categorical_column_with_hash_bucket('cabin',32),2)
feature_columns.append(cabin)

# 交叉列
pclass_cate = tf.feature_column.categorical_column_with_vocabulary_list(
          key='pclass',vocabulary_list=[1,2,3])

crossed_feature = tf.feature_column.indicator_column(
    tf.feature_column.crossed_column([age_buckets, pclass_cate],hash_bucket_size=15))

feature_columns.append(crossed_feature)


# ========================
# 3. 定义模型
# ========================
print_log("Step3: define model")
tf.keras.backend.clear_session()

model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(feature_columns), # 将特征列放入到 tf.keras.layers.DenseFeatures 中
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])


# ===================
# 4. 训练模型
# ===================
print_log("Step4: train model")
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(ds_train,
          validation_data=ds_test,
          epochs=10)


# ====================
# 5.评估模型
# ====================
print_log("Step5: eval model")
model.summary()

def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title(f'Training and validation {metric}')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([f"train_{metric}", f'val_{metric}'])
    plt.show()
    plt.close()

plot_metric(history, "accuracy")
plot_metric(history, "loss")