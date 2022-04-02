#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: The high-level API of TensorFlow2
@Python Version: 3.8.12
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-03-29
"""

import tensorflow as tf

""" 
TensorFlow 的高阶 API
high-level API of TensorFlow 主要是 tensorflow.keras.models

1. 模型的构建: Sequential、functional API、Model子类化
2. 模型的训练: 内置 fit 方法, 内置 train_on_batch 方法, 自定义训练循环, 单 GPU 训练模型, 多 GPU 训练模型, TPU训练模型
3. 模型的部署: tensorflow serving 部署模型, 使用 spark(scala) 调用 tensorflow 模型

模型的训练方式主要有:
1. 内置 fit method 方法
2. 内置 tran_on_batch method 方法
3. 自定义训练循环

注: fit_generator 方法在 tf.keras 中不推荐使用, 其功能已经被 fit 包含
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


# ----------------------------------------------------------------------------------------
# Reuters dataset 文本分类数据集
# https://paperswithcode.com/dataset/reuters-21578
# https://archive.ics.uci.edu/ml/datasets/reuters-21578+text+categorization+collection
# ----------------------------------------------------------------------------------------
MAX_LEN = 300
BATCH_SIZE = 32

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.reuters.load_data()
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_LEN)

MAX_WORDS = x_train.max()+1
CAT_NUM = y_train.max()+1

ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
          .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
          .prefetch(tf.data.experimental.AUTOTUNE).cache()

ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
          .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
          .prefetch(tf.data.experimental.AUTOTUNE).cache()


# ------------------------------------------------
# way-1 for Training with built-method fit
# 该方法功能非常强大, 支持对 numpy array, tf.data.Dataset 以及 Python generator 数据进行训练
# 并且可以通过设置回调函数 callbacks 实现对训练过程的复杂控制逻辑
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
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.SparseTopKCategoricalAccuracy(5)]) 
    return model

model_fit = create_model()
model_fit.summary()
model_fit = compile_model(model_fit)
history_fit = model_fit.fit(ds_train, validation_data=ds_test, epochs=10)


# ------------------------------------------------
# way-2 for Training with built-method train_on_batch
# 该内置方法相比较 fit 方法更加灵活, 可以不通过回调函数而直接在批次层次上更加精细地控制训练的过程
model_batch = create_model()
model_batch.summary()
model_batch = compile_model(model_batch)

def train_model(model, ds_train, ds_val, epochs):
    for epoch in tf.range(1, epochs + 1):
        model.reset_metrics()

        # 在后期降低学习率
        if epoch == 5:
            model.optimizer.lr.assign(model.optimizer.lr / 2.0)
            tf.print("Lowering optimizer Learning Rate \n\n")

        for x, y in ds_train:
            train_result = model.train_on_batch(x, y)

        for x, y in ds_val:
            valid_result = model.test_on_batch(x, y, reset_metrics=False)

        if epoch % 1 == 0:
            printbar()
            tf.print("epoch = ", epoch)
            print("train:",dict(zip(model.metrics_names, train_result)))
            print("valid:",dict(zip(model.metrics_names, valid_result)))
            print("")

train_model(model_batch, ds_train, ds_test, 10)


# ------------------------------------------------
# way-3 for Training with custom training
# 自定义训练循环无需编译模型,直接利用优化器根据损失函数反向传播迭代参数,拥有最高的灵活性
model_custom = create_model()
model_custom.summary()

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


def train_model(model,ds_train,ds_valid,epochs):
    for epoch in tf.range(1,epochs+1):
        for features, labels in ds_train:
            train_step(model, features,labels)

        for features, labels in ds_valid:
            valid_step(model, features,labels)

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

train_model(model_custom, ds_train, ds_test, 10)