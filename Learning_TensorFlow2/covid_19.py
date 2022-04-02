#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: TensorFlow2 for time series analysis
@Brief: 新冠肺炎病毒 时间序列分析和建模
@Dataset: https://gitee.com/Python_Ai_Road/eat_tensorflow2_in_30_days/tree/master/data/covid-19.csv
@Dataset: https://github.com/lyhue1991/eat_tensorflow2_in_30_days/tree/master/data/covid-19.csv
@Python Version: 3.8.12
@Author: Wei Li (weili_yzzcq@163.com)
@Date: 2022-03-20
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

import os, pathlib
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Stage 1. Dataset Processing
# ==========================================
path2cvs_file = r"./covid-19.csv"

df_data_raw = pd.read_csv(pathlib.Path(path2cvs_file), sep="\t")
df_data_raw.plot(x="date", y=["confirmed_num", "cured_num", "dead_num"])
plt.xticks(rotation=60)
# plt.show()
plt.close()

df_data = df_data_raw.set_index("date")
df_diff = df_data.diff(periods=1).dropna()
df_diff = df_diff.reset_index("date")

df_diff.plot(x="date", y=["confirmed_num", "cured_num", "dead_num"])
plt.xticks(rotation=60)
# plt.show()
plt.close()
df_diff = df_diff.drop("date", axis=1).astype("float32")

# 用某日前 8 天窗口数据作为输入预测该日数据
WINDOW_SIZE = 8

def batch_dataset(dataset):
    dataset_batched = dataset.batch(WINDOW_SIZE, drop_remainder=True)
    
    return dataset_batched

ds_data = tf.data.Dataset.from_tensor_slices(tf.constant(df_diff.values, dtype = tf.float32)).window(WINDOW_SIZE, shift=1).flat_map(batch_dataset)

ds_label = tf.data.Dataset.from_tensor_slices(tf.constant(df_diff.values[WINDOW_SIZE:], dtype = tf.float32))

# 数据较小，可以将全部训练数据放入到一个 batch 中，提升性能
ds_train = tf.data.Dataset.zip((ds_data, ds_label)).batch(38).cache()

# Stage 2. Define Models
# ==========================================
class Block(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Block, self).__init__(**kwargs)

    def call(self, x_input, x):
        x_out = tf.maximum((1+x) * x_input[:, -1, :], 0.0)

        return x_out

    def get_config(self):
        config = super(Block, self).get_config()

        return config

tf.keras.backend.clear_session()
x_input = tf.keras.layers.Input(shape=(None,3), dtype=tf.float32)
x = tf.keras.layers.LSTM(3, return_sequences = True, input_shape=(None,3))(x_input)
x = tf.keras.layers.LSTM(3, return_sequences = True, input_shape=(None,3))(x)
x = tf.keras.layers.LSTM(3, return_sequences = True, input_shape=(None,3))(x)
x = tf.keras.layers.LSTM(3, input_shape=(None,3))(x)
x = tf.keras.layers.Dense(3)(x)

#考虑到新增确诊，新增治愈，新增死亡人数数据不可能小于 0，设计如下结构
# x = tf.maximum((1+x) * x_input[:,-1,:], 0.0)
x = Block()(x_input,x)
model = tf.keras.models.Model(inputs=[x_input], outputs=[x])
model.summary()
        

# Stage 3. Training Models
# ==========================================
#自定义损失函数，考虑平方差和预测目标的比值
class MSPE(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(MSPE, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        err_percent = (y_true - y_pred)**2 / (tf.maximum(y_true**2, 1e-7))
        mean_err_percent = tf.reduce_mean(err_percent)

        return mean_err_percent

    def get_config(self):
        config = super(MSPE, self).get_config()

        return config

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss=MSPE(name="MSPE"))

path2logs_folder = r"./tensorboard/covid"
os.makedirs(path2logs_folder, exist_ok=True)

stamp_time = datetime.datetime.now().strftime(r"%Y%m%d-%H%M%S")
log_folder = pathlib.Path(os.path.join(path2logs_folder, stamp_time))

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_folder, histogram_freq=1)

# 如果 loss 在 100 epochs 后没有提升，学习率减半
learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=100)

# 当 loss 在 200 epochs 后没有提升，则提前终止训练
stop_early_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=200)

history = model.fit(ds_train,
    epochs=500, 
    callbacks=[tensorboard_callback, learning_rate_callback, stop_early_callback])


# Stage 4. Evaluate Models
# ==========================================
path2img_save = r"./images"
os.makedirs(path2img_save, exist_ok=True)

def plot_metric(history, metric):
    train_metrics = history.history[metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.title('Training '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric])
    # plt.show()
    plt.savefig(pathlib.Path(os.path.join(path2img_save, f"covid_{metric}.png")), dpi=120)
    plt.close()

plot_metric(history, "loss")


# Stage 5. Using Modles
# ==========================================
# Stage 6. Saving Models and Load Models
# ==========================================
path2model_save = r"./Models/covid_19"
os.makedirs(path2model_save, exist_ok=True)

model.save(pathlib.Path(path2model_save), save_format="tf")
print(f"The Trained Model Save at \033[1;33;40m {path2model_save} \033[0m with TensorFlow Format Successfully.")

print("-------------------------------------------------------")
model_loaded = tf.keras.models.load_model(pathlib.Path(path2model_save))
model_loaded.compile(optimizer=optimizer, loss=MSPE(name="MSPE"))
model_loaded.predict(ds_train)