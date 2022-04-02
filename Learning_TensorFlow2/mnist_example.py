#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 机器学习(Machine Learning)-深度学习(Deep Learning)入门之手写体数字识别(MNIST dataset)
@Python Version: 3.8.12
@Author: Wei Li
@Date: 2022-03-12
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# 1. tensorflow.keras.Sequential
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.Conv2D(64, (3, 3), activation= "relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ]
)

# 2. functional approach : function that returns a model
def functional_model():
    input = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(input)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs=input, outputs=x)

    return model

# 3. tensorflow.keras.Model : inherit from this class
class CustomModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.convolution1 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")
        self.convolution2 = tf.keras.layers.Conv2D(64, (3, 3), activation= "relu")
        self.maxpool1 = tf.keras.layers.MaxPool2D()
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        self.convolution3 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu")
        self.maxpool2 = tf.keras.layers.MaxPool2D()
        self.batchnorm2 = tf.keras.layers.BatchNormalization()

        self.globalavgagepool = tf.keras.layers.GlobalAvgPool2D()
        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dense2 = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, input):
        x = self.convolution1(input)
        x = self.convolution2(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)
        x = self.convolution3(x)
        x = self.batchnorm2(x)
        x = self.globalavgagepool(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x


def display_examples(examples, labels, num):
    """Show some examples in MNIST dataset.

    Args:
        examples (numpy.ndarray): [batch_num, H, W] for gray images
        labels (numpy.ndarray): [batch_num,] for the corresponding digit to images
    """
    plt.figure(figsize=(10, 10))

    for i in range(num):
        idx_img = np.random.randint(0, examples.shape[0] - 1)
        image = examples[idx_img]
        label = labels[idx_img]

        plt.subplot(int(np.sqrt(num)), int(np.sqrt(num)), i+1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(image, cmap="gray")
        # plt.imshow(image)

    plt.show()


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # print(f"x_train shape = {x_train.shape} and x_train type and data type = {type(x_train)} and {x_train.dtype}")
    # print(f"y_train shape = {y_train.shape} and y_train type and data type = {type(y_train)} and {y_train.dtype}")
    # print(f"x_test shape = {x_test.shape} and x_test type and data type = {type(x_test)} and {x_test.dtype}")
    # print(f"y_test shape = {y_test.shape} and y_test type and data type = {type(y_test)} and {y_test.dtype}")

    if False:
        display_examples(x_train, y_train, 25)
        # display_examples(x_test, y_test, 25)
    
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # One-hot encode for classification
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    # print(f"x_train shape = {x_train.shape} and x_train type and data type = {type(x_train)} and {x_train.dtype}")
    # print(f"y_train shape = {y_train.shape} and y_train type and data type = {type(y_train)} and {y_train.dtype}")
    # print(f"x_test shape = {x_test.shape} and x_test type and data type = {type(x_test)} and {x_test.dtype}")
    # print(f"y_test shape = {y_test.shape} and y_test type and data type = {type(y_test)} and {y_test.dtype}")

    # step 3. 
    model = CustomModel()
    # step 2. 
    # model = functional_model()
    # step 1. 
    # model.summary()
    model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(), metrics="accuracy")
    # model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics="accuracy")

    model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2)

    model.evaluate(x_test, y_test, batch_size=64)