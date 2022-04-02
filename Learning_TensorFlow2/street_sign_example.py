#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: 机器学习(Machine Learning)-深度学习(Deep Learning)之Kaggle-街区信号牌识别
@Dataset: GTSRB - German Traffic Sign Recognition Benchmark
@Site: https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/code?datasetId=82373
@Python Version: 3.8.12
@Author: Wei Li
@Date: 2022-03-15
"""

import os
import glob
from sklearn.model_selection import train_test_split
import shutil
import csv
import tensorflow as tf


def split_data(path2data, path2train_save, path2val_save, split_size=0.1):
    folders = os.listdir(path2data)
    for folder in folders:
        full_path = os.path.join(path2data, folder)
        images_path = glob.glob(os.path.join(full_path, "*.png"))

        x_train, x_val = train_test_split(images_path, test_size=split_size)
        for x in x_train:
            # basename = os.path.basename(x)
            path2foler = os.path.join(path2train_save, folder)
            if not os.path.isdir(path2foler):
                os.makedirs(path2foler)

            shutil.copy(x, path2foler)
        for x in x_train:
            # basename = os.path.basename(x)
            path2foler = os.path.join(path2train_save, folder)
            
            # os.makedirs(path2foler, exist_ok=True)
            if not os.path.isdir(path2foler):
                os.makedirs(path2foler)

            shutil.copy(x, path2foler)

        for x in x_val:
            # basename = os.path.basename(x)
            path2foler = os.path.join(path2val_save, folder)

            os.makedirs(path2foler, exist_ok=True)
            # if not os.path.isdir(path2foler):
            #     os.makedirs(path2foler)

            shutil.copy(x, path2foler)

def order_testset(path2images, path2csv):
    try:
        with open(path2csv, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for i, row in enumerate(reader):
                if i == 0:
                    continue

                img_name = row[-1].replace("Test/", "")
                label = row[-2]

                path2folder = os.path.join(path2images, label)
                os.makedirs(path2folder, exist_ok=True)

                img_full_path = os.path.join(path2images, img_name)
                shutil.move(img_full_path, path2folder)

        print(f"[INFO] : Testset is making Successfully at {path2images}")

    except:
        print(f"[INFO] : Error reading CSV file at {path2csv}")


def create_generators(batch_size, train_data_path, val_data_path, test_data_path):
    train_preprocessor =  tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1 / 255.,
        rotation_range = 10,
        width_shift_range = 0.1)

    test_preprocessor =  tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1 / 255.)

    train_generator = train_preprocessor.flow_from_directory(
        train_data_path,
        class_mode = "categorical",
        target_size = (60, 60),
        color_mode = "rgb",
        shuffle = True,
        batch_size = batch_size
    )

    val_generator = test_preprocessor.flow_from_directory(
        val_data_path,
        class_mode = "categorical",
        target_size = (60, 60),
        color_mode = "rgb",
        shuffle = True,
        batch_size = batch_size
    )

    test_generator = test_preprocessor.flow_from_directory(
        test_data_path,
        class_mode = "categorical",
        target_size = (60, 60),
        color_mode = "rgb",
        shuffle = True,
        batch_size = batch_size
    )

    return train_generator, val_generator, test_generator


def streetsingns_model(num_classes):
    input = tf.keras.layers.Input(shape=(60, 60, 3))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(input)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs=input, outputs=x)


if __name__ == "__main__":
    # python street_sign_example.py | tee log.txt  
    
    # Training data
    if False:
        path2data = r"./archive/Train"
        path2train_save = r"./archive/training_data/train"
        path2val_save = r"./archive/training_data/val"

        split_data(path2data, path2train_save, path2val_save)

    # Testing data
    if False:
        print(os.getcwd())
        path2images = r"./archive/Test"
        path2csv = r"./archive/Test.csv"
        order_testset(path2images, path2csv)

    # training on GPU
    path2train = r"./archive/training_data/train"
    path2val = r"./archive/training_data/val"
    path2test = r"./archive/Test"

    batch_size = 64
    epochs = 10
    train_generator, val_generator, test_generator = create_generators(batch_size, path2train, path2val, path2test)
    num_classes = train_generator.num_classes

    TRAIN = False
    TEST = True

    if TRAIN:
        path2save_model = r"./Models"
        os.makedirs(path2save_model, exist_ok=True)
        ckpt_saver_callback = tf.keras.callbacks.ModelCheckpoint(
            path2save_model,
            monitor = "val_accuracy",
            mode = "max",
            save_best_only = True,
            save_freq = "epoch",
            verbose = 1
        )

        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor = "val_accuracy", patience = 10)

        model = streetsingns_model(num_classes)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_generator,
            epochs = epochs,
            batch_size = batch_size,
            validation_data = val_generator,
            callbacks=[ckpt_saver_callback, early_stop_callback])

    if TEST:
        model = tf.keras.models.load_model("./Models")
        model.summary()

        print("Evaluating validation dataset")
        model.evaluate(val_generator)

        print("Evaluating test dataset")
        model.evaluate(test_generator)