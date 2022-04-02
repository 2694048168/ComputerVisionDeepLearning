#!/usr/bin/env python3
# encoding: utf-8

"""
@Function: Kaggle-街区信号牌识别 单独测试图像
@Python Version: 3.8.12
@Author: Wei Li
@Date: 2022-03-15
"""

import tensorflow as tf
import numpy as np

def predict_model(model, img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [60, 60]) # [H, W, C]
    image = tf.expand_dims(image, axis=0) # # [B, H, W, C]

    predictions = model.predict(image) # [0.001, 0.0004, 0.89, 0.00001, .....]
    predictions = np.argmax(predictions) # 2

    return predictions


if __name__ == "__main__":
    # img_path = r"./archive/Test/2/00409.png"
    img_path = r"./archive/Test/0/03420.png"
    model = tf.keras.models.load_model("./Models")
    prediction = predict_model(model, img_path)
    print(f"prediction = {prediction}")