#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 12_MorphologicalImageProcessing.py
@Python Version: 3.12.13
@Author: Wei Li (Ithaca)
@Email: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/
@Date: 2026-09-18
@copyright Copyright (c) 2026 Wei Li
@Description: Morphological image processing refers to techniques
that modify the shape and structure of objects in an image
using a defined kernel. It is commonly used for noise removal,
shape refinement, and feature enhancement.
@Paper: I
@Link:
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------
if __name__ == "__main__":
    image = cv2.imread("image/Ganeshji.webp")
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)

    dilated = cv2.dilate(image_gray, kernel, iterations=2)
    eroded = cv2.erode(image_gray, kernel, iterations=2)
    opening = cv2.morphologyEx(image_gray, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(image_gray, cv2.MORPH_CLOSE, kernel)

    fig, axs = plt.subplots(2, 2, figsize=(7, 7))
    axs[0, 0].imshow(dilated, cmap="Greys"), axs[0, 0].set_title("Dilated Image")
    axs[0, 1].imshow(eroded, cmap="Greys"), axs[0, 1].set_title("Eroded Image")
    axs[1, 0].imshow(opening, cmap="Greys"), axs[1, 0].set_title("Opening")
    axs[1, 1].imshow(closing, cmap="Greys"), axs[1, 1].set_title("Closing")

    for ax in axs.flatten():
        ax.set_xticks([]), ax.set_yticks([])

    plt.tight_layout()
    plt.show()
