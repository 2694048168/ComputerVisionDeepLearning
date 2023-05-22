#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: image_filter.py
@Python Version: 3.11.2
@Author: Wei Li
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2023/05/22 14:50:46
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Description: image denoise with filters
'''

from PIL import Image
# https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html
from PIL import ImageFilter


if __name__ == "__main__":
    img = Image.open("./images/panda.jpg")
    im_blur = img.filter(ImageFilter.BLUR)
    im_blur.save("./images/panda_blur.png")

    im2 = img.filter(ImageFilter.CONTOUR)
    im2.save("./images/panda_contour.png")

    im3 = img.filter(ImageFilter.FIND_EDGES)
    im3.save("./images/panda_edges.png")

    im4 = img.filter(ImageFilter.EMBOSS)
    im4.save("./images/panda_emboss.png")

    im5 = img.filter(ImageFilter.SMOOTH)
    im5.save("./images/panda_smooth.png")
