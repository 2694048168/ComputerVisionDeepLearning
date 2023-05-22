#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: img_split_merge.py
@Python Version: 3.11.3
@Author: Wei Li
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2023/05/22 00:00:34
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Description: split and merger image channels via Pillow library
'''

from PIL import Image


if __name__ == "__main__":
    img = Image.open("./images/split_merge.jpg")
    if img.mode == "RGB":
        r, g, b = img.split()
    elif img.mode == "RGBA":
        r, g, b, a = img.split()
    else:
        print(f"this image not multi-channels, maybe single grayscale image.")

    img_merge = Image.merge(mode="RGB", bands=[b, g, r])
    img_merge.save("./images/split_merge_bgr.jpg")

    # ------- merge and blend methods -------
    sun_img = Image.open("./images/sunflower.jpg")
    # 两张图片的合并要求两张图片的模式、图像大小必须要保持一致
    sun_image = sun_img.resize(img.size)
    r_sun, g_run, b_sun = sun_image.split()
    merge_sun = Image.merge("RGB", [r_sun, g, b_sun])
    merge_sun.save("./images/sunflower_merge.png")

    # Image 类提供 blend() 方法来混合 RGBA 模式(PNG)
    img1 = Image.open("./images/image_01153_save.png")
    img2 = Image.open("./images/sunflower_merge.png")
    img2 = img2.resize(img1.size)
    # blend = (1 - alpha) * img1 + alpha * img2
    Image.blend(img1, img2, alpha=0.4).save("./images/sunflower_blend.png")
