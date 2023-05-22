#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: image_gif.py
@Python Version: 3.11.2
@Author: Wei Li
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2023/05/22 15:24:52
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Description: 
'''

import os
import glob
from PIL import Image

"""
GIF(Graphics Interchange Format,图形交换格式)是一种位图图像格式, 以.gif作为图像的扩展名,
GIF图片采用图像预压缩技术,在一定程度上减少了图像传播、加载所消耗的时间;
GIF还有一项非常重要的应用就是生成动态图, Pillow可以将静态格式图片(png、jpg)合成为GIF动态图.

注意: Pillow 总是以灰度模式 L 或调色板模式 P 来读取GIF文件.
"""
def generate_gif(image_folder, gif_name):
    image_list = []
    for extension in ["png", "jpg", "jpeg", "bmp"]:
        image_list += glob.glob(os.path.join(image_folder, f"*.{extension}"))
    assert len(image_list), f"there is not any image in the {image_folder}!"

    frames = list()
    for frame_file in image_list:
        frame = Image.open(frame_file)
        frames.append(frame)
    
    # 以第一张图片作为开始，将后续5张图片合并成 gif 动态图
    # 参数说明：
    # save_all 保存图像;
    # transparency 设置透明背景色;
    # duration 单位毫秒，动画持续时间， 
    # loop=0 无限循环;disposal=2 恢复原背景颜色
    frames[0].save(gif_name, save_all=True, 
                   append_images=frames[1:],
                   transparency=0,
                   duration=2000,
                   loop=0,
                   disposal=2)


if __name__ == "__main__":
    img_folder = r"./images/"
    savepath = r"./images/generate.gif"
    generate_gif(img_folder, savepath)
