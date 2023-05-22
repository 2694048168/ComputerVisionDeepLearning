#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: img_watermark.py
@Python Version: 3.11.2
@Author: Wei Li
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2023/05/22 15:05:18
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Description: 
'''

from PIL import Image
# https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html
# https://pillow.readthedocs.io/en/stable/reference/ImageFont.html
from PIL import ImageFont
from PIL import ImageDraw


font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', size=30)
# https://www.geeksforgeeks.org/python-pillow-creating-a-watermark/
def creating_watermark(im, text, font=font):
    # 给水印添加透明度，因此需要转换图片的格式
    im_rgba = im.convert('RGBA')
    im_text_canvas = Image.new('RGBA', im_rgba.size, (255, 255, 255, 0))
    print(im_rgba.size[0])
    draw = ImageDraw.Draw(im_text_canvas)
    # 设置文本文字大小
    text_x_width, text_y_height = draw.textsize(text, font=font)
    print(text_x_width, text_y_height)
    text_xy = (im_rgba.size[0] - text_x_width, im_rgba.size[1] - text_y_height)
    print(text_xy)
    # 设置文本颜色（绿色）和透明度（半透明）
    draw.text(text_xy, text, font=font, fill=(255,255,255,128))
    # 将原图片与文字复合
    im_text = Image.alpha_composite(im_rgba, im_text_canvas)
    return  im_text


if __name__ == "__main__":
    filepath = r"./images/image_01153.jpg"
    with Image.open(filepath) as image:
        image_water = creating_watermark(image, '@WeiLi')
        image_water.save("./images/image_01153_watermark.png")
