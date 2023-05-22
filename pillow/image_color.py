#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: image_color.py
@Python Version: 3.11.2
@Author: Wei Li
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2023/05/22 15:00:42
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Description: 
'''

from PIL import Image
# https://pillow.readthedocs.io/en/stable/reference/ImageColor.html
from PIL import ImageColor


if __name__ == "__main__":
    # ----------- getrgb() -----------
    color1 = ImageColor.getrgb("blue")
    print(color1)

    color2 = ImageColor.getrgb('#DCDCDC')
    print(color2)

    # 使用HSL模式红色
    color3 = ImageColor.getrgb('HSL(0,100%,50%)')
    print(color3)

    # ----------- getcolor() -----------
    color4=ImageColor.getcolor('#EEA9B8','L')
    print(color4)

    color5=ImageColor.getcolor('yellow','RGBA')
    print(color5)
    