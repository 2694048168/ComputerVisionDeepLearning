#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: img_transformer.py
@Python Version: 3.11.2
@Author: Wei Li
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2023/05/22 14:35:01
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Description: image geomertry transformer
'''

from PIL import Image
# https://pillow.readthedocs.io/en/stable/reference/Image.html


if __name__ == "__main__":
    filepath = r"./images/split_merge.jpg"
    """
    FLIP_LEFT_RIGHT = 0
    FLIP_TOP_BOTTOM = 1
    ROTATE_180 = 3
    ROTATE_270 = 4
    ROTATE_90 = 2
    TRANSPOSE = 5
    TRANSVERSE = 6
    """
    with Image.open(filepath) as im:
        im.transpose(Image.Transpose.FLIP_LEFT_RIGHT).show()
        im.rotate(45).show()
        im.transform((250,250),
                     Image.Transform.EXTENT,
                     data=[0, 0, 30 + im.width // 4, im.height // 3]).show()

# -----------------------------------------------------------------------------
# https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Transpose
# https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.rotate
# https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.transform