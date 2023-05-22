#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: basic_operators.py
@Python Version: 3.11.3
@Author: Wei Li
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2023/05/21 23:06:46
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Description: the Pillow image processing library with Python
'''

from PIL import Image
import numpy as np
import torch
import torchvision


if __name__ == "__main__":
    filepath = r"./images/image_01153.jpg"
    img = Image.open(filepath, mode="r")
    img.show()

    size_resolution = (256, 256) # (width, height)
    img_new = Image.new(mode="RGB", size=size_resolution, color="#ff0000")
    img_new.show()

    # Pillow Image object attrs.
    print(f"the Pillow Image object {img}")
    print(f"the type of Pillow Image object {type(img)}")
    print(f"the resolution(W,H) of image {(img.width, img.height)}")
    print(f"the size(W,H) of image {img.size}")
    print(f"the format of image {img.format}")
    print(f"the readonly for this image {img.readonly}")
    print(f"the information for this image {img.info}")
    print(f"the mode for this image {img.mode}")

    """ Pillow Image Mode:
    -------------------------------------------------------
    mode	描述
    1	    1 位像素(取值范围 0-1), 0表示黑, 1表示白, 单色通道
    L	    8 位像素(取值范围 0 -255), 灰度图, 单色通道
    P	    8 位像素, 使用调色板映射到任何其他模式, 单色通道
    RGB	    3 x 8位像素, 真彩色, 三色通道, 每个通道的取值范围 0-255
    RGBA	4 x 8位像素, 真彩色+透明通道, 四色通道
    CMYK	4 x 8位像素, 四色通道, 可以适应于打印图片
    YCbCr	3 x 8位像素, 彩色视频格式, 三色通道
    LAB	    3 x 8位像素, L * a * b 颜色空间, 三色通道
    HSV	    3 x 8位像素, 色相, 饱和度, 值颜色空间, 三色通道
    I	    32 位有符号整数像素, 单色通道
    F	    32 位浮点像素, 单色通道
    -------------------------------------------------------
    """

    filename1 = r"./images/image_01153_save.png"
    filename2 = r"./images/image_01153_save.bmp"
    img.save(fp=filename1) # why not? channels
    img.save(fp=filename2, format="bmp")

    img_rgb = Image.open("images/image_01153_save.bmp")
    filename3 = r"./images/image_01153_save2.png"
    # img_rgb.save(filename3)
    img_png = img.convert(mode="RGBA") # mode different?
    img_png.save(filename3)

    # -------------------------------------------------------
    """
    # Step 1. torch.Tensor -> PIL.Image.
    tensor = torch.rand([3, 256, 256])
    image = Image.fromarray(torch.clamp(tensor * 255, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy())
    # Equivalently way as follow
    image = torchvision.transforms.functional.to_pil_image(tensor)

    # Step 2. PIL.Image -> torch.Tensor.
    tensor = torch.from_numpy(np.asarray(Image.open(filepath))).permute(2, 0, 1).float() / 255
    # Equivalently way as follow
    tensor = torchvision.transforms.functional.to_tensor(Image.open(filepath))

    # Step 3. np.ndarray -> PIL.Image.
    image = Image.fromarray(np.ndarray.astypde(np.uint8))

    # Step 4. PIL.Image -> np.ndarray.
    ndarray = np.asarray(Image.open(filepath))
    """
    # -------------------------------------------------------
