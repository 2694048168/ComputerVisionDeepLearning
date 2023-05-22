#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@File: image_resize.py
@Python Version: 3.11.3
@Author: Wei Li
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Time: 2023/05/21 23:12:33
@Version: 0.1
@License: Apache License Version 2.0, January 2004
@Description: image processing via Pillow library
'''

from PIL import Image


if __name__ == "__main__":
    # Image 类提供的 resize() 方法能够实现任意缩小和放大图像
    img = Image.open("images/image_01155.jpg")
    
    try:
# https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize
        up_size = (img.width * 4, img.height * 4)
        down_size = (img.width // 2, img.height // 2)
        image1 = img.resize(size=up_size, resample=Image.Resampling.LANCZOS)
        image2 = img.resize(size=down_size, resample=Image.Resampling.LANCZOS)
        image1.save("./images/image_01155_upsample.jpg")
        image2.save("./images/image_01155_downsample.jpg")

        # (top_left, bottom_right) rectangle by (w, h) coordates
        box_roi = (255, 150, 440, 328)
        roi_size = (440 - 255, 328 - 150)
        image_roi = img.resize(size=roi_size,
                           resample=Image.Resampling.LANCZOS,
                           box=box_roi,
                           reducing_gap=3.0)
        image_roi.save("./images/image_01155_roi.jpg")

        width, height = img.size
        crop_size = min(width, height)
        # center crop image 中心裁剪保证 H=W
        # ------------------------------------------
        # image copy() | paste() | crop() function
        # ------------------------------------------
        crop_box = ((width - crop_size) // 2, (height - crop_size) // 2,
                    (width + crop_size) // 2, (height + crop_size) // 2)
        # (top_left, bottom_right)
        region_img = img.crop(crop_box)
        region_img.save("./images/image_01155_crop.jpg")

    except IOError:        
        print(f"resize the image failed.")
