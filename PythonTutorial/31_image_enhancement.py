#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 31_image_enhancement.py
@Python Version: 3.12.13
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2026-10-18
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V0.1
@License: Apache License Version 2.0, January 2004
    Copyright 2026. All rights reserved.

@Description: 低照度图像增强算法-TAGC Image Enhancement
@paper name: Tuning adaptive gamma correction (TAGC) for enhancing images in low ligh
@paper link: https://www.arxiv.org/abs/2507.19574
"""

import os
import pathlib
import glob
import cv2
import numpy as np


def image_enhancement_TAGC(srcImg: cv2.Mat) -> tuple[bool, cv2.Mat]:
    """
    低照度图像增强算法-TAGC Image Enhancement
    :param srcImg: 输入图像，期望为 uint8 的 3 通道 BGR 图像
    :return: (是否成功, 结果图像)
    """
    # 输入参数检查 检查图像类型是否为 8位3通道 (类似 C++ 的 CV_8UC3)
    if (
        srcImg is None
        or srcImg.dtype != np.uint8
        or len(srcImg.shape) != 3
        or srcImg.shape[2] != 3
    ):
        return False, None

    # 转换为浮点数并归一化到 [0, 1]
    src_float = srcImg.astype(np.float32) / 255.0

    # 拆分通道 (注意 OpenCV 默认是 BGR 顺序)
    blue = src_float[:, :, 0]
    green = src_float[:, :, 1]
    red = src_float[:, :, 2]

    # 计算 L 和 A (矩阵点对点运算)
    L = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    A = (blue + green + red) / 3.0

    # 计算自适应 Gamma 值
    Gamma = 5.0 + (0.5 - L) * (1.0 - A) - 2.0 * L

    # 防止 Gamma 为 0 导致除以 0 的异常（可选的安全保护）
    Gamma = np.clip(Gamma, 1e-5, None)

    # 计算 2.0 / Gamma 幂指数
    exponent = 2.0 / Gamma

    # 初始化目标图像并按通道计算功率谱增强
    dstImg = np.empty_like(srcImg)
    # np.clip 用于自动截断到 [0, 255]，.astype(np.uint8) 实现类似 cvRound 的效果
    dstImg[:, :, 0] = np.clip(np.power(blue, exponent) * 255.0 + 0.5, 0, 255)
    dstImg[:, :, 1] = np.clip(np.power(green, exponent) * 255.0 + 0.5, 0, 255)
    dstImg[:, :, 2] = np.clip(np.power(red, exponent) * 255.0 + 0.5, 0, 255)

    return True, dstImg


def main(
    inFolder: str,
    saveFolder: str,
    orientation="H",
    mirror: bool = True,
):
    image_list = []
    for extension in ["png", "jpg", "jpeg", "bmp", "tiff", "gif", "webp"]:
        image_list += glob.glob(os.path.join(inFolder, f"*.{extension}"))
    assert len(image_list), f"there is not any image in the {inFolder}!"

    for file in image_list:
        img = cv2.imread(file, flags=cv2.IMREAD_UNCHANGED)

        _, dstImg = image_enhancement_TAGC(img)

        # 为了让对比更清晰，在图像上绘制标签文本
        # 注意：cv2.putText 不原生支持中文，这里使用英文标签；如需中文可用 Pillow 绘制
        img_src_labeled = img.copy()
        img_dst_labeled = dstImg.copy()

        # 如果开启镜像，使用 cv2.flip 沿 Y 轴（参数 1）翻转右图
        if mirror:
            img_dst_labeled = cv2.flip(img_dst_labeled, 1)

        # 设置文字参数（白色字，带黑色阴影底色提高可见度）
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (255, 255, 255)
        shadow_color = (0, 0, 0)

        # 在左上角标注 "Original" 和 "Enhanced"
        cv2.putText(
            img_src_labeled,
            "Original",
            (20, 40),
            font,
            font_scale,
            shadow_color,
            thickness + 2,
        )
        cv2.putText(
            img_src_labeled, "Original", (20, 40), font, font_scale, color, thickness
        )

        cv2.putText(
            img_dst_labeled,
            "TAGC Enhanced",
            (20, 40),
            font,
            font_scale,
            shadow_color,
            thickness + 2,
        )
        cv2.putText(
            img_dst_labeled,
            "TAGC Enhanced",
            (20, 40),
            font,
            font_scale,
            color,
            thickness,
        )

        # 根据参数进行图像拼接
        if orientation.upper() == "H":
            # 水平拼接 (左右)
            comparison_img = np.hstack((img_src_labeled, img_dst_labeled))
        elif orientation.upper() == "V":
            # 垂直拼接 (上下)
            comparison_img = np.vstack((img_src_labeled, img_dst_labeled))
        else:
            print("错误：未知的拼接方向，请使用 'H' (水平) 或 'V' (垂直)")
            break

        basename = os.path.basename(file)
        cv2.imwrite(os.path.join(saveFolder, basename), comparison_img)
        print(f"--->Finished enhancement processing for the image {basename}.")


# ===========================
if __name__ == "__main__":
    filepath = pathlib.Path(
        "D:/ImageProcessingTeam/ChinaMM2021/eval15/low/test/input/179.png"
    )

    srcImg = cv2.imread(filepath)
    flag, dstImg = image_enhancement_TAGC(srcImg)

    saveFilepath = "./image/179_enhancement.png"
    cv2.imwrite(saveFilepath, dstImg)

    imgFloder = r"D:/ImageProcessingTeam/ChinaMM2021/eval15/low/test/input/"
    main(imgFloder, "./image/")
